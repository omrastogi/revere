"""Phase 2: align frozen V-JEPA 2 clip embeddings with UCA sentence text embeddings, plus a GREEN/YELLOW/RED zone head.

python align.py text           # login node: OpenAI text-embedding-3-small for every UCA sentence (needs OPENAI_API_KEY)
python align.py clips          # GPU: one 16-frame clip per sentence -> V-JEPA 2 ViT-L, mean-pooled 1024-d; run on any
                               #   number of GPU jobs at once, each claims vjepa_emb/part_XX via lock files
python align.py train          # GPU: MLP head, symmetric InfoNCE + 0.5 * weighted zone CE, early stop on val R@5
python align.py eval           # retrieval, zone metrics, 40 qualitative examples, linear zone baseline
All paths are relative to the working directory: uca_labels/, text_emb/, vjepa_emb/, heads/, results/.
"""
import json, math, os, sys, time, urllib.error, urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H5 = os.environ.get("UCF_H5", "/projects/adsum-mllm-om/data/ucf_crime_h5")  # Dropbox mp4s -> shard_*.h5 + index.csv
LABELS = "uca_labels/sentence_labels.jsonl"
DICT = "uca_labels/dictionary.jsonl"  # standard concept dictionary (uca_pipeline.py stage 3): inference-time vocabulary
ZONES = ("GREEN", "YELLOW", "RED")
MODEL = "facebook/vjepa2-vitl-fpc64-256"
T, S, PARTS = 16, 256, 48  # frames per clip, encoder input size, work units for the clip jobs
CLAIM_MIN = float(os.environ.get("CLAIM_MIN", 45))  # stop claiming new parts after this (sharing partition: 1 h limit)
BATCH, EPOCHS, PATIENCE, VAL_FRAC = 2048, 50, 10, 0.1


def read_rows():
    return [r for r in map(json.loads, open(LABELS)) if r.get("zone") in ZONES and r.get("text", "").strip()]


def log_time(step, seconds, **extra):
    os.makedirs("results", exist_ok=True)
    p = "results/phase2_times.json"
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[step] = dict(seconds=round(seconds, 1), **extra)
    json.dump(d, open(p, "w"), indent=1)


# ---------------------------------------------------------------- STEP 1: text embeddings
def embed_api(texts):
    body = json.dumps({"model": "text-embedding-3-small", "input": texts}).encode()
    for a in range(6):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/embeddings", body, {
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"})
            data = json.load(urllib.request.urlopen(req, timeout=120))["data"]
            return np.array([e["embedding"] for e in sorted(data, key=lambda e: e["index"])], np.float32)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            print(f"  retry {a + 1}/6: {e}", flush=True)
            time.sleep(2 ** a)
    raise RuntimeError("embedding API failed 6 times")


def cmd_text():
    t0 = time.time()
    rows = read_rows()
    os.makedirs("text_emb/cache", exist_ok=True)
    have = {}  # cache: one npz per finished API call, keyed by the strings it holds, so reruns skip them
    for f in sorted(os.listdir("text_emb/cache")):
        z = np.load(f"text_emb/cache/{f}")
        have.update(zip(z["texts"].tolist(), z["emb"]))
    concepts = list(map(json.loads, open(DICT))) if os.path.exists(DICT) else []
    pairs = [(c["id"], s) for c in concepts for s in dict.fromkeys([c["phrase"], *c.get("synonyms", [])]) if s.strip()]
    uniq = sorted({r["text"] for r in rows} | {s for _, s in pairs})
    todo = [t for t in uniq if t not in have]
    print(f"{len(rows)} sentences, {len(concepts)} dictionary concepts ({len(pairs)} phrases+synonyms), "
          f"{len(uniq)} unique strings, {len(todo)} to embed", flush=True)
    for i in range(0, len(todo), 1000):
        b = todo[i:i + 1000]
        e = embed_api(b)
        np.savez(f"text_emb/cache/{time.time_ns()}.npz", texts=np.array(b), emb=e)
        have.update(zip(b, e))
        print(f"  {i + len(b)}/{len(todo)}", flush=True)
    E = np.stack([have[r["text"]] for r in rows])
    np.save("text_emb/sentences.npy", (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32))
    np.save("text_emb/sentences_ids.npy", np.array([r["id"] for r in rows]))
    if pairs:  # one row per phrase/synonym; dictionary_ids.json says which concept each row belongs to
        E = np.stack([have[s] for _, s in pairs])
        np.save("text_emb/dictionary.npy", (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32))
        json.dump(pairs, open("text_emb/dictionary_ids.json", "w"))
    log_time("text", time.time() - t0, sentences=len(rows), unique=len(uniq), api_calls=math.ceil(len(todo) / 1000))


# ---------------------------------------------------------------- STEP 2: one clip per sentence
def h5_index():
    """{video_id: index.csv row (shard, n_frames, split, src_fps, ...)}; frames are stored at src_fps / 4."""
    import csv
    return {r["vid"]: r for r in csv.DictReader(open(f"{H5}/index.csv"))}


def clip_positions(n, start, end, fps):
    """T frame positions (0..n-1) for a [start, end] s span: uniform; spans shorter than T are widened around their center
    (shifted to stay inside the video); videos shorter than T repeat frames."""
    a = min(start * fps, n - 1)
    b = min(max(end, start) * fps, n - 1)
    w = max(b - a, min(T - 1, n - 1))
    a = min(max((a + b) / 2 - w / 2, 0), n - 1 - w)
    return np.round(np.linspace(a, a + w, T)).astype(int)


class Clips(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items, self.files = items, {}  # [(shard file, video id, frame positions)]; h5 handles opened per worker

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import h5py
        shard, vid, pos = self.items[i]
        if shard not in self.files:
            self.files[shard] = h5py.File(f"{H5}/{shard}", "r", rdcc_nbytes=64 << 20)
        u, inv = np.unique(pos, return_inverse=True)  # h5py needs increasing unique indices
        return torch.from_numpy(self.files[shard][vid]["frames"][u][inv])  # (T, 224, 224, 3) uint8 RGB


class Encoder:
    def __init__(self, dev, dtype=None):
        from transformers import AutoModel, AutoVideoProcessor
        vp = AutoVideoProcessor.from_pretrained(MODEL)
        self.dtype = dtype or (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)  # V100: no bf16
        self.mean = torch.tensor(vp.image_mean, device=dev).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(vp.image_std, device=dev).view(1, 1, 3, 1, 1)
        self.model = AutoModel.from_pretrained(MODEL, dtype=self.dtype).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, x):  # (B, T, H, W, 3) uint8 -> (B, 1024) float32
        b = x.shape[0]
        x = x.to(self.dev, non_blocking=True).permute(0, 1, 4, 2, 3).flatten(0, 1).float() / 255
        x = F.interpolate(x, size=(S, S), mode="bicubic", align_corners=False).clamp(0, 1).view(b, T, 3, S, S)
        x = ((x - self.mean) / self.std).to(self.dtype)
        return self.model(pixel_values_videos=x, skip_predictor=True).last_hidden_state.float().mean(1).cpu()


def clip_rows():
    idx = h5_index()
    rows = read_rows()
    missing = sorted({r["video_id"] for r in rows if r["video_id"] not in idx})
    rows = sorted((r for r in rows if r["video_id"] in idx), key=lambda r: (idx[r["video_id"]]["shard"], r["video_id"], r["start"]))
    return idx, rows, missing


def cmd_clips():
    """Claim parts (contiguous blocks of sentences, grouped by shard) via lock files until none are left, so any number of
    jobs on any GPU type share the work. ponytail: a job killed mid-part leaves a stale .lock; delete it and rerun."""
    t0 = time.time()
    idx, rows, missing = clip_rows()
    print(f"{len(rows)} sentences from {len({r['video_id'] for r in rows})} videos; "
          f"{len(missing)} labelled videos not in the h5: {missing[:10]}", flush=True)
    os.makedirs("vjepa_emb", exist_ok=True)
    bounds = np.linspace(0, len(rows), PARTS + 1).astype(int)
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
    enc = None
    for k in range(PARTS):
        out = f"vjepa_emb/part_{k:02d}.npz"
        if time.time() - t0 > CLAIM_MIN * 60:
            break
        try:
            os.close(os.open(out + ".lock", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        enc = enc or Encoder("cuda")
        sel = rows[bounds[k]:bounds[k + 1]]
        items = []
        for r in sel:
            v = idx[r["video_id"]]
            items.append((v["shard"], r["video_id"], clip_positions(int(v["n_frames"]), r["start"], r["end"], float(v["src_fps"]) / 4)))
        dl = torch.utils.data.DataLoader(Clips(items), batch_size=64, num_workers=workers, pin_memory=True, prefetch_factor=4)
        t1 = time.time()
        X = torch.cat([enc(x) for x in dl]).numpy()
        secs = time.time() - t1
        with open(out + ".tmp", "wb") as f:
            np.savez(f, X=X, ids=np.array([r["id"] for r in sel]), secs=secs, gpu=torch.cuda.get_device_name())
        os.replace(out + ".tmp", out)
        print(f"part {k:02d}: {len(sel)} clips in {secs:.0f}s = {len(sel) / secs:.0f} clips/s on {torch.cuda.get_device_name()}",
              flush=True)


# ---------------------------------------------------------------- STEP 3: alignment head
class Head(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.body = nn.Sequential(nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.1))
        self.align = nn.Linear(1024, 1536)  # into the text-embedding-3-small space
        self.zone = nn.Linear(1024, len(ZONES))
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))

    def forward(self, x):
        h = self.body((x - self.mean) / self.std)
        return F.normalize(self.align(h), dim=-1), self.zone(h)


def load(split, dev):
    parts = [np.load(f"vjepa_emb/part_{k:02d}.npz") for k in range(PARTS)]  # raises if a clip part is missing
    X, ids = np.concatenate([p["X"] for p in parts]), np.concatenate([p["ids"] for p in parts])
    rows, idx = {r["id"]: r for r in read_rows()}, h5_index()
    keep = np.array([idx[rows[i]["video_id"]]["split"] == split for i in ids])
    X, ids = X[keep], ids[keep]
    pos = {k: i for i, k in enumerate(np.load("text_emb/sentences_ids.npy"))}
    te = np.load("text_emb/sentences.npy")[[pos[i] for i in ids]]
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    return dict(X=t(X), Y=t(te), Z=t([ZONES.index(rows[i]["zone"]) for i in ids], torch.long),
                vid=np.array([rows[i]["video_id"] for i in ids]), text=[rows[i]["text"] for i in ids], ids=ids)


def sub(d, m):  # rows of every field where the numpy bool mask m is set
    pick = lambda v: (v[torch.from_numpy(m).to(v.device)] if torch.is_tensor(v) else
                      v[m] if isinstance(v, np.ndarray) else [x for x, keep in zip(v, m) if keep])
    return {k: pick(v) for k, v in d.items()}


def info_nce(c, t, tau):
    """Symmetric InfoNCE; other sentences with (near-)identical text (cos > 0.95) are not used as negatives."""
    logits = c @ t.T / tau
    fn = (t @ t.T > 0.95) & ~torch.eye(len(t), dtype=torch.bool, device=t.device)
    logits = logits.masked_fill(fn, float("-inf"))
    y = torch.arange(len(c), device=c.device)
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2


@torch.no_grad()
def recall(q, k, ks):
    """q[i] should retrieve k[i]. Rank = number of keys scoring strictly higher (identical texts tie and count as hits)."""
    sim = q @ k.T
    rank = (sim > sim.diagonal()[:, None]).sum(1)
    return {f"R@{n}": (rank < n).float().mean().item() for n in ks}


def cmd_train():
    t0, dev = time.time(), "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    d = load("train", dev)
    vids = np.unique(d["vid"])
    val_v = set(np.random.default_rng(0).permutation(vids)[:max(1, round(VAL_FRAC * len(vids)))])
    is_val = np.isin(d["vid"], list(val_v))
    tr, va = sub(d, ~is_val), sub(d, is_val)
    print(f"train {len(tr['ids'])} sentences / {len(vids) - len(val_v)} videos, val {len(va['ids'])} / {len(val_v)}", flush=True)
    model = Head(tr["X"].mean(0), tr["X"].std(0) + 1e-6).to(dev)
    cnt = torch.bincount(tr["Z"], minlength=len(ZONES)).float()
    w = cnt.sum() / (len(ZONES) * cnt.clamp(min=1))  # inverse-frequency zone weights
    n = len(tr["ids"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS * math.ceil(n / BATCH))
    best, bad, hist = -1.0, 0, []
    os.makedirs("heads", exist_ok=True)
    for ep in range(EPOCHS):
        model.train()
        perm, tot = torch.randperm(n, device=dev), 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            c, zl = model(tr["X"][b])
            loss = info_nce(c, tr["Y"][b], model.log_tau.exp().clamp(min=0.01)) + 0.5 * F.cross_entropy(zl, tr["Z"][b], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            c, zl = model(va["X"])
        r = recall(c, va["Y"], (1, 5, 10))
        zacc = (zl.argmax(1) == va["Z"]).float().mean().item()
        hist.append(dict(epoch=ep + 1, loss=tot / n, tau=model.log_tau.exp().item(), zone_acc=zacc, **r))
        print(f"ep {ep + 1:2d} loss {tot / n:.3f} tau {hist[-1]['tau']:.3f} val R@1 {r['R@1']:.3f} R@5 {r['R@5']:.3f} "
              f"R@10 {r['R@10']:.3f} zone acc {zacc:.3f}", flush=True)
        if r["R@5"] > best:
            best, bad = r["R@5"], 0
            torch.save(dict(state=model.state_dict(), epoch=ep + 1, val=hist[-1], zones=ZONES), "heads/align_head.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    print(f"best val R@5 {best:.3f} -> heads/align_head.pt", flush=True)
    json.dump(dict(history=hist, n_train=n, n_val=len(va["ids"]), val_videos=sorted(val_v), zone_weights=w.tolist()),
              open("results/phase2_train.json", "w"), indent=1)
    log_time("train", time.time() - t0, epochs=len(hist), best_val_R5=best)


# ---------------------------------------------------------------- STEP 4: evaluation
def zone_report(z, p):
    cm = np.bincount(3 * z + p, minlength=9).reshape(3, 3)  # rows = true, cols = predicted
    tp = np.diag(cm)
    P, R = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)
    F1 = 2 * P * R / np.maximum(P + R, 1e-9)
    return dict(confusion=cm.tolist(), precision=P.tolist(), recall=R.tolist(), f1=F1.tolist(), support=cm.sum(1).tolist(),
                accuracy=float(tp.sum() / cm.sum()), macro_f1=float(F1.mean()), red_as_green=float(cm[2, 0] / max(cm[2].sum(), 1)))


def cmd_eval():
    t0, dev = time.time(), "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load("heads/align_head.pt", map_location=dev)
    te, tr = load("test", dev), load("train", dev)
    model = Head(torch.zeros(1024), torch.ones(1024)).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    with torch.no_grad():
        c, zl = model(te["X"])
    N = len(te["ids"])
    res = dict(n_test=N, checkpoint_epoch=ck["epoch"], val=ck["val"],
               clip_to_sentence=recall(c, te["Y"], (1, 5, 10)), sentence_to_clip=recall(te["Y"], c, (1, 5, 10)),
               chance={f"R@{k}": min(1, k / N) for k in (1, 5, 10)})
    z = te["Z"].cpu().numpy()
    res["zone_head"] = zone_report(z, zl.argmax(1).cpu().numpy())

    # baseline: linear zone classifier on the raw V-JEPA embeddings, no text loss
    from sklearn.linear_model import LogisticRegression
    Xtr, Xte = tr["X"].cpu().numpy(), te["X"].cpu().numpy()
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit((Xtr - mu) / sd, tr["Z"].cpu().numpy())
    res["zone_linear_baseline"] = zone_report(z, lr.predict((Xte - mu) / sd))

    # qualitative: 20 RED + 20 YELLOW test clips -> top-3 GT test sentences by aligned clip embedding
    rng, sim, ex = np.random.default_rng(0), (c @ te["Y"].T).cpu().numpy(), []
    for zi in (2, 1):
        pick = np.flatnonzero(z == zi)
        for i in sorted(rng.choice(pick, min(20, len(pick)), replace=False)):
            top = np.argsort(-sim[i])[:3]
            ex.append(dict(video=te["vid"][i], zone=ZONES[zi], pred_zone=ZONES[int(zl[i].argmax())], text=te["text"][i],
                           rank_of_gt=int((sim[i] > sim[i, i]).sum()) + 1,
                           top3=[dict(text=te["text"][j], zone=ZONES[z[j]], cos=float(sim[i, j]), same_video=bool(te["vid"][j] == te["vid"][i]),
                                      is_gt=bool(j == i)) for j in top]))
    res["examples"] = ex
    json.dump(res, open("results/phase2_eval.json", "w"), indent=1)
    for k in ("clip_to_sentence", "sentence_to_clip", "chance"):
        print(k, {a: round(b, 4) for a, b in res[k].items()})
    for k in ("zone_head", "zone_linear_baseline"):
        r = res[k]
        print(f"{k}: acc {r['accuracy']:.3f} macro-F1 {r['macro_f1']:.3f} RED->GREEN {r['red_as_green']:.3f} "
              f"P {np.round(r['precision'], 3)} R {np.round(r['recall'], 3)} cm {r['confusion']}")
    for e in ex[:3]:
        print(json.dumps(e, indent=1))
    log_time("eval", time.time() - t0, n_test=N)


# ---------------------------------------------------------------- inference: match clips to the standard dictionary
def load_dictionary(dev, d=None):
    """-> concepts (dictionary.jsonl rows), D (S, 1536) phrase/synonym embeddings, owner (S,) concept index per row.
    d: a folder holding dictionary.jsonl, dictionary.npy, dictionary_ids.json (default: the training layout)."""
    paths = (DICT, "text_emb/dictionary_ids.json", "text_emb/dictionary.npy") if d is None else \
        (f"{d}/dictionary.jsonl", f"{d}/dictionary_ids.json", f"{d}/dictionary.npy")
    concepts = list(map(json.loads, open(paths[0], encoding="utf-8")))
    pos = {c["id"]: i for i, c in enumerate(concepts)}
    owner = torch.tensor([pos[cid] for cid, _ in json.load(open(paths[1], encoding="utf-8"))], device=dev)
    return concepts, torch.tensor(np.load(paths[2]), device=dev), owner


def match_dictionary(c, D, owner, n_concepts):
    """Aligned clip embeddings c (N, 1536) -> (N, n_concepts) score = max cosine over each concept's phrase and synonyms."""
    sim = c @ D.T
    return torch.full((len(c), n_concepts), -1.0, device=c.device).scatter_reduce(
        1, owner.expand(len(c), -1), sim, "amax", include_self=False)


def cmd_dict():
    """Test clips -> standard dictionary. Correct concept = one whose source_ids contain the clip's sentence."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    concepts, D, owner = load_dictionary(dev)
    ck = torch.load("heads/align_head.pt", map_location=dev)
    model = Head(torch.zeros(1024), torch.ones(1024)).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    z = np.load("vjepa_emb/sentence_emb.npz")  # merged clip embeddings (zone_heads.py)
    rows, idx = {r["id"]: r for r in read_rows()}, h5_index()
    keep = np.array([idx[rows[i]["video_id"]]["split"] == "test" for i in z["ids"]])
    ids = z["ids"][keep]
    with torch.no_grad():
        c, zl = model(torch.tensor(z["X"][keep], device=dev))
        S = match_dictionary(c, D, owner, len(concepts)).cpu().numpy()
    zone = np.array([ZONES.index(rows[i]["zone"]) for i in ids])
    owners = {}  # sentence id -> concept indices that cover it
    for k, cpt in enumerate(concepts):
        for s in cpt.get("source_ids", []):
            owners.setdefault(s, set()).add(k)
    order = np.argsort(-S, 1)
    covered = [j for j, i in enumerate(ids) if i in owners]
    rank = np.array([np.flatnonzero(np.isin(order[j], list(owners[ids[j]])))[0] for j in covered])  # best covering concept
    czone = np.array([ZONES.index(cpt["zone"]) for cpt in concepts])
    yr = np.flatnonzero(zone >= 1)
    res = dict(n_concepts=len(concepts), n_test=len(ids), n_covered=len(covered),
               concept_recall={f"R@{k}": float((rank < k).mean()) for k in (1, 3, 5, 10)},
               concept_chance={f"R@{k}": min(1, k / len(concepts)) for k in (1, 3, 5, 10)},
               top1_zone_agrees_on_yellow_red=float((czone[order[yr, 0]] == zone[yr]).mean()),
               top1_cos_by_zone={ZONES[k]: float(S[zone == k].max(1).mean()) for k in range(3)})
    rng, ex = np.random.default_rng(0), []
    for zi in (2, 1):
        for j in sorted(rng.choice(np.flatnonzero(zone == zi), 20, replace=False)):
            ex.append(dict(video=rows[ids[j]]["video_id"], zone=ZONES[zi], pred_zone=ZONES[int(zl[j].argmax())],
                           text=rows[ids[j]]["text"], top3=[dict(phrase=concepts[k]["phrase"], path=concepts[k].get("path"),
                           zone=concepts[k]["zone"], severity=concepts[k]["severity"], cos=float(S[j, k]),
                           covers_gt=k in owners.get(ids[j], ())) for k in order[j, :3]]))
    res["examples"] = ex
    json.dump(res, open("results/phase2_dict.json", "w"), indent=1)
    print({k: v for k, v in res.items() if k != "examples"})
    for e in ex:
        print(f"\n[{e['zone']} -> pred {e['pred_zone']}] {e['video']}: {e['text']}")
        for t in e["top3"]:
            print(f"   {t['cos']:.2f} {t['zone']:6} sev {t['severity']} {'*' if t['covers_gt'] else ' '} {t['phrase']}")


if __name__ == "__main__":
    {"text": cmd_text, "clips": cmd_clips, "train": cmd_train, "eval": cmd_eval, "dict": cmd_dict}[sys.argv[1]](*sys.argv[2:])
