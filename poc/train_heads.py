"""STEPS 1-3: anomaly head (A, MIL) and crime-type head (B) on cached V-JEPA 2 segment embeddings.

python train_heads.py sweepA <task> <ntasks>   -> results/headA_task<task>.csv
python train_heads.py finalA                   -> results/headA.csv, results/final_A.json, ckpt/headA.*
python train_heads.py sweepB <task> <ntasks>   -> results/headB_task<task>.csv
python train_heads.py finalB                   -> results/headB.csv, results/final_B.json, ckpt/headB_<labels>.*
Selection is by mean val score over seeds; only selected configs ever touch the test split.
"""
import csv, glob, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
EMB, RES, CKPT = (os.path.join(ROOT, d) for d in ("emb", "results", "ckpt"))
ANNOT = "/projects/adsum-mllm-om/data/ucf_crime_dropbox/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
SEG, D, V, STRIDE = 32, 1024, 4, 4
SEEDS, BATCH, PATIENCE, MAX_EP = (0, 1, 2), 32, 15, 150
GROUPS = {"Theft": ("Robbery", "Stealing", "Burglary", "Shoplifting"),
          "Violence": ("Fighting", "Assault", "Abuse", "Shooting"),
          "FireExplosion": ("Arson", "Explosion"), "RoadAccidents": ("RoadAccidents",),
          "Vandalism": ("Vandalism",), "Arrest": ("Arrest",)}
GRID_A = [dict(loss=l, k=k, lr=lr, aug=a) for l in ("hinge", "topk_bce") for k in (1, 3, 5)
          for lr in (1e-4, 5e-4, 1e-3) for a in (0, 1)]
GRID_B = [dict(labels=s, imb=i, k=k, lr=lr, aug=a) for s in ("13", "grouped") for i in ("wce", "sampler")
          for k in (1, 3, 5) for lr in (1e-4, 5e-4, 1e-3) for a in (0, 1)]
dev = "cuda" if torch.cuda.is_available() else "cpu"


class Data:
    def __init__(self):
        z = np.load(os.path.join(EMB, "all.npz"))
        assert z["done"].all(), "embedding cache incomplete, run extract_emb.py"
        self.vid, self.cls, self.split = z["vid"], z["cls"], z["split"]
        self.n_frames, self.n_src = z["n_frames"], z["n_src_frames"]
        self.y = (self.cls != "NormalVideos").astype(int)
        tr = np.where(self.split == "train")[0]
        self.tr, self.va = train_test_split(tr, test_size=0.15, stratify=self.cls[tr], random_state=0)
        self.te = np.where(self.split == "test")[0]
        x0 = torch.from_numpy(z["X"][self.tr, 0]).float().reshape(-1, D)
        self.mu, self.sd = x0.mean(0), x0.std(0) + 1e-6  # train-set stats (v0)
        self.X = ((torch.from_numpy(z["X"]).float() - self.mu) / self.sd).half().to(dev)  # (N, V, SEG, D)


def mlp(hid, out, p):
    return nn.Sequential(nn.Linear(D, hid), nn.ReLU(), nn.Dropout(p), nn.Linear(hid, out))


def topk_mean(s, k):
    return s.topk(min(k, s.shape[1]), dim=1).values.mean(1)


def views(n, aug, g):
    return torch.randint(0, V, (n,), generator=g, device=dev) if aug else torch.zeros(n, dtype=torch.long, device=dev)


# ---------------- Head A: normal vs anomaly (MIL over 32 segments) ----------------

def bags(d, idx, aug, g):
    idx = torch.as_tensor(idx, device=dev)
    x = d.X[idx, views(len(idx), aug, g)].float()  # (B, SEG, D)
    if aug:  # segment dropout 15% (order kept) + feature noise
        keep = torch.rand(len(idx), SEG, generator=g, device=dev).argsort(1)[:, :round(SEG * 0.85)].sort(1).values
        x = x.gather(1, keep[..., None].expand(-1, -1, D))
        x = x + 0.05 * torch.randn(x.shape, generator=g, device=dev)
    return x


@torch.no_grad()
def seg_scores(m, d, idx, view=0):
    m.eval()
    return m(d.X[torch.as_tensor(idx, device=dev), view].float()).squeeze(-1)  # (n, SEG)


def video_auc(m, d, idx):
    return roc_auc_score(d.y[idx], seg_scores(m, d, idx).max(1).values.cpu().numpy())


def frame_auc(d, idx, s):
    """Each source frame (30 fps) gets the score of the segment containing it."""
    ann = {}
    for line in open(ANNOT):
        p = line.split()
        if p:
            ann[p[0][:-4]] = [int(v) for v in p[2:6]]
    labels, scores = [], []
    for j, i in enumerate(idx):
        f = np.arange(d.n_src[i])
        seg = np.minimum(np.minimum(f // STRIDE, d.n_frames[i] - 1) * SEG // d.n_frames[i], SEG - 1)
        lab = np.zeros(len(f), bool)
        a = ann.get(d.vid[i], [-1] * 4)
        for s0, s1 in ((a[0], a[1]), (a[2], a[3])):
            if s0 >= 0:
                lab[s0:s1 + 1] = True
        labels.append(lab)
        scores.append(s[j][seg])
    return roc_auc_score(np.concatenate(labels), np.concatenate(scores))


def train_A(d, cfg, seed):
    torch.manual_seed(seed)
    g, rng = torch.Generator(device=dev).manual_seed(seed), np.random.default_rng(seed)
    m = nn.Sequential(mlp(512, 1, 0.6), nn.Sigmoid()).to(dev)
    opt = torch.optim.Adam(m.parameters(), cfg["lr"], weight_decay=1e-4)
    an, no = d.tr[d.y[d.tr] == 1], d.tr[d.y[d.tr] == 0]
    best, state, bad, ep = -1, None, 0, 0
    for ep in range(MAX_EP):
        m.train()
        pa = rng.permutation(an)
        for it in range(len(an) // BATCH):  # equal numbers of anomalous and normal videos per batch
            sa = m(bags(d, pa[it * BATCH:(it + 1) * BATCH], cfg["aug"], g)).squeeze(-1)
            sn = m(bags(d, rng.choice(no, BATCH, replace=False), cfg["aug"], g)).squeeze(-1)
            if cfg["loss"] == "hinge":  # Sultani et al. 2018
                loss = (F.relu(1 - topk_mean(sa, cfg["k"]) + topk_mean(sn, cfg["k"])).mean()
                        + 8e-5 * ((sa[:, 1:] - sa[:, :-1]) ** 2).sum(1).mean() + 8e-5 * sa.sum(1).mean())
            else:
                t = torch.cat([topk_mean(sa, cfg["k"]), topk_mean(sn, cfg["k"])]).clamp(1e-6, 1 - 1e-6)
                loss = F.binary_cross_entropy(t, torch.cat([torch.ones(BATCH), torch.zeros(BATCH)]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        auc = video_auc(m, d, d.va)
        if auc > best:
            best, bad, state = auc, 0, {k: v.clone() for k, v in m.state_dict().items()}
        elif (bad := bad + 1) >= PATIENCE:
            break
    m.load_state_dict(state)
    return m, best, ep + 1


# ---------------- Head B: crime type, anomalous videos only ----------------

def label_map(labels):
    names = sorted(c for g in GROUPS.values() for c in g) if labels == "13" else list(GROUPS)
    of = {c: names.index(c if labels == "13" else next(k for k, v in GROUPS.items() if c in v))
          for g in GROUPS.values() for c in g}
    return names, of


def b_inputs(d, S, idx, vw, k):
    """Mean of the top-k segments ranked by Head A scores S (N, V, SEG); k=SEG means all segments."""
    idx = torch.as_tensor(idx, device=dev)
    x = d.X[idx, vw].float()
    if k >= SEG:
        return x.mean(1)
    top = S[idx, vw].topk(k, dim=1).indices
    return x.gather(1, top[..., None].expand(-1, -1, D)).mean(1)


def eval_B(m, d, S, idx, k, lab):
    m.eval()
    with torch.no_grad():
        p = m(b_inputs(d, S, idx, torch.zeros(len(idx), dtype=torch.long, device=dev), k)).argmax(1).cpu().numpy()
    return p, f1_score(lab[idx], p, average="macro"), balanced_accuracy_score(lab[idx], p)


def train_B(d, S, cfg, seed):
    torch.manual_seed(seed)
    g, rng = torch.Generator(device=dev).manual_seed(seed), np.random.default_rng(seed)
    names, of = label_map(cfg["labels"])
    lab = np.array([of.get(c, -1) for c in d.cls])
    tr, va = d.tr[d.y[d.tr] == 1], d.va[d.y[d.va] == 1]
    cnt = np.bincount(lab[tr], minlength=len(names)).astype(float)
    w = None
    if cfg["imb"] == "wce":
        w = 1 / np.sqrt(cnt)
        w = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    pool = {c: tr[lab[tr] == c] for c in range(len(names))}
    ylab = torch.as_tensor(lab, device=dev)
    m = mlp(256, len(names), 0.5).to(dev)
    opt = torch.optim.Adam(m.parameters(), cfg["lr"], weight_decay=1e-4)
    best, state, bad, ep = -1, None, 0, 0
    for ep in range(MAX_EP):
        m.train()
        order = (rng.choice(tr, len(tr), p=(1 / cnt[lab[tr]]) / (1 / cnt[lab[tr]]).sum())
                 if cfg["imb"] == "sampler" else rng.permutation(tr))
        for it in range(0, len(order), BATCH):
            b = order[it:it + BATCH]
            x = b_inputs(d, S, b, views(len(b), cfg["aug"], g), cfg["k"])
            if cfg["aug"]:  # within-class mixup + feature noise
                mate = np.array([rng.choice(pool[lab[i]]) for i in b])
                lam = torch.as_tensor(rng.beta(0.2, 0.2, len(b)), dtype=torch.float32, device=dev)[:, None]
                x = lam * x + (1 - lam) * b_inputs(d, S, mate, views(len(b), 1, g), cfg["k"])
                x = x + 0.05 * torch.randn(x.shape, generator=g, device=dev)
            loss = F.cross_entropy(m(x), ylab[torch.as_tensor(b, device=dev)], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = eval_B(m, d, S, va, cfg["k"], lab)[1]
        if f1 > best:
            best, bad, state = f1, 0, {k: v.clone() for k, v in m.state_dict().items()}
        elif (bad := bad + 1) >= PATIENCE:
            break
    m.load_state_dict(state)
    return m, best, ep + 1, names, lab


# ---------------- sweep / selection plumbing ----------------

def sweep(head, task, ntasks):
    d = Data()
    S = load_scores(d) if head == "B" else None
    grid = GRID_A if head == "A" else GRID_B
    jobs = [(c, s) for c in grid for s in SEEDS][int(task)::int(ntasks)]
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, f"head{head}_task{task}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0]) + ["seed", "val", "epochs", "sec"])
        w.writeheader()
        for cfg, seed in jobs:
            t = time.time()
            out = train_A(d, cfg, seed) if head == "A" else train_B(d, S, cfg, seed)
            w.writerow({**cfg, "seed": seed, "val": round(out[1], 4), "epochs": out[2], "sec": round(time.time() - t, 1)})
            f.flush()
            print(head, cfg, seed, round(out[1], 4), flush=True)


def select(head, keys):
    rows = [r for p in sorted(glob.glob(os.path.join(RES, f"head{head}_task*.csv"))) for r in csv.DictReader(open(p))]
    with open(os.path.join(RES, f"head{head}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(float(r["val"]))
    ranked = sorted(([dict(zip(keys, k)), float(np.mean(v)), float(np.std(v)), len(v)] for k, v in groups.items()),
                    key=lambda r: -r[1])
    return [{"cfg": c, "val_mean": m, "val_std": s, "n": n} for c, m, s, n in ranked]


def typed(cfg):
    return {k: (cfg[k] if k in ("loss", "labels", "imb") else float(cfg[k]) if k == "lr" else int(cfg[k])) for k in cfg}


def load_scores(d):
    m = nn.Sequential(mlp(512, 1, 0.6), nn.Sigmoid()).to(dev)
    m.load_state_dict(torch.load(os.path.join(CKPT, "headA.pt"), map_location=dev))
    m.eval()
    with torch.no_grad():
        return torch.stack([m(d.X[:, v].float()).squeeze(-1) for v in range(V)], 1)  # (N, V, SEG)


def finalA():
    d = Data()
    ranked = select("A", list(GRID_A[0]))
    cfg = typed(ranked[0]["cfg"])
    runs = []
    for seed in SEEDS:
        m, val, ep = train_A(d, cfg, seed)
        s = seg_scores(m, d, d.te).cpu().numpy()
        runs.append(dict(seed=seed, val_auc=val, test_auc=roc_auc_score(d.y[d.te], s.max(1)),
                         test_frame_auc=frame_auc(d, d.te, s), epochs=ep, model=m))
    best = max(runs, key=lambda r: r["val_auc"])
    os.makedirs(CKPT, exist_ok=True)
    torch.save(best["model"].state_dict(), os.path.join(CKPT, "headA.pt"))
    np.savez(os.path.join(CKPT, "feat_norm.npz"), mean=d.mu.numpy(), std=d.sd.numpy())
    json.dump(dict(cfg=cfg, seed=best["seed"], arch="1024-512-1 relu dropout0.6 sigmoid",
                   video_score="max over 32 segments", segments=SEG, frames_per_segment=16, fps=7.5,
                   size=224, model="facebook/vjepa2-vitl-fpc64-256"), open(os.path.join(CKPT, "headA.json"), "w"), indent=1)
    out = dict(ranked=ranked, cfg=cfg, runs=[{k: v for k, v in r.items() if k != "model"} for r in runs])
    json.dump(out, open(os.path.join(RES, "final_A.json"), "w"), indent=1, default=float)
    print("FINAL_A", cfg, [(r["test_auc"], r["test_frame_auc"]) for r in runs])


def finalB():
    d = Data()
    S = load_scores(d)
    ranked_all = select("B", list(GRID_B[0]))
    out = {}
    for labels in ("13", "grouped"):
        ranked = [r for r in ranked_all if r["cfg"]["labels"] == labels]
        cfg = typed(ranked[0]["cfg"])
        res = {}
        for name, c in (("selected", cfg), ("all_segments_baseline", {**cfg, "k": SEG})):
            runs = []
            for seed in SEEDS:
                m, val, ep, names, lab = train_B(d, S, c, seed)
                te = d.te[d.y[d.te] == 1]
                p, f1, bacc = eval_B(m, d, S, te, c["k"], lab)
                runs.append(dict(seed=seed, val_f1=val, test_f1=f1, test_bacc=bacc, epochs=ep,
                                 per_class_f1=f1_score(lab[te], p, average=None, labels=range(len(names))).tolist(),
                                 cm=confusion_matrix(lab[te], p, labels=range(len(names))).tolist(), model=m))
            best = max(runs, key=lambda r: r["val_f1"])
            if name == "selected":
                torch.save(best["model"].state_dict(), os.path.join(CKPT, f"headB_{labels}.pt"))
                json.dump(dict(cfg=c, seed=best["seed"], classes=names, arch="1024-256-C relu dropout0.5",
                               input="mean of top-k segments ranked by headA (k=cfg.k)"),
                          open(os.path.join(CKPT, f"headB_{labels}.json"), "w"), indent=1)
            res[name] = dict(cfg=c, best_seed=best["seed"], classes=names,
                             test_counts=np.bincount(lab[te], minlength=len(names)).tolist(),
                             runs=[{k: v for k, v in r.items() if k != "model"} for r in runs])
        out[labels] = dict(ranked=ranked, **res)
    json.dump(out, open(os.path.join(RES, "final_B.json"), "w"), indent=1, default=float)
    print("FINAL_B done")


if __name__ == "__main__":
    mode = sys.argv[1]
    {"sweepA": lambda *a: sweep("A", *a), "sweepB": lambda *a: sweep("B", *a),
     "finalA": finalA, "finalB": finalB}[mode](*sys.argv[2:])
