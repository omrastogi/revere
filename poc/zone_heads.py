"""Phase 2b: zone Head A (ordinal GREEN < YELLOW < RED) and crime-type Head B on the per-sentence V-JEPA clip embeddings.

cd poc && python zone_heads.py   -> heads/head_a_zone.pt, heads/head_b_{13,grouped}.pt, results/zone_heads.json
Inputs: ../vjepa_emb/part_XX.npz (align.py clips; 16 frames per UCA sentence span, 224 px, 7.5 fps, from the HDF5),
../uca_labels/sentence_labels.jsonl, and the HDF5 index.csv (class and official split, written from the HDF5 attrs).
"""
import copy, csv, itertools, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

U = "/projects/adsum-mllm-om/ucf_anomaly"
H5 = "/projects/adsum-mllm-om/data/ucf_crime_h5"
PARTS, ZONES = 48, ("GREEN", "YELLOW", "RED")
GROUP = {"Burglary": "Theft", "Robbery": "Theft", "Stealing": "Theft", "Shoplifting": "Theft",
         "Abuse": "Violence", "Assault": "Violence", "Fighting": "Violence", "Shooting": "Violence",
         "Arson": "FireExplosion", "Explosion": "FireExplosion", "RoadAccidents": "RoadAccidents",
         "Vandalism": "Vandalism", "Arrest": "Arrest"}
GRID = list(itertools.product((1e-3, 3e-4), (0.1, 0.3), (256, 512)))  # lr, dropout, hidden
BS, MAX_EP, PATIENCE, SEED, VAL = 256, 100, 15, 0, 0.15
TH = np.round(np.arange(0.05, 0.96, 0.05), 2)  # threshold grid for t_yellow, t_red
dev = "cuda" if torch.cuda.is_available() else "cpu"


def merged():
    """All labelled sentences with an embedding; stops if more than 1% have none. Writes vjepa_emb/sentence_emb.npz."""
    rows = [r for r in map(json.loads, open(f"{U}/uca_labels/sentence_labels.jsonl"))
            if r.get("zone") in ZONES and r.get("text", "").strip()]
    emb = {}
    for k in range(PARTS):
        p = f"{U}/vjepa_emb/part_{k:02d}.npz"
        if os.path.exists(p):
            z = np.load(p)
            emb.update(zip(z["ids"].tolist(), z["X"]))
        else:
            print(f"missing shard {p}")
    miss = [r["id"] for r in rows if r["id"] not in emb]
    print(f"{len(rows)} labelled sentences, {len(emb)} embeddings, {len(miss)} missing ids: {miss[:100]}", flush=True)
    if len(miss) > 0.01 * len(rows):
        sys.exit(f"STOP: {len(miss) / len(rows):.1%} of labelled sentences have no embedding")
    rows = [r for r in rows if r["id"] in emb]
    X = np.stack([emb[r["id"]] for r in rows]).astype(np.float32)
    np.savez(f"{U}/vjepa_emb/sentence_emb.npz", X=X, ids=np.array([r["id"] for r in rows]))
    return rows, X


def mlp(out, hid, p):
    return nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hid), nn.GELU(), nn.Dropout(p), nn.Linear(hid, out))


def fit(X, loss_fn, score_fn, Xv, out, lr, p, hid):
    """AdamW wd 0.05, cosine schedule, early stop on score_fn(val logits) (macro-F1); returns the best-epoch model."""
    torch.manual_seed(SEED)
    m = mlp(out, hid, p).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EP * math.ceil(len(X) / BS))
    best, state, bad, best_ep = -1.0, None, 0, 0
    for ep in range(MAX_EP):
        m.train()
        for b in torch.randperm(len(X), device=dev).split(BS):
            loss = loss_fn(m(X[b]), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        m.eval()
        with torch.no_grad():
            s = score_fn(m(Xv))
        if s > best:
            best, state, bad, best_ep = s, copy.deepcopy(m.state_dict()), 0, ep + 1
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(state)
    return m.eval(), best, best_ep


def metrics(y, p, names):
    """Per-class precision/recall/F1/one-vs-rest accuracy; macro averages over classes present in y."""
    C = len(names)
    cm = np.bincount(C * np.asarray(y) + np.asarray(p), minlength=C * C).reshape(C, C)  # rows true, cols predicted
    tp, sup = np.diag(cm), cm.sum(1)
    P, R = tp / np.maximum(cm.sum(0), 1), tp / np.maximum(sup, 1)
    F1 = 2 * P * R / np.maximum(P + R, 1e-9)
    acc1 = (cm.sum() - cm.sum(0) - sup + 2 * tp) / max(cm.sum(), 1)  # one-vs-rest accuracy
    on = sup > 0
    return dict(classes=list(names), confusion=cm.tolist(), support=sup.tolist(), precision=P.tolist(), recall=R.tolist(),
                f1=F1.tolist(), acc_ovr=acc1.tolist(), accuracy=float(tp.sum() / max(cm.sum(), 1)),
                macro_p=float(P[on].mean()), macro_r=float(R[on].mean()), macro_f1=float(F1[on].mean()))


def table(title, m):
    print(f"\n{title}  (n={sum(m['support'])}, accuracy {m['accuracy']:.3f}, macro P/R/F1 "
          f"{m['macro_p']:.3f}/{m['macro_r']:.3f}/{m['macro_f1']:.3f})")
    print(f"| {'class':14} | {'n':>5} | {'P':>5} | {'R':>5} | {'F1':>5} | {'acc':>5} |")
    for i, c in enumerate(m["classes"]):
        print(f"| {c:14} | {m['support'][i]:5d} | {m['precision'][i]:5.3f} | {m['recall'][i]:5.3f} | {m['f1'][i]:5.3f} | "
              f"{m['acc_ovr'][i]:5.3f} |")
    print("confusion (rows true, cols pred):", m["confusion"])


def zone_pred(P, ty, tr):  # P (N, 2) = [p(at least YELLOW), p(RED)]
    return np.where(P[:, 1] > tr, 2, np.where(P[:, 0] > ty, 1, 0))


def head_a(X, z, low, tr, va, te):
    t0 = time.time()
    Y = torch.stack([z >= 1, z == 2], 1).float()
    pw = (len(tr) - Y[tr].sum(0)) / Y[tr].sum(0)  # positive-class weights from training frequencies
    W = torch.where(low, 0.5, 1.0)  # low-confidence labels count half
    Xtr, Ytr, Wtr = X[tr], Y[tr], W[tr]
    zv = z[va].cpu().numpy()

    def loss_fn(lo, b):
        l = F.binary_cross_entropy_with_logits(lo, Ytr[b], pos_weight=pw, reduction="none").mean(1)
        return (l * Wtr[b]).sum() / Wtr[b].sum()

    f1 = lambda lo: metrics(zv, zone_pred(lo.sigmoid().cpu().numpy(), 0.5, 0.5), ZONES)["macro_f1"]
    sweep = []
    for lr, p, hid in GRID:
        m, s, ep = fit(Xtr, loss_fn, f1, X[va], 2, lr, p, hid)
        sweep.append(dict(lr=lr, dropout=p, hidden=hid, val_macro_f1=s, epoch=ep, model=m))
        print(f"A lr {lr:g} dropout {p} hidden {hid}: val macro-F1 {s:.3f} (epoch {ep})", flush=True)
    best = max(sweep, key=lambda d: d["val_macro_f1"])
    m = best.pop("model")
    for d in sweep:
        d.pop("model", None)
    with torch.no_grad():
        Pv, Pt = m(X[va]).sigmoid().cpu().numpy(), m(X[te]).sigmoid().cpu().numpy()

    # thresholds: maximize val macro-F1 subject to RED recall >= 0.8 (else the highest reachable RED recall)
    cands = []
    for ty, tr_ in itertools.product(TH, TH):
        mm = metrics(zv, zone_pred(Pv, ty, tr_), ZONES)
        cands.append((mm["recall"][2] >= 0.8, mm["macro_f1"] if mm["recall"][2] >= 0.8 else mm["recall"][2], mm["macro_f1"], ty, tr_))
    ok, _, vf1, ty, tr_ = max(cands)
    print(f"A best {best}; thresholds t_yellow {ty} t_red {tr_} (val macro-F1 {vf1:.3f}, RED recall>=0.8 reached: {ok})")
    zt = z[te].cpu().numpy()
    res = dict(sweep=sweep, best=best, t_yellow=float(ty), t_red=float(tr_), red_recall_constraint_met=bool(ok),
               val=metrics(zv, zone_pred(Pv, ty, tr_), ZONES), test=metrics(zt, zone_pred(Pt, ty, tr_), ZONES),
               test_default_thresholds=metrics(zt, zone_pred(Pt, 0.5, 0.5), ZONES), seconds=round(time.time() - t0, 1))
    res["test"]["red_as_green"] = res["test"]["confusion"][2][0] / max(res["test"]["support"][2], 1)
    torch.save(dict(state=m.state_dict(), hidden=best["hidden"], dropout=best["dropout"], t_yellow=float(ty),
                    t_red=float(tr_), zones=ZONES, outputs=["at_least_yellow", "red"]), "heads/head_a_zone.pt")
    table("HEAD A val (tuned thresholds)", res["val"])
    table("HEAD A test (tuned thresholds)", res["test"])
    table("HEAD A test (thresholds 0.5/0.5)", res["test_default_thresholds"])
    print(f"RED predicted GREEN on test: {res['test']['red_as_green']:.1%}")
    return res, zone_pred(Pt, ty, tr_)


def head_b(X, cls, z, tr, va, te, zt_pred, names_of):
    """Crime type on YELLOW/RED sentences of crime videos; names_of maps a UCF class to its label."""
    t0 = time.time()
    names = sorted({str(names_of(c)) for c in cls if c != "NormalVideos"})
    lab = np.array([names.index(names_of(c)) if c != "NormalVideos" else -1 for c in cls])
    use = (lab >= 0) & (z.cpu().numpy() >= 1)
    tr, va, te_gt = tr[use[tr]], va[use[va]], te[use[te]]
    L = torch.tensor(lab, device=dev).clamp(min=0)
    cnt = torch.bincount(L[tr], minlength=len(names)).float()
    w = cnt.sum() / (len(names) * cnt.clamp(min=1))  # inverse class frequency
    Xtr, Ltr, lv = X[tr], L[tr], lab[va]
    loss_fn = lambda lo, b: F.cross_entropy(lo, Ltr[b], weight=w)
    f1 = lambda lo: metrics(lv, lo.argmax(1).cpu().numpy(), names)["macro_f1"]
    sweep = []
    for lr, p, hid in GRID:
        m, s, ep = fit(Xtr, loss_fn, f1, X[va], len(names), lr, p, hid)
        sweep.append(dict(lr=lr, dropout=p, hidden=hid, val_macro_f1=s, epoch=ep, model=m))
        print(f"B{len(names)} lr {lr:g} dropout {p} hidden {hid}: val macro-F1 {s:.3f} (epoch {ep})", flush=True)
    best = max(sweep, key=lambda d: d["val_macro_f1"])
    m = best.pop("model")
    for d in sweep:
        d.pop("model", None)
    pred = lambda idx: m(X[idx]).argmax(1).cpu().numpy() if len(idx) else np.zeros(0, int)
    with torch.no_grad():
        res = dict(sweep=sweep, best=best, n_train=len(tr), n_val=len(va), val=metrics(lv, pred(va), names),
                   test=metrics(lab[te_gt], pred(te_gt), names))
        # cascade: test crime-video sentences that Head A flags YELLOW/RED (whatever their GT zone)
        flag = te[(lab[te] >= 0) & (zt_pred >= 1)]
        res["test_cascade"] = metrics(lab[flag], pred(flag), names)
    res["seconds"] = round(time.time() - t0, 1)
    tag = "13" if len(names) == 13 else "grouped"
    torch.save(dict(state=m.state_dict(), hidden=best["hidden"], dropout=best["dropout"], classes=names), f"heads/head_b_{tag}.pt")
    table(f"HEAD B {tag} test (GT YELLOW/RED sentences of crime videos)", res["test"])
    table(f"HEAD B {tag} test cascade (sentences Head A flags YELLOW/RED)", res["test_cascade"])
    return res


def main():
    t0 = time.time()
    os.makedirs("heads", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    rows, X = merged()
    idx = {r["vid"]: r for r in csv.DictReader(open(f"{H5}/index.csv"))}
    vid = np.array([r["video_id"] for r in rows])
    cls = np.array([idx[v]["cls"] for v in vid])
    split = np.array([idx[v]["split"] for v in vid])
    train_v = sorted(set(vid[split == "train"]))
    tv, vv = train_test_split(train_v, test_size=VAL, stratify=[idx[v]["cls"] for v in train_v], random_state=SEED)
    tr, va, te = (np.flatnonzero(np.isin(vid, list(tv))), np.flatnonzero(np.isin(vid, list(vv))),
                  np.flatnonzero(split == "test"))
    print(f"sentences train {len(tr)} ({len(tv)} videos) / val {len(va)} ({len(vv)} videos) / test {len(te)} "
          f"({len(set(vid[te]))} videos); device {dev}", flush=True)
    Xt = torch.tensor(X, device=dev)
    z = torch.tensor([ZONES.index(r["zone"]) for r in rows], device=dev)
    low = torch.tensor([r.get("confidence") == "low" for r in rows], device=dev)
    for name, ix in (("train", tr), ("val", va), ("test", te)):
        print(f"  {name} zones: " + "  ".join(f"{ZONES[k]} {int((z[ix] == k).sum())}" for k in range(3)))

    res = dict(n=dict(train=len(tr), val=len(va), test=len(te), train_videos=len(tv), val_videos=len(vv)))
    res["head_a"], zt_pred = head_a(Xt, z, low, tr, va, te)
    res["head_b_13"] = head_b(Xt, cls, z, tr, va, te, zt_pred, lambda c: c)
    res["head_b_grouped"] = head_b(Xt, cls, z, tr, va, te, zt_pred, lambda c: GROUP[c])
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open("results/zone_heads.json", "w"), indent=1)
    print(f"\ndone in {res['seconds']:.0f}s -> results/zone_heads.json, heads/head_a_zone.pt, heads/head_b_*.pt")


if __name__ == "__main__":
    main()
