"""Print POC result tables from results/final_A.json and results/final_B.json (text only)."""
import json, os

import numpy as np

RES = os.environ.get("POC_RES") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
KMEANS_AUC = 0.62  # earlier k-means baseline, V-JEPA 2 at k=10
ms = lambda v: f"{np.mean(v):.3f} +/- {np.std(v):.3f}"
cfg_str = lambda c: " ".join(f"{k}={v}" for k, v in c.items())


def head_a():
    a = json.load(open(os.path.join(RES, "final_A.json")))
    print("HEAD A (normal vs anomaly): configs ranked by val AUC, mean +/- std over seeds")
    for i, r in enumerate(a["ranked"]):
        print(f"  {i + 1:3d}. {cfg_str(r['cfg']):45s} val AUC {r['val_mean']:.3f} +/- {r['val_std']:.3f}")
    runs = a["runs"]
    print(f"\n  winner: {cfg_str(a['cfg'])}")
    print(f"  test video AUC  {ms([r['test_auc'] for r in runs])}   (k-means baseline {KMEANS_AUC})")
    print(f"  test frame AUC  {ms([r['test_frame_auc'] for r in runs])}\n")


def head_b():
    b = json.load(open(os.path.join(RES, "final_B.json")))
    for labels, x in b.items():
        print(f"HEAD B (crime type, {labels} labels): top configs by val macro-F1")
        for i, r in enumerate(x["ranked"][:10]):
            print(f"  {i + 1:3d}. {cfg_str(r['cfg']):55s} val F1 {r['val_mean']:.3f} +/- {r['val_std']:.3f}")
        for name in ("selected", "all_segments_baseline"):
            s = x[name]
            runs = s["runs"]
            print(f"\n  {name} ({cfg_str(s['cfg'])}): test macro-F1 {ms([r['test_f1'] for r in runs])}, "
                  f"balanced acc {ms([r['test_bacc'] for r in runs])}")
        s = x["selected"]
        best = next(r for r in s["runs"] if r["seed"] == s["best_seed"])
        names, n = s["classes"], s["test_counts"]
        print(f"\n  per class (seed {s['best_seed']}):   {'class':15s} {'n_test':>6s} {'F1':>6s}")
        for c, k, f in zip(names, n, best["per_class_f1"]):
            print(f"  {'':22s}{c:15s} {k:6d} {f:6.2f}")
        print("\n  confusion matrix (rows = true, cols = predicted):")
        print("  " + " " * 15 + "".join(f"{c[:5]:>6s}" for c in names))
        for c, row in zip(names, best["cm"]):
            print(f"  {c:15s}" + "".join(f"{v:6d}" for v in row))
        print()


if __name__ == "__main__":
    head_a()
    head_b()
