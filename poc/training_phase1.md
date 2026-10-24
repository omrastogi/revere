# Training Phase 1: frozen V-JEPA 2 + MIL heads on UCF-Crime

**Goal:** from surveillance video, (A) tell *when* a crime happens and (B) say *which* crime,
without frame-level training labels and without fine-tuning a video model.

**Headline (official test split, 290 videos):**

| | Metric | Phase 1 | Baseline |
|---|---|---|---|
| Head A, crime vs normal | video ROC AUC | **0.948 ± 0.003** | 0.62 (k-means on V-JEPA 2) |
| Head A | frame ROC AUC | **0.792 ± 0.005** | 0.754 (Sultani et al. 2018, reported) |
| Head B, crime type (6 groups) | macro-F1 | 0.35 ± 0.03 | 0.39 (mean of all segments) |
| Head B, crime type (13 classes) | macro-F1 | 0.23 ± 0.02 | 0.19 (mean of all segments) |

---

## 1. Core ideas

1. **Frozen foundation model, small heads.** V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`) is a
   self-supervised *video* encoder, so it already models motion. We never fine-tune it. We embed every
   video once, cache the vectors, and train only tiny MLPs on top. Training a head takes seconds,
   which makes a full hyperparameter sweep cheap.
2. **Multiple-instance learning (MIL).** UCF-Crime train labels are per *video* ("this video contains
   a robbery somewhere"), not per frame. Each video is a *bag* of 32 segments. The head scores every
   segment, and only the top-k segment scores are pushed up for crime videos and down for normal ones.
   Segment-level localisation comes out for free.
3. **Two heads, two jobs.** Head A answers "is something wrong, and when?". Head B answers "what is it?",
   using only the segments Head A found most suspicious (top-k), so it is not diluted by minutes of
   empty footage.
4. **Augment in embedding space, cheaply.** We cache 4 views per segment (original, horizontal flip,
   temporal jitter, brightness/contrast/blur) instead of augmenting pixels during training. On top of
   that come feature-space tricks: segment dropout, Gaussian noise, and within-class mixup.
5. **Honest selection.** Configs are chosen by mean validation score over 3 seeds. The test split is
   touched once, only by the selected configs.

## 2. Data

| | |
|---|---|
| Source | Official UCF-Crime (Dropbox): 1,900 raw mp4s, 320x240, 30 fps, 128 h |
| Split | Official: 1,610 train (800 normal, 810 crime) / 290 test (150 normal, 140 crime) |
| Validation | 15% of train, stratified by class, seed 0 |
| Classes | NormalVideos + 13 crimes. Grouped into 6 for Head B: Theft (Robbery, Stealing, Burglary, Shoplifting), Violence (Fighting, Assault, Abuse, Shooting), Fire/Explosion (Arson, Explosion), RoadAccidents, Vandalism, Arrest |
| Test labels | Temporal annotations (crime start/end frames) for test videos, used only for frame AUC |

Normal footage is 73% of all hours, and several crime classes have only 2-5 test videos, so
per-class test numbers for those classes are noisy.

## 3. What we did

| Step | What | Where / how long |
|---|---|---|
| 0a | Decode mp4 → HDF5: 224x224 RGB uint8, 7.5 fps (every 4th frame), 64 shards + master file, 518 GB | `build_ucf_h5.py`, 96 CPU cores, ~13 min |
| 0b | Embeddings: 32 equal segments per video × 16 frames (~2 s) from each centre → V-JEPA 2 (fp16, 224 input, no resize) → mean-pool tokens → (4 views, 32, 1024) per video | `extract_emb.py`, 4 V100s, ~35 min |
| 1 | Head A sweep: 108 runs | `train_heads.py sweepA`, 3 V100s, ~6 min |
| 2 | Head B sweep: 216 runs, using the selected Head A to rank segments | `train_heads.py sweepB`, ~5 min |
| 3 | Retrain selected configs × 3 seeds, test once, save checkpoints | `train_heads.py finalA/finalB` |
| 4 | Local demo on a laptop RTX 3050: annotated video with "CRIME DETECTED" + type | `demo.py`, ~6 s per video |

**Head A** (normal vs crime)
- MLP 1024 → 512 → 1, dropout 0.6, sigmoid per segment. Features standardised with train mean/std.
- Each batch holds 32 crime and 32 normal videos. Video score = max over segments.
- Losses tried:
  - **hinge ranking** (Sultani et al.): top-k crime > top-k normal by a margin, plus temporal-smoothness and sparsity terms.
  - **top-k BCE**: BCE on the mean of the top-k segment scores.

**Head B** (crime type)
- MLP 1024 → 256 → C, dropout 0.5, trained only on crime videos.
- Input = mean of the top-k segments ranked by Head A.
- Imbalance handled by either cross-entropy weighted by 1/√count or a class-balanced sampler.

**Augmentation** (switch `aug` on/off in the sweep)
- One random cached view per video each epoch.
- Gaussian noise σ = 0.05 on standardised features.
- Head A: 15% segment dropout. Head B: within-class mixup, α = 0.2.

**Sweep grids** (3 seeds each, Adam, weight decay 1e-4, early stopping with patience 15)
- A: loss {hinge, top-k BCE} × k {1, 3, 5} × lr {1e-4, 5e-4, 1e-3} × aug {off, on}
- B: labels {13, grouped} × imbalance {weighted CE, sampler} × k {1, 3, 5} × lr {1e-4, 5e-4, 1e-3} × aug {off, on}

## 4. What we learned

- **Video understanding is the main win.** k-means on the same embeddings reached 0.62. A tiny
  supervised MIL head reaches 0.95. Hyperparameters barely matter: the top 8 Head A configs are within
  0.002 val AUC of each other.
- **Augmentation helps Head A a little.** Mean val AUC is 0.951 with aug and 0.948 without, and 7 of
  the top 8 configs use it. For Head B it is mixed: the grouped winner uses it, the 13-class winner
  does not.
- **Top-k BCE is about the same as hinge ranking.** Mean val AUC is 0.950 vs 0.948. The winner is
  top-k BCE, k=5, lr 1e-3, aug on.
- **Localisation is coarse.** Frame AUC (0.79) trails video AUC (0.95). The model knows *that* a
  video has a crime better than *exactly when*, because 32 segments of 2 s each are blunt.
- **Head B is data-starved.** There are about 690 training crime videos, some classes with ~40.
  - Selecting segments with Head A did **not** beat averaging all 32 segments for grouped labels
    (0.35 vs 0.39). It only slightly helped for the 13 classes (0.23 vs 0.19).
  - Violence is mostly predicted as Theft. Arrest is never right.
- **The local demo matches the cluster exactly** (embedding cosine 1.000). It also flags a
  non-CCTV phone clip of a scuffle (`video/rudra.mp4`, 5.6-8.6 s).

## 5. Class-wise scores (official test split)

### Head A: crime detected or not (checkpoint seed 2, threshold 0.5)

How to read the columns:
- Each crime class is scored against the 150 normal test videos.
- **prec** = TP / (TP + FP), where the FP are the 10 normal videos wrongly flagged. Small classes get
  low precision just from those 10 false alarms.
- **recall** = share of that class's videos that were flagged.
- **acc** = accuracy on (that class + 150 normals).
- **AUC** = threshold-free ranking of that class vs normals.
- **frame AUC** = per-frame localisation, using the temporal annotations.

| Class | n test | Precision | Recall | F1 | Accuracy | AUC vs normal | Frame AUC |
|---|---|---|---|---|---|---|---|
| Abuse | 2 | 0.167 | 1.000 | 0.286 | 0.934 | 0.990 | 0.986 |
| Arrest | 5 | 0.333 | 1.000 | 0.500 | 0.935 | 0.992 | 0.925 |
| Arson | 9 | 0.474 | 1.000 | 0.643 | 0.937 | 0.984 | 0.867 |
| Assault | 3 | 0.167 | 0.667 | 0.267 | 0.928 | 0.944 | 0.941 |
| Burglary | 13 | 0.524 | 0.846 | 0.647 | 0.926 | 0.976 | 0.886 |
| Explosion | 21 | 0.655 | 0.905 | 0.760 | 0.930 | 0.918 | 0.833 |
| Fighting | 5 | 0.333 | 1.000 | 0.500 | 0.935 | 0.996 | 0.954 |
| RoadAccidents | 23 | 0.643 | 0.783 | 0.706 | 0.913 | 0.923 | 0.876 |
| Robbery | 5 | 0.286 | 0.800 | 0.421 | 0.929 | 0.908 | 0.833 |
| Shooting | 23 | 0.688 | 0.957 | 0.800 | 0.936 | 0.983 | 0.919 |
| Shoplifting | 21 | 0.643 | 0.857 | 0.735 | 0.924 | 0.922 | 0.908 |
| Stealing | 5 | 0.333 | 1.000 | 0.500 | 0.935 | 0.983 | 0.935 |
| Vandalism | 5 | 0.286 | 0.800 | 0.421 | 0.929 | 0.976 | 0.942 |
| **All crimes vs normal** | 140 + 150 | **0.925** | **0.886** | **0.905** | **0.910** | **0.951** | **0.786** |
| NormalVideos | 150 | specificity 0.933 (140/150 correctly not flagged) | | | | | |

### Head B: crime type, 6 groups (selected: sampler, k=3, lr 1e-3, aug on; best-val seed)

Accuracy here is one-vs-rest over the 140 test crime videos. Overall accuracy is 0.457 and macro-F1
is 0.327 (0.346 ± 0.026 over 3 seeds).

| Class | n test | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| Theft | 44 | 0.455 | 0.795 | 0.579 | 0.636 |
| Violence | 33 | 0.412 | 0.212 | 0.280 | 0.743 |
| FireExplosion | 30 | 0.727 | 0.267 | 0.390 | 0.821 |
| RoadAccidents | 23 | 0.500 | 0.565 | 0.531 | 0.836 |
| Vandalism | 5 | 0.167 | 0.200 | 0.182 | 0.936 |
| Arrest | 5 | 0.000 | 0.000 | 0.000 | 0.943 |
| **Macro avg** | 140 | **0.377** | **0.340** | **0.327** | |

### Head B: crime type, 13 classes (selected: weighted CE, k=5, lr 5e-4, aug off; best-val seed)

Overall accuracy is 0.293 and macro-F1 is 0.202 (0.228 ± 0.019 over 3 seeds).

| Class | n test | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| Abuse | 2 | 0.000 | 0.000 | 0.000 | 0.950 |
| Arrest | 5 | 0.000 | 0.000 | 0.000 | 0.936 |
| Arson | 9 | 0.357 | 0.556 | 0.435 | 0.907 |
| Assault | 3 | 0.000 | 0.000 | 0.000 | 0.943 |
| Burglary | 13 | 0.267 | 0.308 | 0.286 | 0.857 |
| Explosion | 21 | 1.000 | 0.143 | 0.250 | 0.871 |
| Fighting | 5 | 0.200 | 0.200 | 0.200 | 0.943 |
| RoadAccidents | 23 | 0.548 | 0.739 | 0.630 | 0.857 |
| Robbery | 5 | 0.065 | 0.400 | 0.111 | 0.771 |
| Shooting | 23 | 1.000 | 0.043 | 0.083 | 0.843 |
| Shoplifting | 21 | 1.000 | 0.333 | 0.500 | 0.900 |
| Stealing | 5 | 0.091 | 0.200 | 0.125 | 0.900 |
| Vandalism | 5 | 0.000 | 0.000 | 0.000 | 0.907 |
| **Macro avg** | 140 | **0.348** | **0.225** | **0.202** | |

Classes with 2-5 test videos swing by 0.2-0.5 F1 from a single video. Don't read much into them.

## 6. Next steps

1. **Head B needs more signal, not more tuning.** Options:
   - fine-tune the last V-JEPA 2 blocks on crime videos only;
   - use the attentive-probe head V-JEPA 2 ships with, instead of mean-pooled tokens;
   - add more labelled crime clips.
2. **Finer localisation.** Use more, shorter segments (e.g. 64 × 1 s) or a sliding window at
   inference. Add a temporal model (1-D conv or small transformer) over segment features.
3. **Calibrate the 0.5 threshold** on validation for the demo. The 10 false positives on normal test
   videos are the main precision cost.
4. **Domain gap for the demo:** test more phone and webcam footage. The photometric view (v3) was added
   for this, but its effect has not been measured separately.

## Files

- Cluster: `/projects/adsum-mllm-om/ucf_anomaly/poc/`
  - `emb/all.npz`: cached embeddings
  - `results/headA.csv`, `results/headB.csv`: every sweep run
  - `results/final_*.json`
  - `ckpt/`
- Local: `E:\Om\hackachussets\poc\`
  - scripts: `extract_emb.py`, `train_heads.py`, `sweep.sh`, `report.py`, `demo.py`
  - `ckpt/`
  - `examples/out/*_annotated.mp4`
