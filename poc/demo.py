"""Run the trained POC on one video file: anomaly timeline (Head A) and crime type (Head B, if trained).

python demo.py <video.mp4> [more videos ...]   -> examples/out/<name>_annotated.mp4
Annotated video: "CRIME DETECTED" + crime type on frames whose segment scores above THRESH.
Same preprocessing as training: frames at 7.5 fps, squashed to 224x224, 32 segments x 16 frames,
frozen V-JEPA 2 (fp16), view 0 only. Checkpoints are read from ./ckpt.
"""
import json, os, sys, time

import cv2
import numpy as np
import torch

from extract_emb import D, SEG, T, Embedder, clip_starts

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "out")
FPS, SIZE, THRESH = 7.5, 224, 0.5  # THRESH: segment anomaly score above which a frame is flagged


def mlp(hid, out, p):  # same as train_heads.mlp (not imported: train_heads needs sklearn)
    return torch.nn.Sequential(torch.nn.Linear(D, hid), torch.nn.ReLU(), torch.nn.Dropout(p), torch.nn.Linear(hid, out))


def read_frames(path):
    """Decode at ~7.5 fps, resized to 224x224 RGB. ponytail: keeps all kept frames in RAM (~1 GB per 15 min of video)."""
    cap = cv2.VideoCapture(path)
    stride = max(1, round(cap.get(cv2.CAP_PROP_FPS) / FPS))
    frames, n = [], 0
    while cap.grab():
        if n % stride == 0:
            f = cv2.resize(cap.retrieve()[1], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        n += 1
    return np.stack(frames), stride


def clips(frames):
    center, _ = clip_starts(len(frames), np.random.default_rng(0))  # view 0 = center clips (jitter unused)
    out = []
    for s in center:
        x = frames[s:s + T]
        if len(x) < T:
            x = np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)])
        out.append(x)
    return torch.from_numpy(np.stack(out)).permute(0, 1, 4, 2, 3).float() / 255  # (SEG, T, 3, S, S)


def write_annotated(path, out_path, s, n_kept, stride, crime):
    """Re-read the original video and burn in the verdict of the segment each frame belongs to."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, max(0.4, w / 640)
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        score = float(s[min(min(f // stride, n_kept - 1) * SEG // n_kept, SEG - 1)])
        if score > THRESH:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), max(3, w // 80))
            cv2.putText(img, "CRIME DETECTED", (10, int(30 * scale)), font, scale, (0, 0, 255), 2, cv2.LINE_AA)
            if crime:
                cv2.putText(img, crime, (10, int(60 * scale)), font, scale, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "normal", (10, int(30 * scale)), font, scale * 0.8, (0, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"anomaly {score:.2f}  t={f / fps:.1f}s", (10, h - 10), font, scale * 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out.write(img)
        f += 1
    out.release()


def load_head(name, out, hid, p, sigmoid=False):
    m = mlp(hid, out, p)
    m = torch.nn.Sequential(m, torch.nn.Sigmoid()) if sigmoid else m
    m.load_state_dict(torch.load(os.path.join(CKPT, name), map_location="cpu"))
    return m.eval()


def main(paths):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = np.load(os.path.join(CKPT, "feat_norm.npz"))
    mu, sd = torch.tensor(norm["mean"]), torch.tensor(norm["std"])
    head_a = load_head("headA.pt", 1, 512, 0.6, sigmoid=True)
    heads_b = []
    for labels in ("grouped", "13"):
        cfg_path = os.path.join(CKPT, f"headB_{labels}.json")
        if os.path.exists(cfg_path):
            c = json.load(open(cfg_path))
            heads_b.append((labels, c, load_head(f"headB_{labels}.pt", len(c["classes"]), 256, 0.5)))
    t = time.time()
    emb = Embedder(dev)
    print(f"V-JEPA 2 loaded on {dev} in {time.time() - t:.0f}s")

    for path in paths:
        t = time.time()
        frames, stride = read_frames(path)
        x = clips(frames).to(dev)
        feats = (torch.from_numpy(emb(x)).float() - mu) / sd  # (SEG, 1024) standardized
        with torch.no_grad():
            s = head_a(feats).squeeze(-1)
        bounds = np.linspace(0, len(frames), SEG + 1) / FPS
        print(f"\n=== {os.path.basename(path)}  ({len(frames) / FPS:.0f}s, {time.time() - t:.1f}s to score)")
        print(f"anomaly score (max over segments): {s.max():.3f}  -> {'ANOMALY' if s.max() > THRESH else 'normal'}")
        for i, v in enumerate(s.tolist()):
            print(f"  {bounds[i]:6.1f}-{bounds[i + 1]:6.1f}s  src frames {int(bounds[i] * FPS * stride):5d}+  "
                  f"{v:.2f} {'#' * round(v * 40)}")
        crime = None
        for labels, c, m in heads_b:
            k = c["cfg"]["k"]
            top = s.topk(min(k, SEG)).indices if k < SEG else torch.arange(SEG)
            with torch.no_grad():
                p = m(feats[top].mean(0, keepdim=True)).softmax(-1)[0]
            best = p.argsort(descending=True)[:3].tolist()
            print(f"crime type ({labels}): " + ", ".join(f"{c['classes'][j]} {p[j]:.2f}" for j in best))
            crime = crime or f"{c['classes'][best[0]]} ({p[best[0]]:.0%})"  # first head listed (grouped) wins
        os.makedirs(OUT, exist_ok=True)
        out_path = os.path.join(OUT, os.path.splitext(os.path.basename(path))[0] + "_annotated.mp4")
        write_annotated(path, out_path, s.cpu(), len(frames), stride, crime)
        print(f"annotated video -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
