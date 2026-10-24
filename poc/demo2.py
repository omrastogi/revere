"""Phase-2 inference following inference_flowchart.excalidraw, replaying a video file as if it were a live stream.

python poc/demo2.py <video.mp4> [...]   (heads in poc/heads/, standard dictionary + its embeddings in poc/dictionary/)
  -> poc/examples/out/<name>_live.mp4    dashboard view: feed + zone + alert card + score graph
     poc/examples/out/<name>_alerts.json alert cards and the per-clip log
     poc/examples/out/<name>_alert<k>.mp4 4 s around each alert's peak, faces blurred
Per clip (rolling buffer of 16 frames at 7.5 fps, every 8 frames): V-JEPA 2 (fp16, mean-pooled 1024-d) -> Head A
P(>=yellow), P(red) -> mean of the last 4 clips + hysteresis -> GREEN: log only | YELLOW/RED: Head C (alignment head,
1536-d) -> top-3 dictionary phrases = explanation + severity | RED only: Head B (13 classes), type if top-1 >= 0.6 else
"Unclassified incident" -> alert; the same event within 10 s updates (and escalates) the open alert.
Deviation: V-JEPA 2 is frozen, not fine-tuned (no fine-tuned checkpoint exists yet).
"""
import collections, json, os, sys, time

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import T, Encoder, Head, load_dictionary, match_dictionary

POC = os.path.dirname(os.path.abspath(__file__))
HEADS, OUT = f"{POC}/heads", f"{POC}/examples/out"
FPS, SIZE, STRIDE, BATCH = 7.5, 224, 8, 4
# ponytail: thresholds 0.5, not head_a_zone.pt's tuned 0.10/0.05: with t_red 0.05, "exit below t_red - 0.1" never happens
T_YELLOW, T_RED, HYST, SMOOTH, TYPE_MIN, DEDUP_S, CLIP_S = 0.5, 0.5, 0.1, 4, 0.6, 10.0, 4.0
COLOR = {"GREEN": (0, 170, 0), "YELLOW": (0, 200, 255), "RED": (0, 0, 255)}  # BGR
FACES = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def read_frames(path):
    """Flowchart preprocess: center-crop to 4:3, resize to 224x224, sample at 7.5 fps. -> (N,224,224,3) RGB, stride, fps"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(fps / FPS))
    frames, n = [], 0
    while cap.grab():
        if n % stride == 0:
            img = cap.retrieve()[1]
            h, w = img.shape[:2]
            if w * 3 > h * 4:
                cw = round(h * 4 / 3)
                img = img[:, (w - cw) // 2:(w - cw) // 2 + cw]
            else:
                ch = round(w * 3 / 4)
                img = img[(h - ch) // 2:(h - ch) // 2 + ch]
            frames.append(cv2.cvtColor(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        n += 1
    return np.stack(frames), stride, fps


def mlp_head(name, dev):
    ck = torch.load(f"{HEADS}/{name}", map_location="cpu", weights_only=False)  # our own checkpoints (numpy str class names)
    m = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, ck["hidden"]), nn.GELU(), nn.Dropout(ck["dropout"]),
                      nn.Linear(ck["hidden"], ck["state"]["4.weight"].shape[0]))
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck


class Pipeline:
    """Everything after the V-JEPA pass; step() takes one clip embedding, exactly as a live loop would."""

    def __init__(self, dev):
        self.head_a, _ = mlp_head("head_a_zone.pt", dev)
        self.head_b, cb = mlp_head("head_b_13.pt", dev)
        self.classes = cb["classes"]
        self.head_c = Head(torch.zeros(1024), torch.ones(1024))  # alignment head: 1024-d clip -> 1536-d text space
        self.head_c.load_state_dict(torch.load(f"{HEADS}/head_c_explain.pt", map_location="cpu")["state"])
        self.head_c = self.head_c.to(dev).eval()
        self.concepts, self.D, self.owner = load_dictionary(dev, f"{POC}/dictionary")
        self.buf, self.red, self.yel, self.alerts = collections.deque(maxlen=SMOOTH), False, False, []

    @torch.no_grad()
    def step(self, x, t):
        """x: (1024,) embedding of the clip that ends at t seconds -> per-clip log record."""
        x = x[None]
        self.buf.append(self.head_a(x).sigmoid()[0].cpu().numpy())
        py, pr = np.mean(self.buf, 0)
        self.red = bool(pr > (T_RED - HYST if self.red else T_RED))  # hysteresis: enter > t, exit < t - 0.1
        self.yel = bool(py > (T_YELLOW - HYST if self.yel else T_YELLOW))
        zone = "RED" if self.red else "YELLOW" if self.yel else "GREEN"
        rec = dict(t=round(t, 2), zone=zone, p_yellow=float(py), p_red=float(pr))
        if zone == "GREEN":
            return rec
        S = match_dictionary(self.head_c(x)[0], self.D, self.owner, len(self.concepts))[0]
        top = S.topk(3).indices.tolist()
        rec["explanation"] = [dict(phrase=self.concepts[k]["phrase"], zone=self.concepts[k]["zone"],
                                   severity=self.concepts[k]["severity"], cos=round(float(S[k]), 3)) for k in top]
        rec["severity"] = self.concepts[top[0]]["severity"]
        if zone == "RED":
            pb = self.head_b(x).softmax(1)[0]
            k = int(pb.argmax())
            rec["type_p"] = round(float(pb[k]), 3)
            rec["type"] = self.classes[k] if pb[k] >= TYPE_MIN else "Unclassified incident"
        self.alert(rec)
        return rec

    def alert(self, rec):
        a = self.alerts[-1] if self.alerts else None
        if a and rec["t"] - a["t_last"] <= DEDUP_S:  # same event: update the open alert, escalate yellow -> red
            a["t_last"], a["updates"] = rec["t"], a["updates"] + 1
            a["max_severity"] = max(a["max_severity"], rec["severity"])
            if rec["zone"] == "RED" and a["zone"] == "YELLOW":
                a.update(zone="RED", escalated_at=rec["t"])
            if rec.get("type") and a.get("type") in (None, "Unclassified incident"):
                a["type"] = rec["type"]
            if rec["p_red"] + rec["p_yellow"] > a["peak_score"]:
                a.update(peak_t=rec["t"], peak_score=rec["p_red"] + rec["p_yellow"], explanation=rec["explanation"],
                         severity=rec["severity"])
        else:
            a = dict(id=len(self.alerts), t_start=rec["t"], t_last=rec["t"], zone=rec["zone"], type=rec.get("type"),
                     explanation=rec["explanation"], severity=rec["severity"], max_severity=rec["severity"], peak_t=rec["t"],
                     peak_score=rec["p_red"] + rec["p_yellow"], updates=0)
            self.alerts.append(a)
        rec["alert"] = a["id"]


def text(img, s, y, color, scale, thick=2, x=8):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def graph(log, dur, W, H=140):
    """Score graph strip: smoothed P(>=yellow) and P(red) over the whole video, zone bands, threshold lines."""
    g = np.full((H, W, 3), 30, np.uint8)
    X = lambda t: int(t / max(dur, 1e-6) * (W - 1))
    Y = lambda p: int((1 - p) * (H - 20)) + 10
    for a, b in zip(log, log[1:] + [dict(t=dur)]):
        cv2.rectangle(g, (X(a["t"]), H - 8), (X(b["t"]), H - 1), COLOR[a["zone"]], -1)
    for p, col in ((T_YELLOW, COLOR["YELLOW"]), (T_RED, COLOR["RED"])):
        cv2.line(g, (0, Y(p)), (W - 1, Y(p)), tuple(c // 2 for c in col), 1)
    for key, col in (("p_yellow", COLOR["YELLOW"]), ("p_red", COLOR["RED"])):
        pts = np.array([[X(r["t"]), Y(r[key])] for r in log], np.int32)
        if len(pts) > 1:
            cv2.polylines(g, [pts], False, col, 2, cv2.LINE_AA)
    text(g, "P(>=yellow)", 14, COLOR["YELLOW"], 0.4, 1)
    text(g, "P(red)", 14, COLOR["RED"], 0.4, 1, x=100)
    return g


def blur_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for x, y, w, h in FACES.detectMultiScale(gray, 1.1, 4, minSize=(12, 12)):
        img[y:y + h, x:x + w] = cv2.GaussianBlur(img[y:y + h, x:x + w], (0, 0), max(w, h) / 4)
    return img


def render(path, name, log, alerts):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = max(1.0, 720 / w)
    W, H = int(w * s) // 2 * 2, int(h * s) // 2 * 2
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    g = graph(log, dur, W)
    live = cv2.VideoWriter(f"{OUT}/{name}_live.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + g.shape[0]))
    clips = {a["id"]: cv2.VideoWriter(f"{OUT}/{name}_alert{a['id']}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
             for a in alerts}
    ts = np.array([r["t"] for r in log])
    sc, f = 0.35 + 0.25 * s, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        tf = f / fps
        for a in alerts:  # 4 s around the alert's peak clip (clip centre = end time - ~1 s), faces blurred
            if abs(tf - (a["peak_t"] - T / FPS / 2)) <= CLIP_S / 2:
                clips[a["id"]].write(blur_faces(img.copy()))
        img = cv2.resize(img, (W, H))
        i = int(np.searchsorted(ts, tf, "right")) - 1  # latest clip already finished at this frame (live semantics)
        lh = int(30 * sc)
        if i < 0:
            text(img, "buffering...", lh, (200, 200, 200), sc)
        else:
            r = log[i]
            col = COLOR[r["zone"]]
            cv2.rectangle(img, (0, 0), (W - 1, H - 1), col, max(3, W // 120))
            if r["zone"] == "GREEN":
                text(img, "GREEN  normal", lh, col, sc * 1.1, 3)
            else:
                head = f"RED ALERT: {r['type']}" if r["zone"] == "RED" else "YELLOW: Suspicious activity"
                text(img, head, lh, col, sc * 1.1, 3)
                text(img, f"severity {r['severity']}/5   alert #{r['alert']}", 2 * lh, col, sc * 0.8)
                for j, e in enumerate(r["explanation"]):
                    text(img, f"{e['cos']:.2f}  {e['phrase']}", int((2.9 + 0.85 * j) * lh), COLOR[e["zone"]], sc * 0.7, 1)
        text(img, f"t={tf:5.1f}s", H - 10, (255, 255, 255), sc * 0.7, 1)
        gi = g.copy()
        cv2.line(gi, (int(tf / max(dur, 1e-6) * (W - 1)), 0), (int(tf / max(dur, 1e-6) * (W - 1)), gi.shape[0] - 1),
                 (255, 255, 255), 1)
        live.write(np.vstack([img, gi]))
        f += 1
    live.release()
    for c in clips.values():
        c.release()


def main(paths):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.time()
    enc = Encoder(dev, torch.float16)
    print(f"V-JEPA 2 (fp16) + heads A/B/C + dictionary loaded on {dev} in {time.time() - t:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    for path in paths:
        t = time.time()
        name = os.path.splitext(os.path.basename(path))[0]
        frames, stride, fps = read_frames(path)
        n = len(frames)
        ends = list(range(T, n + 1, STRIDE)) or [n]  # rolling buffer: a clip every 8 frames once 16 are buffered
        idx = [np.clip(np.arange(e - T, e), 0, n - 1) for e in ends]
        X = torch.cat([enc(torch.from_numpy(frames[np.stack(idx[i:i + BATCH])])) for i in range(0, len(idx), BATCH)]).to(dev)
        pipe = Pipeline(dev)
        log = [pipe.step(X[i], e * stride / fps) for i, e in enumerate(ends)]
        print(f"\n=== {name}: {n / FPS:.0f}s, {len(log)} clips, {(time.time() - t) / len(log) * 1000:.0f} ms/clip")
        for r in log:
            extra = (f"{r.get('type', ''):22} sev {r['severity']} | {r['explanation'][0]['phrase']}"
                     if r["zone"] != "GREEN" else "")
            print(f"  t={r['t']:6.1f}s  Py {r['p_yellow']:.2f} Pr {r['p_red']:.2f}  {r['zone']:6} {extra}")
        for a in pipe.alerts:
            print(f"  ALERT #{a['id']} {a['zone']:6} {a['t_start']:.1f}-{a['t_last']:.1f}s type={a['type']} "
                  f"max severity {a['max_severity']} | at peak: {a['explanation'][0]['phrase']} (peak t={a['peak_t']:.1f}s, {a['updates']} updates)")
        json.dump(dict(alerts=pipe.alerts, log=log), open(f"{OUT}/{name}_alerts.json", "w"), indent=1)
        render(path, name, log, pipe.alerts)
        print(f"-> {OUT}/{name}_live.mp4 (+ {len(pipe.alerts)} alert clips)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
