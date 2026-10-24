#!/usr/bin/env python
# coding: utf-8
"""
revere_websocket_new.py
=======================
Next-generation REVERE detector built on the POC pipeline
(frozen V-JEPA 2 ViT-L + MIL heads), with a polished GUI and live WebSocket
streaming to the REVERE backend.

Pipeline (from poc/):
    video -> 7.5 fps frames, 224x224 -> 32 segments x 16 frames
          -> V-JEPA 2 embeddings (frozen, cached per segment)
          -> Head A  : per-segment anomaly score + "crime vs normal"
          -> Head B  : crime type, from the top-k segments Head A flagged
    If anomalous, the flagged span (+ padding) is saved as a webm clip and
    streamed with metadata to  ws://127.0.0.1:8000/ws/ingest.

Run (same machine as the backend):
    python revere_websocket_new.py

Run (backend on another PC — just pass its IPv4):
    python revere_websocket_new.py --host 192.168.1.42
    python revere_websocket_new.py --backend ws://192.168.1.42:8000/ws/ingest
You can also set the backend from the GUI's "Server connection" field at runtime.

Dependencies: torch, transformers, opencv-python, numpy, websockets  (+ tkinter)
Checkpoints:  poc/ckpt/{headA.pt, headB_grouped.pt, headB_13.pt, feat_norm.npz}
Model:        facebook/vjepa2-vitl-fpc64-256 (downloaded on first run by transformers)

Environment overrides:
    REVERE_POC_DIR      path to the poc/ folder        (default: ./poc)
    REVERE_CLIP_DIR     where clips are written         (default: ./clips)
    REVERE_WS_BACKEND   backend ingest websocket URL    (default: ws://127.0.0.1:8000/ws/ingest)
    REVERE_THRESHOLD    anomaly score threshold         (default: 0.5)
"""

import asyncio
import base64
import datetime
import json
import os
import sys
import threading
import time
from urllib.request import urlopen

import cv2
import numpy as np

try:
    import websockets
except ImportError:
    websockets = None

import tkinter as tk
from tkinter import font as tkfont
from tkinter.filedialog import askopenfilename


# --------------------------------------------------------------------------- #
#  Paths / config
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POC_DIR = os.environ.get("REVERE_POC_DIR", os.path.join(BASE_DIR, "poc"))
CKPT_DIR = os.path.join(POC_DIR, "ckpt")
CLIP_DIR = os.environ.get("REVERE_CLIP_DIR", os.path.join(BASE_DIR, "clips"))
CLIP_PATH = os.path.join(CLIP_DIR, "output_clip.webm")
THRESHOLD = float(os.environ.get("REVERE_THRESHOLD", "0.5"))
DEFAULT_PORT = 8000


def _normalize_backend(value):
    """Accept any of: a bare IP (192.168.1.42), host:port, or full ws:// URL,
    and return a complete  ws://host:port/ws/ingest  URL."""
    if not value:
        return None
    v = value.strip()
    if v.startswith(("ws://", "wss://")):
        return v
    # strip an accidental http(s):// prefix
    if v.startswith("http://"):
        v = v[len("http://"):]
    elif v.startswith("https://"):
        v = v[len("https://"):]
    v = v.rstrip("/")
    if "/" in v:                       # host[:port]/path given
        host, path = v.split("/", 1)
        host = host if ":" in host else f"{host}:{DEFAULT_PORT}"
        return f"ws://{host}/{path}"
    if ":" not in v:                    # bare IP/hostname -> add default port
        v = f"{v}:{DEFAULT_PORT}"
    return f"ws://{v}/ws/ingest"


def _resolve_backend():
    """Backend URL from, in order: --backend/--host CLI flag, REVERE_WS_BACKEND
    env var, else localhost."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ("--backend", "--host") and i + 1 < len(argv):
            return _normalize_backend(argv[i + 1])
        if a.startswith("--backend="):
            return _normalize_backend(a.split("=", 1)[1])
        if a.startswith("--host="):
            return _normalize_backend(a.split("=", 1)[1])
    env = os.environ.get("REVERE_WS_BACKEND")
    if env:
        return _normalize_backend(env)
    return "ws://127.0.0.1:8000/ws/ingest"


WS_BACKEND = _resolve_backend()

# POC constants (mirror poc/extract_emb.py)
FPS = 7.5           # sampling rate used by the POC
SIZE = 224          # frame size fed to V-JEPA 2
SEG = 32            # segments per video
T = 16              # frames per segment
CHUNK = 16          # clips per forward pass
MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"

# Padding for the saved clip around the flagged span (seconds, source-time)
PRE_PAD_SECONDS = 3.0
POST_PAD_SECONDS = 3.0
DEFAULT_FPS = 15

# Live-camera recording window (seconds) before each scoring pass
LIVE_WINDOW_SECONDS = 20

os.makedirs(CLIP_DIR, exist_ok=True)

# make the poc package importable
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)


# --------------------------------------------------------------------------- #
#  Lazy heavy imports (torch + the POC helpers) — loaded on demand
# --------------------------------------------------------------------------- #
torch = None
F = None
clip_starts = None
D = 1024


def _import_backend():
    """Import torch and the POC segmenting helper once."""
    global torch, F, clip_starts, D
    if torch is not None:
        return
    import torch as _torch
    import torch.nn.functional as _F
    # We only reuse the POC's deterministic segment-centering helper + dim.
    from extract_emb import clip_starts as _clip_starts, D as _D
    torch = _torch
    F = _F
    clip_starts = _clip_starts
    D = _D


class VJepaEmbedder:
    """Version-robust V-JEPA 2 embedder.

    Mirrors poc/extract_emb.py's Embedder, but loads the model without the
    `dtype=` kwarg (which only newer transformers accept) and casts to fp16
    afterwards, so it works across transformers 4.52+.
    """

    def __init__(self, dev):
        from transformers import AutoModel, AutoVideoProcessor
        vp = AutoVideoProcessor.from_pretrained(MODEL_NAME)
        self.mean = torch.tensor(vp.image_mean, device=dev).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(vp.image_std, device=dev).view(1, 1, 3, 1, 1)
        # load then cast — avoids the dtype= / torch_dtype= API drift
        model = AutoModel.from_pretrained(MODEL_NAME)
        self.model = model.half().to(dev).eval()
        self.dev = dev

    def __call__(self, x):  # x (SEG, T, 3, S, S) float [0,1] -> (SEG, D) fp16 np
        with torch.no_grad():
            x = ((x - self.mean) / self.std).half()
            out = [self.model(pixel_values_videos=x[i:i + CHUNK], skip_predictor=True)
                   .last_hidden_state.float().mean(1)
                   for i in range(0, SEG, CHUNK)]
            return torch.cat(out).half().cpu().numpy()


# --------------------------------------------------------------------------- #
#  Model wrapper: V-JEPA 2 + Head A + Head B
# --------------------------------------------------------------------------- #
class RevereModel:
    def __init__(self, on_log=None):
        self.on_log = on_log or (lambda *a, **k: None)
        self.ready = False
        self.dev = None
        self.embedder = None
        self.head_a = None
        self.heads_b = []       # list of (labels, cfg, module)
        self.mu = None
        self.sd = None

    def _mlp(self, hid, out, p):
        return torch.nn.Sequential(
            torch.nn.Linear(D, hid), torch.nn.ReLU(),
            torch.nn.Dropout(p), torch.nn.Linear(hid, out))

    def _load_head(self, name, out, hid, p, sigmoid=False):
        m = self._mlp(hid, out, p)
        if sigmoid:
            m = torch.nn.Sequential(m, torch.nn.Sigmoid())
        # trusted project checkpoints; weights_only=False silences the FutureWarning
        state = torch.load(os.path.join(CKPT_DIR, name), map_location="cpu", weights_only=False)
        m.load_state_dict(state)
        return m.eval()

    def load(self):
        _import_backend()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.on_log(f"Device: {self.dev}")

        norm = np.load(os.path.join(CKPT_DIR, "feat_norm.npz"))
        self.mu = torch.tensor(norm["mean"])
        self.sd = torch.tensor(norm["std"])

        self.on_log("Loading Head A / Head B heads…")
        self.head_a = self._load_head("headA.pt", 1, 512, 0.6, sigmoid=True)
        for labels in ("grouped", "13"):
            cfg_path = os.path.join(CKPT_DIR, f"headB_{labels}.json")
            if os.path.exists(cfg_path):
                c = json.load(open(cfg_path))
                m = self._load_head(f"headB_{labels}.pt", len(c["classes"]), 256, 0.5)
                self.heads_b.append((labels, c, m))

        self.on_log("Loading V-JEPA 2 (first run downloads the model)…")
        self.embedder = VJepaEmbedder(self.dev)
        self.ready = True
        self.on_log("Model ready.")

    # ---- inference ----
    def _segment_feats(self, frames):
        """frames: (N, H=224, W=224, 3) uint8 RGB -> standardized (SEG, D) tensor."""
        center, _ = clip_starts(len(frames), np.random.default_rng(0))  # view 0 = center clips
        clips = []
        for s in center:
            x = frames[s:s + T]
            if len(x) < T:
                x = np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)])
            clips.append(x)
        x = torch.from_numpy(np.stack(clips)).permute(0, 1, 4, 2, 3).float() / 255  # (SEG, T, 3, S, S)
        feats = torch.from_numpy(self.embedder(x.to(self.dev))).float()
        return (feats - self.mu) / self.sd  # (SEG, D) standardized

    def score(self, frames):
        """Run Head A + Head B. Returns dict with per-segment scores, max score, crime type."""
        feats = self._segment_feats(frames)
        with torch.no_grad():
            seg_scores = self.head_a(feats).squeeze(-1)  # (SEG,)
        max_score = float(seg_scores.max())

        crime = "Normal"
        if max_score > THRESHOLD and self.heads_b:
            labels, c, m = self.heads_b[0]  # grouped head first
            k = c["cfg"]["k"]
            top = seg_scores.topk(min(k, SEG)).indices if k < SEG else torch.arange(SEG)
            with torch.no_grad():
                probs = m(feats[top].mean(0, keepdim=True)).softmax(-1)[0]
            crime = c["classes"][int(probs.argmax())]

        return {
            "seg_scores": seg_scores.cpu().numpy(),
            "max_score": max_score,
            "crime": crime,
            "anomalous": max_score > THRESHOLD,
        }


# --------------------------------------------------------------------------- #
#  Video helpers
# --------------------------------------------------------------------------- #
def read_frames(path):
    """Decode at ~FPS (7.5), resized to 224x224 RGB. Returns (frames_224, stride, src_fps)."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(src_fps / FPS))
    frames, n = [], 0
    while cap.grab():
        if n % stride == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            f = cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        n += 1
    cap.release()
    if not frames:
        return np.zeros((0, SIZE, SIZE, 3), np.uint8), stride, src_fps
    return np.stack(frames), stride, src_fps


def save_clip_from_source(src_path, seg_scores, out_path):
    """Cut a contiguous webm clip from the ORIGINAL video spanning the flagged
    segments (+ padding), at the source's real fps. Returns (ok, n_frames, fps)."""
    flagged = np.where(seg_scores > THRESHOLD)[0]
    if len(flagged) == 0:
        return False, 0, DEFAULT_FPS

    cap = cv2.VideoCapture(src_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # map flagged segments -> source-frame span
    seg_first, seg_last = int(flagged.min()), int(flagged.max())
    start = int(seg_first / SEG * total) - int(PRE_PAD_SECONDS * src_fps)
    end = int((seg_last + 1) / SEG * total) + int(POST_PAD_SECONDS * src_fps)
    start = max(0, start)
    end = min(total - 1, end) if total else end

    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    writer = cv2.VideoWriter(out_path, fourcc, float(src_fps), (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    n = 0
    idx = start
    while True:
        ok, frame = cap.read()
        if not ok or idx > end:
            break
        writer.write(frame)
        n += 1
        idx += 1
    writer.release()
    cap.release()
    return n > 0, n, src_fps


def save_clip_from_frames(frames_bgr, out_path, fps):
    """Write a list of BGR frames to a webm clip (used for live camera)."""
    if not frames_bgr:
        return False
    h, w, _ = frames_bgr[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    for f in frames_bgr:
        writer.write(f)
    writer.release()
    return True


# --------------------------------------------------------------------------- #
#  Location + time
# --------------------------------------------------------------------------- #
def get_location():
    try:
        data = json.load(urlopen("http://ipinfo.io/json", timeout=5))
        return (data.get("city", "NA"),
                data.get("region", data.get("org", "NA")),
                data.get("loc", "NA"))
    except Exception:
        return "NA", "NA", "NA"


def get_date_and_time():
    now = datetime.datetime.now()
    return now.strftime("%Y_%m_%d"), now.strftime("%H_%M_%S")


# --------------------------------------------------------------------------- #
#  WebSocket sender (thread-safe)
# --------------------------------------------------------------------------- #
async def _ws_send(payload, server):
    async with websockets.connect(server, max_size=None) as ws:
        await ws.send(json.dumps(payload))
        return await ws.recv()


async def _ws_connect_only(server):
    async with websockets.connect(server, max_size=None, open_timeout=8):
        return True


def _ws_ping(server):
    """Open+close a websocket to confirm reachability. Returns None on success,
    else an error string."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_ws_connect_only(server))
        return None
    except Exception as e:  # noqa: BLE001
        return str(e)
    finally:
        loop.close()


def _ws_send_blocking(payload, server, result):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result["reply"] = loop.run_until_complete(_ws_send(payload, server))
    except Exception as e:  # noqa: BLE001
        result["error"] = str(e)
    finally:
        loop.close()


def send_detection(crime, city, locality, coordinates, date, time_str, video_path, confidence=None):
    if websockets is None:
        return None
    payload = {
        "crime": crime, "location": city, "locality": locality,
        "coordinates": coordinates, "date": date, "time": time_str,
    }
    if confidence is not None:
        payload["confidence"] = round(float(confidence), 3)
    if video_path and os.path.exists(video_path):
        try:
            with open(video_path, "rb") as f:
                payload["video_b64"] = base64.b64encode(f.read()).decode("ascii")
            payload["filename"] = os.path.basename(video_path)
        except Exception:  # noqa: BLE001
            pass
    result = {}
    t = threading.Thread(target=_ws_send_blocking, args=(payload, WS_BACKEND, result))  # noqa: E501 (uses live global)
    t.start()
    t.join(60)
    if result.get("error"):
        return None
    return result.get("reply")


# --------------------------------------------------------------------------- #
#  Analyzers  (report progress back to the GUI via on_event)
# --------------------------------------------------------------------------- #
class FileAnalyzer:
    def __init__(self, model, path, on_event=None):
        self.model = model
        self.path = path
        self.on_event = on_event or (lambda *a, **k: None)

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        t0 = time.time()
        name = os.path.basename(self.path)
        self.on_event("start", f"Reading {name} @ {FPS} fps…")
        frames, stride, src_fps = read_frames(self.path)
        if len(frames) == 0:
            self.on_event("error", "Could not read any frames from the video.")
            self.on_event("done", "")
            return

        self.on_event("start", f"Embedding {len(frames)} frames with V-JEPA 2…")
        res = self.model.score(frames)

        if not res["anomalous"]:
            self.on_event("none",
                          f"No crime detected (max score {res['max_score']:.2f} < {THRESHOLD}).")
            self.on_event("done", f"Done in {time.time()-t0:.1f}s")
            return

        self.on_event("detect",
                      f"⚠ {res['crime']} detected ({res['max_score']*100:.0f}% conf.)")

        ok, n, clip_fps = save_clip_from_source(self.path, res["seg_scores"], CLIP_PATH)
        if not ok:
            self.on_event("error", "Failed to cut the incident clip.")
            self.on_event("done", "")
            return
        self.on_event("clip", f"Clip cut: {n} frames (~{n/clip_fps:.1f}s) at {clip_fps:.0f} fps")

        self._deliver(res["crime"], res["max_score"])
        self.on_event("done", f"Done in {time.time()-t0:.1f}s")

    def _deliver(self, crime, conf):
        city, locality, coordinates = get_location()
        date, time_str = get_date_and_time()
        archive = CLIP_PATH
        try:
            archive = os.path.join(CLIP_DIR, f"{city}_{date}+{time_str}.webm")
            os.replace(CLIP_PATH, archive)
        except Exception:  # noqa: BLE001
            archive = CLIP_PATH
        self.on_event("sending", f"Streaming '{crime}' @ {city}/{locality}…")
        reply = send_detection(crime, city, locality, coordinates, date, time_str, archive, conf)
        if reply:
            self.on_event("sent", f"Delivered · {crime} @ {city}")
        else:
            self.on_event("error", "Backend unreachable — is the server running?")


class CameraAnalyzer:
    """Records rolling windows from the webcam, scores each with the POC model,
    and streams any anomaly. Press q in the preview window to stop."""

    def __init__(self, model, on_event=None, cam_index=0):
        self.model = model
        self.on_event = on_event or (lambda *a, **k: None)
        self.cam_index = cam_index
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False

    def _run(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            self.on_event("error", "Could not open the camera.")
            self.on_event("done", "")
            return
        cam_fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        sample_stride = max(1, round(cam_fps / FPS))
        window_frames = int(LIVE_WINDOW_SECONDS * cam_fps)

        self.on_event("start", f"Live camera · scoring every {LIVE_WINDOW_SECONDS}s window…")
        buf_bgr = []            # full-res BGR frames for clip saving
        buf_small = []          # 224 RGB frames for the model
        n = 0
        while self.running:
            ok, frame = cap.read()
            if not ok:
                break
            buf_bgr.append(frame)
            if n % sample_stride == 0:
                small = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                buf_small.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            n += 1

            cv2.imshow("REVERE — Live Camera (press q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if len(buf_bgr) >= window_frames and len(buf_small) >= SEG:
                self._score_window(list(buf_small), list(buf_bgr), cam_fps)
                buf_bgr.clear()
                buf_small.clear()

        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        self.on_event("done", "Camera stopped.")

    def _score_window(self, small_rgb, bgr, cam_fps):
        res = self.model.score(np.stack(small_rgb))
        if not res["anomalous"]:
            self.on_event("none", f"Window clear (max {res['max_score']:.2f}).")
            return
        self.on_event("detect", f"⚠ {res['crime']} detected ({res['max_score']*100:.0f}% conf.)")
        if save_clip_from_frames(bgr, CLIP_PATH, cam_fps):
            self.on_event("clip", f"Clip saved ({len(bgr)} frames)")
            city, locality, coordinates = get_location()
            date, time_str = get_date_and_time()
            archive = CLIP_PATH
            try:
                archive = os.path.join(CLIP_DIR, f"{city}_{date}+{time_str}.webm")
                os.replace(CLIP_PATH, archive)
            except Exception:  # noqa: BLE001
                archive = CLIP_PATH
            self.on_event("sending", f"Streaming '{res['crime']}' @ {city}/{locality}…")
            reply = send_detection(res["crime"], city, locality, coordinates,
                                   date, time_str, archive, res["max_score"])
            self.on_event("sent" if reply else "error",
                          f"Delivered · {res['crime']} @ {city}" if reply
                          else "Backend unreachable — is the server running?")


# --------------------------------------------------------------------------- #
#  GUI — dark REVERE-branded control panel
# --------------------------------------------------------------------------- #
BG, PANEL, PANEL2, LINE = "#0b1220", "#131c2b", "#1a2538", "#243247"
TEXT, MUTED, TEAL, BLUE = "#e8eef6", "#8aa0b8", "#2dd4bf", "#60a5fa"
OK, WARN, DANGER = "#22c55e", "#f59e0b", "#ef4444"


class RevereApp:
    def __init__(self, root):
        self.root = root
        self.model = RevereModel(on_log=lambda m: self.on_event("log", m))
        self.camera = None

        root.title("REVERE — V-JEPA 2 Detector")
        root.geometry("700x660")
        root.configure(bg=BG)
        root.minsize(640, 580)

        self.f_title = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        self.f_sub = tkfont.Font(family="Segoe UI", size=10)
        self.f_lbl = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.f_small = tkfont.Font(family="Segoe UI", size=9)
        self.f_mono = tkfont.Font(family="Consolas", size=9)

        self._header()
        self._status()
        self._server()
        self._actions()
        self._log()
        self._footer()

        self.log("Starting up…", MUTED)
        self.log(f"Backend: {WS_BACKEND}", MUTED)
        self.log(f"Checkpoints: {CKPT_DIR}", MUTED)
        if websockets is None:
            self.log("'websockets' not installed — detections can't be sent.", DANGER)

        # load the (heavy) model in the background
        threading.Thread(target=self._load_model, daemon=True).start()

    # ---- build ----
    def _header(self):
        bar = tk.Frame(self.root, bg="#0d1627", height=84)
        bar.pack(fill="x"); bar.pack_propagate(False)
        wrap = tk.Frame(bar, bg="#0d1627"); wrap.pack(side="left", padx=24, pady=16)
        eye = tk.Canvas(wrap, width=44, height=44, bg="#0d1627", highlightthickness=0)
        eye.create_oval(4, 4, 40, 40, fill=TEAL, outline="")
        eye.create_oval(15, 15, 29, 29, fill="#0d1627", outline="")
        eye.pack(side="left", padx=(0, 14))
        t = tk.Frame(wrap, bg="#0d1627"); t.pack(side="left")
        tk.Label(t, text="REVERE", font=self.f_title, fg=TEXT, bg="#0d1627").pack(anchor="w")
        tk.Label(t, text="V-JEPA 2 Crime Detection · Desktop Agent", font=self.f_sub,
                 fg=MUTED, bg="#0d1627").pack(anchor="w")
        pill = tk.Frame(bar, bg="#0d1627"); pill.pack(side="right", padx=24)
        self.dot = tk.Canvas(pill, width=12, height=12, bg="#0d1627", highlightthickness=0)
        self._dot("#555"); self.dot.pack(side="left", padx=(0, 7), pady=4)
        self.conn = tk.Label(pill, text="loading…", font=self.f_small, fg=MUTED, bg="#0d1627")
        self.conn.pack(side="left")

    def _status(self):
        card = self._card("CURRENT STATUS")
        self.status_var = tk.StringVar(value="Loading model…")
        tk.Label(card, textvariable=self.status_var, font=self.f_lbl, fg=TEXT, bg=PANEL,
                 anchor="w", justify="left", wraplength=620).pack(fill="x", padx=18, pady=(2, 10))
        row = tk.Frame(card, bg=PANEL); row.pack(fill="x", padx=18, pady=(0, 14))
        self.chip = tk.Label(row, text="No incident", font=self.f_small, fg=MUTED, bg=PANEL2, padx=12, pady=5)
        self.chip.pack(side="left")

    def _server(self):
        card = self._card("SERVER CONNECTION")
        row = tk.Frame(card, bg=PANEL); row.pack(fill="x", padx=18, pady=(2, 8))
        tk.Label(row, text="Backend", font=self.f_small, fg=MUTED, bg=PANEL).pack(side="left", padx=(0, 8))
        self.backend_var = tk.StringVar(value=WS_BACKEND)
        entry = tk.Entry(row, textvariable=self.backend_var, font=self.f_mono, bg="#0e1726",
                         fg=TEXT, insertbackground=TEXT, bd=0, highlightthickness=1,
                         highlightbackground=LINE, highlightcolor=BLUE)
        entry.pack(side="left", expand=True, fill="x", ipady=5, padx=(0, 8))
        tk.Button(row, text="Apply", command=self._apply_backend, font=self.f_small,
                  fg="#06101f", bg=BLUE, bd=0, cursor="hand2", padx=12).pack(side="left", padx=(0, 6))
        tk.Button(row, text="Test", command=self._test_backend, font=self.f_small,
                  fg=TEXT, bg=PANEL2, bd=0, cursor="hand2", padx=12).pack(side="left")
        tk.Label(card,
                 text="On another PC, enter the server's IPv4 (e.g. 192.168.1.42) and click Apply.",
                 font=self.f_small, fg=MUTED, bg=PANEL, wraplength=620, justify="left").pack(
                 anchor="w", padx=18, pady=(0, 14))

    def _apply_backend(self):
        global WS_BACKEND
        url = _normalize_backend(self.backend_var.get())
        if not url:
            self.log("Enter an IP or URL first.", WARN)
            return
        WS_BACKEND = url
        self.backend_var.set(url)
        self.log(f"Backend set to {url}", OK)

    def _test_backend(self):
        self._apply_backend()
        self.log(f"Testing {WS_BACKEND} …", MUTED)
        threading.Thread(target=self._test_backend_worker, args=(WS_BACKEND,), daemon=True).start()

    def _test_backend_worker(self, url):
        if websockets is None:
            self.on_event("error", "'websockets' not installed.")
            return
        err = _ws_ping(url)
        if err:
            self.on_event("error", f"Cannot reach backend: {err}")
        else:
            self.on_event("sent", "Backend reachable ✓")

    def _actions(self):
        card = self._card("DETECTION SOURCE")
        row = tk.Frame(card, bg=PANEL); row.pack(fill="x", padx=18, pady=(4, 10))
        self.btn_file = tk.Button(row, text="📁  Analyze Video File", command=self.on_file,
                                  font=self.f_lbl, fg="#06101f", bg=BLUE, bd=0, cursor="hand2",
                                  state="disabled")
        self.btn_file.pack(side="left", expand=True, fill="x", padx=(0, 8), ipady=10)
        self.btn_cam = tk.Button(row, text="📷  Live Camera", command=self.on_camera,
                                 font=self.f_lbl, fg="#06101f", bg=TEAL, bd=0, cursor="hand2",
                                 state="disabled")
        self.btn_cam.pack(side="left", expand=True, fill="x", padx=(8, 0), ipady=10)
        tk.Label(card, text=f"Head A flags anomalies (threshold {THRESHOLD}); Head B names the crime. "
                            "Press q in the preview to stop.",
                 font=self.f_small, fg=MUTED, bg=PANEL, wraplength=620, justify="left").pack(
                 anchor="w", padx=18, pady=(0, 14))

    def _log(self):
        card = self._card("ACTIVITY LOG", grow=True)
        frame = tk.Frame(card, bg=PANEL2, highlightbackground=LINE, highlightthickness=1)
        frame.pack(fill="both", expand=True, padx=18, pady=(2, 16))
        self.logbox = tk.Text(frame, bg=PANEL2, fg=TEXT, font=self.f_mono, bd=0,
                              padx=12, pady=10, wrap="word", state="disabled", height=9)
        self.logbox.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(frame, command=self.logbox.yview); sb.pack(side="right", fill="y")
        self.logbox.config(yscrollcommand=sb.set)

    def _footer(self):
        f = tk.Frame(self.root, bg=BG); f.pack(fill="x", side="bottom")
        tk.Label(f, text="REVERE · detections are reviewed by officials before going public",
                 font=self.f_small, fg=MUTED, bg=BG).pack(pady=8)

    def _card(self, title, grow=False):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both" if grow else "x", expand=grow, padx=20, pady=(14, 0))
        card = tk.Frame(outer, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        card.pack(fill="both", expand=True)
        tk.Label(card, text=title, font=self.f_small, fg=MUTED, bg=PANEL).pack(
            anchor="w", padx=18, pady=(12, 2))
        return card

    def _dot(self, color):
        self.dot.delete("all")
        self.dot.create_oval(1, 1, 11, 11, fill=color, outline="")

    # ---- model load ----
    def _load_model(self):
        try:
            self.model.load()
            self.root.after(0, self._model_ready)
        except Exception as e:  # noqa: BLE001
            self.on_event("error", f"Model load failed: {e}")

    def _model_ready(self):
        self.set_conn("ready", OK)
        self.set_status("Ready — choose a source to begin")
        self.btn_file.config(state="normal")
        self.btn_cam.config(state="normal")

    # ---- logging + status ----
    def log(self, msg, color=TEXT):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.logbox.config(state="normal")
        tag = f"c{self.logbox.index('end')}"
        self.logbox.insert("end", f"[{ts}] ", ("muted",))
        self.logbox.insert("end", msg + "\n", (tag,))
        self.logbox.tag_config("muted", foreground=MUTED)
        self.logbox.tag_config(tag, foreground=color)
        self.logbox.see("end")
        self.logbox.config(state="disabled")

    def set_status(self, text):
        self.status_var.set(text)

    def set_conn(self, text, color):
        self.conn.config(text=text, fg=color)
        self._dot(color)

    # ---- event bridge (worker thread -> UI thread) ----
    def on_event(self, kind, message):
        self.root.after(0, lambda: self._handle(kind, message))

    def _handle(self, kind, message):
        colors = {"log": MUTED, "start": BLUE, "detect": WARN, "clip": TEAL,
                  "sending": BLUE, "sent": OK, "error": DANGER, "none": MUTED, "done": OK}
        if message:
            self.log(message, colors.get(kind, TEXT))
            if kind != "log":
                self.set_status(message)
        if kind == "start":
            self.set_conn("working", WARN); self._busy(True)
        elif kind == "detect":
            crime = message.split()[1]
            self.chip.config(text=f"⚠ {crime}", bg="#2a1313", fg="#fca5a5")
        elif kind == "sent":
            self.set_conn("delivered", OK)
        elif kind == "error":
            self.set_conn("error", DANGER)
        elif kind == "done":
            self.set_conn("ready", OK); self._busy(False)

    def _busy(self, busy):
        # camera keeps its own stop button via the preview window; just gate file btn
        self.btn_file.config(state="disabled" if busy else "normal")

    # ---- actions ----
    def on_file(self):
        if not self.model.ready:
            return
        path = askopenfilename(title="Select a video file",
                               filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm")])
        if path:
            self.chip.config(text="No incident", bg=PANEL2, fg=MUTED)
            FileAnalyzer(self.model, path, on_event=self.on_event).start()

    def on_camera(self):
        if not self.model.ready:
            return
        self.chip.config(text="No incident", bg=PANEL2, fg=MUTED)
        self.camera = CameraAnalyzer(self.model, on_event=self.on_event)
        self.camera.start()


def main():
    root = tk.Tk()
    RevereApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
