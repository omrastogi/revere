#!/usr/bin/env python
# coding: utf-8
"""
revere_websocket_camera.py
==========================
REVERE detector built on the Phase-2 pipeline (frozen V-JEPA 2 ViT-L + zone /
type / explanation heads), with the SAME polished GUI and flow as
revere_websocket_new.py — just powered by the newer Phase-2 weights.

Pipeline (from poc/, Phase 2):
    video -> 7.5 fps frames, 224x224 -> 32 segments x 16 frames
          -> V-JEPA 2 embeddings (frozen, per segment)
          -> Head A (zone) : per-segment P(>=yellow), P(red)  [anomaly score = P(red)]
          -> Head B (type) : crime type from the flagged segments
          -> Head C (expl) : nearest dictionary phrases -> explanation + severity
    If anomalous, the flagged span (+ padding) is saved as a webm clip and
    streamed with metadata to  ws://127.0.0.1:8000/ws/ingest.

Run (same machine as the backend):
    python revere_websocket_camera.py

Run (backend on another PC — just pass its IPv4):
    python revere_websocket_camera.py --host 192.168.1.42
    python revere_websocket_camera.py --backend ws://192.168.1.42:8000/ws/ingest
You can also set the backend from the GUI's "Server connection" field at runtime.

Dependencies: torch, transformers>=4.52, opencv-python, numpy, websockets  (+ tkinter)
Checkpoints:  poc/heads/{head_a_zone.pt, head_b_13.pt, head_c_explain.pt}
              poc/dictionary/{dictionary.jsonl, dictionary.npy, dictionary_ids.json}
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
HEADS_DIR = os.path.join(POC_DIR, "heads")
DICT_DIR = os.path.join(POC_DIR, "dictionary")
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
nn = None
clip_starts = None
Encoder = None
Head = None
load_dictionary = None
match_dictionary = None
D = 1024


F = None


def _import_backend():
    """Import torch and the Phase-2 POC helpers once."""
    global torch, nn, F, clip_starts, Head, load_dictionary, match_dictionary, D
    if torch is not None:
        return
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from extract_emb import clip_starts as _clip_starts, D as _D
    # We reuse the POC's alignment head + dictionary matcher, but NOT its Encoder
    # (that one loads with dtype=, which older transformers reject). We use the
    # version-robust VJepaEncoder below instead.
    from align import Head as _Head, load_dictionary as _ld, match_dictionary as _md
    torch = _torch
    nn = _nn
    F = _F
    clip_starts = _clip_starts
    Head = _Head
    load_dictionary = _ld
    match_dictionary = _md
    D = _D


class VJepaEncoder:
    """Version-robust V-JEPA 2 encoder (same forward as poc/align.py's Encoder,
    but loads the model without the `dtype=` kwarg and casts afterwards, so it
    works across transformers 4.52+). Takes (B, T, H, W, 3) uint8 -> (B, 1024)."""

    def __init__(self, dev):
        from transformers import AutoModel, AutoVideoProcessor
        vp = AutoVideoProcessor.from_pretrained(MODEL_NAME)
        self.mean = torch.tensor(vp.image_mean, device=dev).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(vp.image_std, device=dev).view(1, 1, 3, 1, 1)
        model = AutoModel.from_pretrained(MODEL_NAME)
        self.model = model.half().to(dev).eval()
        self.dev = dev

    def __call__(self, x):  # (B, T, H, W, 3) uint8 -> (B, 1024) float32 on device
        with torch.no_grad():
            b = x.shape[0]
            x = x.to(self.dev).permute(0, 1, 4, 2, 3).flatten(0, 1).float() / 255
            x = F.interpolate(x, size=(SIZE, SIZE), mode="bicubic", align_corners=False)
            x = x.clamp(0, 1).view(b, T, 3, SIZE, SIZE)
            x = ((x - self.mean) / self.std).half()
            return self.model(pixel_values_videos=x, skip_predictor=True) \
                .last_hidden_state.float().mean(1)


# --------------------------------------------------------------------------- #
#  Model wrapper: V-JEPA 2 + Head A (zone) + Head B (type) + Head C (explain)
# --------------------------------------------------------------------------- #
class RevereModel:
    def __init__(self, on_log=None):
        self.on_log = on_log or (lambda *a, **k: None)
        self.ready = False
        self.dev = None
        self.encoder = None
        self.head_a = None       # zone: outputs [P(>=yellow), P(red)]
        self.head_b = None       # crime type (13 classes)
        self.classes = []
        self.head_c = None       # alignment head -> dictionary explanation
        self.concepts = None
        self.D = None
        self.owner = None

    def _mlp_head(self, name):
        """Load a zone_heads.py MLP head: LayerNorm->Linear->GELU->Dropout->Linear."""
        ck = torch.load(os.path.join(HEADS_DIR, name), map_location="cpu", weights_only=False)
        m = nn.Sequential(
            nn.LayerNorm(1024), nn.Linear(1024, ck["hidden"]), nn.GELU(),
            nn.Dropout(ck["dropout"]), nn.Linear(ck["hidden"], ck["state"]["4.weight"].shape[0]))
        m.load_state_dict(ck["state"])
        return m.to(self.dev).eval(), ck

    def load(self):
        _import_backend()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.on_log(f"Device: {self.dev}")

        self.on_log("Loading Head A (zone) / Head B (type) / Head C (explain)…")
        self.head_a, _ = self._mlp_head("head_a_zone.pt")
        self.head_b, cb = self._mlp_head("head_b_13.pt")
        self.classes = cb["classes"]
        self.head_c = Head(torch.zeros(1024), torch.ones(1024))
        self.head_c.load_state_dict(
            torch.load(os.path.join(HEADS_DIR, "head_c_explain.pt"),
                       map_location="cpu", weights_only=False)["state"])
        self.head_c = self.head_c.to(self.dev).eval()
        self.concepts, self.D, self.owner = load_dictionary(self.dev, DICT_DIR)

        self.on_log("Loading V-JEPA 2 (first run downloads the model)…")
        self.encoder = VJepaEncoder(self.dev)
        self.ready = True
        self.on_log("Model ready.")

    # ---- inference ----
    def _segment_embs(self, frames):
        """frames: (N, 224, 224, 3) uint8 RGB -> (SEG, 1024) tensor (one clip per segment)."""
        center, _ = clip_starts(len(frames), np.random.default_rng(0))  # center clips
        clips = []
        for s in center:
            x = frames[s:s + T]
            if len(x) < T:
                x = np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)])
            clips.append(x)
        out = []
        for i in range(0, SEG, CHUNK):
            batch = torch.from_numpy(np.stack(clips[i:i + CHUNK]))  # (b, T, H, W, 3) uint8
            out.append(self.encoder(batch))  # Encoder handles resize/normalize -> (b, 1024)
        return torch.cat(out).to(self.dev).float()  # (SEG, 1024)

    def score(self, frames):
        """Run Head A/B/C. Returns the SAME dict shape as the Phase-1 detector,
        plus zone/severity/explanation so downstream code is unchanged."""
        feats = self._segment_embs(frames)
        with torch.no_grad():
            pa = self.head_a(feats).sigmoid()          # (SEG, 2) = [P(>=yellow), P(red)]
        seg_scores = pa[:, 1]                            # anomaly score = P(red)
        max_score = float(seg_scores.max())
        anomalous = max_score > THRESHOLD

        crime, severity, explanation = "Normal", None, []
        if anomalous:
            peak = int(seg_scores.argmax())
            x = feats[peak:peak + 1]
            with torch.no_grad():
                # Head C: explanation (top-3 dictionary phrases) from the peak segment
                S = match_dictionary(self.head_c(x)[0], self.D, self.owner, len(self.concepts))[0]
                top = S.topk(3).indices.tolist()
                explanation = [dict(phrase=self.concepts[k]["phrase"],
                                    zone=self.concepts[k]["zone"],
                                    severity=self.concepts[k]["severity"],
                                    cos=round(float(S[k]), 3)) for k in top]
                severity = self.concepts[top[0]]["severity"]
                # Head B: crime type from the top-k most anomalous segments
                k = min(5, SEG)
                topk = seg_scores.topk(k).indices
                probs = self.head_b(feats[topk].mean(0, keepdim=True)).softmax(-1)[0]
                crime = self.classes[int(probs.argmax())]

        return {
            "seg_scores": seg_scores.cpu().numpy(),
            "max_score": max_score,
            "crime": crime,
            "anomalous": anomalous,
            "severity": severity,
            "explanation": explanation,
        }


# --------------------------------------------------------------------------- #
#  Video helpers
# --------------------------------------------------------------------------- #
def center_crop_43(img):
    """Center-crop a BGR frame to 4:3 (the Phase-2 training preprocess)."""
    h, w = img.shape[:2]
    if w * 3 > h * 4:
        cw = round(h * 4 / 3)
        return img[:, (w - cw) // 2:(w - cw) // 2 + cw]
    ch = round(w * 3 / 4)
    return img[(h - ch) // 2:(h - ch) // 2 + ch]


def read_frames(path):
    """Phase-2 flowchart preprocess (matches poc/demo2.py): sample at ~7.5 fps,
    center-crop to 4:3, resize to 224x224 RGB. Returns (frames_224, stride, src_fps)."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(src_fps / FPS))
    frames, n = [], 0
    while cap.grab():
        if n % stride == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            # center-crop to 4:3 exactly as the model was trained (demo2.read_frames)
            h, w = img.shape[:2]
            if w * 3 > h * 4:
                cw = round(h * 4 / 3)
                img = img[:, (w - cw) // 2:(w - cw) // 2 + cw]
            else:
                ch = round(w * 3 / 4)
                img = img[(h - ch) // 2:(h - ch) // 2 + ch]
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
                # match training preprocess: center-crop to 4:3, then resize to 224
                crop = center_crop_43(frame)
                small = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
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

        root.title("REVERE — V-JEPA 2 Detector (Phase 2)")
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
        self.log(f"Heads: {HEADS_DIR}", MUTED)
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
        tk.Label(t, text="V-JEPA 2 · zone / type / explanation · Desktop Agent", font=self.f_sub,
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
