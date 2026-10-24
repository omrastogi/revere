#!/usr/bin/env python
# coding: utf-8
"""
revere_websocket_camera.py
==========================
Live-camera REVERE detector built on the Phase-2 pipeline
(frozen V-JEPA 2 + Head A zone / Head B type / Head C explanation + dictionary),
streaming alerts to the REVERE backend over a WebSocket, with a polished GUI.

Pipeline (mirrors poc/demo2.py, adapted to a live webcam):
    rolling 16-frame clip @ 7.5 fps, a new clip every STRIDE frames
      -> V-JEPA 2 ViT-L (frozen, mean-pooled 1024-d)
      -> Head A: P(>=yellow), P(red)  -> 4-clip smoothing + hysteresis -> GREEN / YELLOW / RED
      -> YELLOW/RED: Head C alignment -> top-3 dictionary phrases (explanation + severity)
      -> RED:        Head B (13 classes) -> crime type if confident
    Each YELLOW/RED event is de-duplicated into an alert (escalates YELLOW -> RED).
    When an alert fires, a short webm clip of the moment is saved and streamed to the backend.

Run (backend on same machine):
    python revere_websocket_camera.py
Run (backend on another PC on the same Wi-Fi):
    python revere_websocket_camera.py --host 192.168.1.42

Dependencies: torch, transformers>=4.52, opencv-python, numpy, websockets  (+ tkinter)
Checkpoints:  poc/heads/{head_a_zone.pt, head_b_13.pt, head_c_explain.pt}
              poc/dictionary/{dictionary.jsonl, dictionary.npy, dictionary_ids.json}
Model:        facebook/vjepa2-vitl-fpc64-256 (downloaded on first run)

Environment overrides:
    REVERE_POC_DIR      path to the poc/ folder        (default: ./poc)
    REVERE_CLIP_DIR     where alert clips are written  (default: ./clips)
    REVERE_WS_BACKEND   backend ingest websocket URL   (default: ws://127.0.0.1:8000/ws/ingest)
"""

import asyncio
import base64
import collections
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


# --------------------------------------------------------------------------- #
#  Paths / config
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POC_DIR = os.environ.get("REVERE_POC_DIR", os.path.join(BASE_DIR, "poc"))
HEADS_DIR = os.path.join(POC_DIR, "heads")
DICT_DIR = os.path.join(POC_DIR, "dictionary")
CLIP_DIR = os.environ.get("REVERE_CLIP_DIR", os.path.join(BASE_DIR, "clips"))
CLIP_PATH = os.path.join(CLIP_DIR, "output_clip.webm")
DEFAULT_PORT = 8000

# Pipeline constants (mirror poc/demo2.py)
FPS = 7.5           # sampling rate fed to the model
SIZE = 224          # V-JEPA input size
T = 16              # frames per clip
STRIDE_FRAMES = 8   # a new clip every 8 sampled frames (~1s at 7.5fps)
SMOOTH = 4          # clips averaged for Head A
T_YELLOW, T_RED, HYST = 0.5, 0.5, 0.1
TYPE_MIN = 0.6      # Head B confidence to name a crime type
DEDUP_S = 10.0      # same-event window
CLIP_SECONDS = 6.0  # length of the alert clip we save/stream

os.makedirs(CLIP_DIR, exist_ok=True)
if POC_DIR not in sys.path:
    sys.path.insert(0, POC_DIR)


def _normalize_backend(value):
    if not value:
        return None
    v = value.strip()
    if v.startswith(("ws://", "wss://")):
        return v
    for p in ("http://", "https://"):
        if v.startswith(p):
            v = v[len(p):]
    v = v.rstrip("/")
    if "/" in v:
        host, path = v.split("/", 1)
        host = host if ":" in host else f"{host}:{DEFAULT_PORT}"
        return f"ws://{host}/{path}"
    if ":" not in v:
        v = f"{v}:{DEFAULT_PORT}"
    return f"ws://{v}/ws/ingest"


def _resolve_backend():
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ("--backend", "--host") and i + 1 < len(argv):
            return _normalize_backend(argv[i + 1])
        if a.startswith("--backend="):
            return _normalize_backend(a.split("=", 1)[1])
        if a.startswith("--host="):
            return _normalize_backend(a.split("=", 1)[1])
    env = os.environ.get("REVERE_WS_BACKEND")
    return _normalize_backend(env) if env else "ws://127.0.0.1:8000/ws/ingest"


WS_BACKEND = _resolve_backend()

# lazy heavy imports
torch = None
Encoder = None
Head = None
load_dictionary = None
match_dictionary = None


def _import_backend():
    global torch, Encoder, Head, load_dictionary, match_dictionary
    if torch is not None:
        return
    import torch as _torch
    # reuse the POC's V-JEPA encoder, alignment head, and dictionary matcher
    from align import Encoder as _Encoder, Head as _Head, \
        load_dictionary as _ld, match_dictionary as _md
    torch = _torch
    Encoder = _Encoder
    Head = _Head
    load_dictionary = _ld
    match_dictionary = _md


# --------------------------------------------------------------------------- #
#  Model: V-JEPA 2 + Head A/B/C + dictionary  (mirrors demo2.Pipeline)
# --------------------------------------------------------------------------- #
def _mlp_head(name, dev):
    ck = torch.load(os.path.join(HEADS_DIR, name), map_location="cpu", weights_only=False)
    import torch.nn as nn
    m = nn.Sequential(
        nn.LayerNorm(1024), nn.Linear(1024, ck["hidden"]), nn.GELU(),
        nn.Dropout(ck["dropout"]), nn.Linear(ck["hidden"], ck["state"]["4.weight"].shape[0]))
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck


class RevereModel:
    def __init__(self, on_log=None):
        self.on_log = on_log or (lambda *a, **k: None)
        self.ready = False
        self.dev = None

    def load(self):
        _import_backend()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.on_log(f"Device: {self.dev}")
        self.on_log("Loading Head A / B / C and dictionary...")
        self.head_a, _ = _mlp_head("head_a_zone.pt", self.dev)
        self.head_b, cb = _mlp_head("head_b_13.pt", self.dev)
        self.classes = cb["classes"]
        self.head_c = Head(torch.zeros(1024), torch.ones(1024))
        self.head_c.load_state_dict(
            torch.load(os.path.join(HEADS_DIR, "head_c_explain.pt"),
                       map_location="cpu", weights_only=False)["state"])
        self.head_c = self.head_c.to(self.dev).eval()
        self.concepts, self.D, self.owner = load_dictionary(self.dev, DICT_DIR)
        self.on_log("Loading V-JEPA 2 (first run downloads the model)...")
        self.encoder = Encoder(self.dev, torch.float16)
        self.ready = True
        self.on_log("Model ready.")

    def embed(self, clip_frames):
        """clip_frames: (T, 224, 224, 3) uint8 RGB -> (1024,) tensor on device."""
        x = torch.from_numpy(clip_frames[None])  # (1, T, H, W, 3)
        return self.encoder(x).to(self.dev)[0]   # Encoder handles resize/normalize


class LivePipeline:
    """Head-A smoothing + hysteresis + Head C/B + alert dedup, fed one clip embedding at a time."""

    def __init__(self, model):
        self.m = model
        self.buf = collections.deque(maxlen=SMOOTH)
        self.red = False
        self.yel = False
        self.alerts = []

    def step(self, x, t):
        with torch.no_grad():
            self.buf.append(self.m.head_a(x[None]).sigmoid()[0].cpu().numpy())
            py, pr = np.mean(self.buf, 0)
            self.red = bool(pr > (T_RED - HYST if self.red else T_RED))
            self.yel = bool(py > (T_YELLOW - HYST if self.yel else T_YELLOW))
            zone = "RED" if self.red else "YELLOW" if self.yel else "GREEN"
            rec = dict(t=round(t, 2), zone=zone, p_yellow=float(py), p_red=float(pr))
            if zone == "GREEN":
                return rec
            # Head C explanation
            S = match_dictionary(self.m.head_c(x[None])[0], self.m.D, self.m.owner,
                                 len(self.m.concepts))[0]
            top = S.topk(3).indices.tolist()
            rec["explanation"] = [dict(phrase=self.m.concepts[k]["phrase"],
                                       zone=self.m.concepts[k]["zone"],
                                       severity=self.m.concepts[k]["severity"],
                                       cos=round(float(S[k]), 3)) for k in top]
            rec["severity"] = self.m.concepts[top[0]]["severity"]
            rec["type"] = None
            if zone == "RED":
                pb = self.m.head_b(x[None]).softmax(1)[0]
                k = int(pb.argmax())
                rec["type_p"] = round(float(pb[k]), 3)
                rec["type"] = self.m.classes[k] if float(pb[k]) >= TYPE_MIN else "Unclassified incident"
            self._alert(rec)
            return rec

    def _alert(self, rec):
        a = self.alerts[-1] if self.alerts else None
        if a and rec["t"] - a["t_last"] <= DEDUP_S:
            a["t_last"] = rec["t"]
            a["updates"] += 1
            a["max_severity"] = max(a["max_severity"], rec["severity"])
            if rec["zone"] == "RED" and a["zone"] == "YELLOW":
                a["zone"] = "RED"
            if rec.get("type") and a.get("type") in (None, "Unclassified incident"):
                a["type"] = rec["type"]
            if rec["p_red"] + rec["p_yellow"] > a["peak_score"]:
                a.update(peak_t=rec["t"], peak_score=rec["p_red"] + rec["p_yellow"],
                         explanation=rec["explanation"], severity=rec["severity"])
            rec["alert_new"] = False
        else:
            a = dict(id=len(self.alerts), t_start=rec["t"], t_last=rec["t"], zone=rec["zone"],
                     type=rec.get("type"), explanation=rec["explanation"], severity=rec["severity"],
                     max_severity=rec["severity"], peak_t=rec["t"],
                     peak_score=rec["p_red"] + rec["p_yellow"], updates=0)
            self.alerts.append(a)
            rec["alert_new"] = True
        rec["alert"] = a["id"]


# --------------------------------------------------------------------------- #
#  Location, clip saving, websocket sender
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


def save_clip(frames_bgr, out_path, fps):
    if not frames_bgr:
        return False
    h, w, _ = frames_bgr[0].shape
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"VP80"), float(fps), (w, h))
    for f in frames_bgr:
        writer.write(f)
    writer.release()
    return True


async def _ws_send(payload, server):
    async with websockets.connect(server, max_size=None) as ws:
        await ws.send(json.dumps(payload))
        return await ws.recv()


def _ws_send_blocking(payload, server, result):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result["reply"] = loop.run_until_complete(_ws_send(payload, server))
    except Exception as e:  # noqa: BLE001
        result["error"] = str(e)
    finally:
        loop.close()


async def _ws_connect_only(server):
    async with websockets.connect(server, max_size=None, open_timeout=8):
        return True


def _ws_ping(server):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_ws_connect_only(server))
        return None
    except Exception as e:  # noqa: BLE001
        return str(e)
    finally:
        loop.close()


def send_detection(crime, zone, severity, explanation, city, locality, coordinates,
                   date, time_str, video_path, confidence=None):
    if websockets is None:
        return None
    payload = {
        "crime": crime, "zone": zone, "severity": severity,
        "explanation": explanation, "location": city, "locality": locality,
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
    t = threading.Thread(target=_ws_send_blocking, args=(payload, WS_BACKEND, result))
    t.start()
    t.join(60)
    return None if result.get("error") else result.get("reply")


# --------------------------------------------------------------------------- #
#  Camera analyzer
# --------------------------------------------------------------------------- #
def center_crop_43(img):
    """Center-crop to 4:3 (flowchart preprocess), then the caller resizes to 224."""
    h, w = img.shape[:2]
    if w * 3 > h * 4:
        cw = round(h * 4 / 3)
        return img[:, (w - cw) // 2:(w - cw) // 2 + cw]
    ch = round(w * 3 / 4)
    return img[(h - ch) // 2:(h - ch) // 2 + ch]


class CameraAnalyzer:
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
        cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        sample_stride = max(1, round(cam_fps / FPS))

        pipe = LivePipeline(self.model)
        sampled = collections.deque(maxlen=T)                    # 224-RGB frames for the model
        recent_bgr = collections.deque(maxlen=int(cam_fps * CLIP_SECONDS))  # full-res for clip saving
        n = 0
        since_clip = 0
        self.on_event("start", "Live camera running - monitoring for incidents...")

        while self.running:
            ok, frame = cap.read()
            if not ok:
                break
            recent_bgr.append(frame.copy())

            if n % sample_stride == 0:
                crop = center_crop_43(frame)
                small = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                sampled.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                since_clip += 1

                # once the rolling buffer is full, score a clip every STRIDE sampled frames
                if len(sampled) == T and since_clip >= STRIDE_FRAMES:
                    since_clip = 0
                    try:
                        emb = self.model.embed(np.stack(sampled))
                        rec = pipe.step(emb, t=n / cam_fps)
                        self._on_rec(rec, list(recent_bgr), cam_fps)
                    except Exception as e:  # noqa: BLE001
                        self.on_event("error", f"inference error: {e}")

            # live preview with a zone-colored border
            self._draw(frame, pipe)
            cv2.imshow("REVERE - Live Camera (press q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            n += 1

        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        self.on_event("done", "Camera stopped.")

    def _on_rec(self, rec, clip_bgr, fps):
        zone = rec["zone"]
        if zone == "GREEN":
            self.on_event("green", f"GREEN - normal (Py {rec['p_yellow']:.2f} Pr {rec['p_red']:.2f})")
            return
        phrase = rec["explanation"][0]["phrase"] if rec.get("explanation") else ""
        if zone == "YELLOW":
            self.on_event("yellow", f"YELLOW - {phrase} (sev {rec['severity']}/5)")
        else:  # RED
            self.on_event("red", f"RED - {rec.get('type','incident')} - {phrase} (sev {rec['severity']}/5)")

        # Only stream a detection when a NEW alert opens (avoid spamming the backend)
        if rec.get("alert_new"):
            self._deliver(rec, clip_bgr, fps)

    def _deliver(self, rec, clip_bgr, fps):
        if not save_clip(clip_bgr, CLIP_PATH, fps):
            self.on_event("error", "could not save alert clip")
            return
        city, locality, coordinates = get_location()
        date, time_str = get_date_and_time()
        archive = CLIP_PATH
        try:
            archive = os.path.join(CLIP_DIR, f"{city}_{date}+{time_str}.webm")
            os.replace(CLIP_PATH, archive)
        except Exception:  # noqa: BLE001
            archive = CLIP_PATH

        crime = rec.get("type") or ("Suspicious activity" if rec["zone"] == "YELLOW" else "Incident")
        explanation = "; ".join(e["phrase"] for e in rec.get("explanation", [])[:2])
        conf = rec["p_red"] if rec["zone"] == "RED" else rec["p_yellow"]
        self.on_event("sending", f"Streaming {rec['zone']} alert '{crime}' @ {city}/{locality}...")
        reply = send_detection(crime, rec["zone"], rec["severity"], explanation,
                               city, locality, coordinates, date, time_str, archive, conf)
        self.on_event("sent" if reply else "error",
                      f"Delivered - {crime} @ {city}" if reply
                      else "Backend unreachable - is the server running?")

    def _draw(self, frame, pipe):
        color = (0, 0, 255) if pipe.red else (0, 200, 255) if pipe.yel else (0, 170, 0)
        label = "RED" if pipe.red else "YELLOW" if pipe.yel else "GREEN"
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, max(3, w // 160))
        cv2.putText(frame, label, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, label, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
#  GUI  (dark REVERE control panel, zone-aware)
# --------------------------------------------------------------------------- #
BG, PANEL, PANEL2, LINE = "#0b1220", "#131c2b", "#1a2538", "#243247"
TEXT, MUTED, TEAL, BLUE = "#e8eef6", "#8aa0b8", "#2dd4bf", "#60a5fa"
OK, WARN, DANGER = "#22c55e", "#f59e0b", "#ef4444"


class RevereApp:
    def __init__(self, root):
        self.root = root
        self.model = RevereModel(on_log=lambda m: self.on_event("log", m))
        self.camera = None

        root.title("REVERE - Live Camera (Phase 2)")
        root.geometry("720x680")
        root.configure(bg=BG)
        root.minsize(660, 600)

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

        self.log("Starting up...", MUTED)
        self.log(f"Backend: {WS_BACKEND}", MUTED)
        self.log(f"Heads: {HEADS_DIR}", MUTED)
        if websockets is None:
            self.log("'websockets' not installed - alerts can't be sent.", DANGER)
        threading.Thread(target=self._load_model, daemon=True).start()

    def _header(self):
        bar = tk.Frame(self.root, bg="#0d1627", height=84); bar.pack(fill="x"); bar.pack_propagate(False)
        wrap = tk.Frame(bar, bg="#0d1627"); wrap.pack(side="left", padx=24, pady=16)
        eye = tk.Canvas(wrap, width=44, height=44, bg="#0d1627", highlightthickness=0)
        eye.create_oval(4, 4, 40, 40, fill=TEAL, outline=""); eye.create_oval(15, 15, 29, 29, fill="#0d1627", outline="")
        eye.pack(side="left", padx=(0, 14))
        t = tk.Frame(wrap, bg="#0d1627"); t.pack(side="left")
        tk.Label(t, text="REVERE", font=self.f_title, fg=TEXT, bg="#0d1627").pack(anchor="w")
        tk.Label(t, text="Live Camera - V-JEPA 2 + zone / type / explanation", font=self.f_sub,
                 fg=MUTED, bg="#0d1627").pack(anchor="w")
        pill = tk.Frame(bar, bg="#0d1627"); pill.pack(side="right", padx=24)
        self.dot = tk.Canvas(pill, width=12, height=12, bg="#0d1627", highlightthickness=0)
        self._dot("#555"); self.dot.pack(side="left", padx=(0, 7), pady=4)
        self.conn = tk.Label(pill, text="loading...", font=self.f_small, fg=MUTED, bg="#0d1627")
        self.conn.pack(side="left")

    def _status(self):
        card = self._card("CURRENT STATUS")
        self.status_var = tk.StringVar(value="Loading model...")
        tk.Label(card, textvariable=self.status_var, font=self.f_lbl, fg=TEXT, bg=PANEL,
                 anchor="w", justify="left", wraplength=640).pack(fill="x", padx=18, pady=(2, 10))
        row = tk.Frame(card, bg=PANEL); row.pack(fill="x", padx=18, pady=(0, 14))
        self.zone_chip = tk.Label(row, text="GREEN", font=self.f_small, fg="#042", bg="#22c55e", padx=14, pady=5)
        self.zone_chip.pack(side="left")

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
        tk.Label(card, text="On another PC, enter the server's IPv4 (e.g. 192.168.1.42) and click Apply.",
                 font=self.f_small, fg=MUTED, bg=PANEL, wraplength=640, justify="left").pack(
                 anchor="w", padx=18, pady=(0, 14))

    def _apply_backend(self):
        global WS_BACKEND
        url = _normalize_backend(self.backend_var.get())
        if not url:
            self.log("Enter an IP or URL first.", WARN); return
        WS_BACKEND = url; self.backend_var.set(url)
        self.log(f"Backend set to {url}", OK)

    def _test_backend(self):
        self._apply_backend()
        self.log(f"Testing {WS_BACKEND} ...", MUTED)
        threading.Thread(target=self._test_worker, args=(WS_BACKEND,), daemon=True).start()

    def _test_worker(self, url):
        if websockets is None:
            self.on_event("error", "'websockets' not installed."); return
        err = _ws_ping(url)
        self.on_event("error" if err else "sent",
                      f"Cannot reach backend: {err}" if err else "Backend reachable")

    def _actions(self):
        card = self._card("CAMERA")
        row = tk.Frame(card, bg=PANEL); row.pack(fill="x", padx=18, pady=(4, 10))
        self.btn_start = tk.Button(row, text="Start Live Camera", command=self.on_start,
                                   font=self.f_lbl, fg="#06101f", bg=TEAL, bd=0, cursor="hand2", state="disabled")
        self.btn_start.pack(side="left", expand=True, fill="x", padx=(0, 8), ipady=10)
        self.btn_stop = tk.Button(row, text="Stop", command=self.on_stop,
                                  font=self.f_lbl, fg=TEXT, bg=PANEL2, bd=0, cursor="hand2", state="disabled")
        self.btn_stop.pack(side="left", ipady=10, padx=(8, 0))
        tk.Label(card, text="GREEN logs only. YELLOW/RED fire an alert with a saved clip, "
                            "an explanation, and (for RED) a crime type - streamed to the dashboard. "
                            "Press q in the preview to stop.",
                 font=self.f_small, fg=MUTED, bg=PANEL, wraplength=640, justify="left").pack(
                 anchor="w", padx=18, pady=(0, 14))

    def _log(self):
        card = self._card("ACTIVITY LOG", grow=True)
        frame = tk.Frame(card, bg=PANEL2, highlightbackground=LINE, highlightthickness=1)
        frame.pack(fill="both", expand=True, padx=18, pady=(2, 16))
        self.logbox = tk.Text(frame, bg=PANEL2, fg=TEXT, font=self.f_mono, bd=0,
                              padx=12, pady=10, wrap="word", state="disabled", height=10)
        self.logbox.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(frame, command=self.logbox.yview); sb.pack(side="right", fill="y")
        self.logbox.config(yscrollcommand=sb.set)

    def _footer(self):
        f = tk.Frame(self.root, bg=BG); f.pack(fill="x", side="bottom")
        tk.Label(f, text="REVERE - alerts are reviewed by officials before going public",
                 font=self.f_small, fg=MUTED, bg=BG).pack(pady=8)

    def _card(self, title, grow=False):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both" if grow else "x", expand=grow, padx=20, pady=(14, 0))
        card = tk.Frame(outer, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        card.pack(fill="both", expand=True)
        tk.Label(card, text=title, font=self.f_small, fg=MUTED, bg=PANEL).pack(anchor="w", padx=18, pady=(12, 2))
        return card

    def _dot(self, color):
        self.dot.delete("all"); self.dot.create_oval(1, 1, 11, 11, fill=color, outline="")

    def _load_model(self):
        try:
            self.model.load()
            self.root.after(0, self._model_ready)
        except Exception as e:  # noqa: BLE001
            self.on_event("error", f"Model load failed: {e}")

    def _model_ready(self):
        self.set_conn("ready", OK)
        self.set_status("Ready - start the camera to begin monitoring")
        self.btn_start.config(state="normal")

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
        self.conn.config(text=text, fg=color); self._dot(color)

    def on_event(self, kind, message):
        self.root.after(0, lambda: self._handle(kind, message))

    def _handle(self, kind, message):
        colors = {"log": MUTED, "start": BLUE, "green": OK, "yellow": WARN, "red": DANGER,
                  "sending": BLUE, "sent": OK, "error": DANGER, "done": MUTED}
        if message:
            self.log(message, colors.get(kind, TEXT))
            if kind not in ("log", "green"):
                self.set_status(message)
        if kind == "start":
            self.set_conn("monitoring", WARN)
            self.btn_start.config(state="disabled"); self.btn_stop.config(state="normal")
        elif kind == "green":
            self.zone_chip.config(text="GREEN", bg="#22c55e", fg="#042")
        elif kind == "yellow":
            self.zone_chip.config(text="YELLOW", bg="#f59e0b", fg="#2a1a00")
        elif kind == "red":
            self.zone_chip.config(text="RED", bg="#ef4444", fg="#2a0000")
        elif kind == "sent":
            self.set_conn("delivered", OK)
        elif kind == "error":
            self.set_conn("error", DANGER)
        elif kind == "done":
            self.set_conn("ready", OK)
            self.btn_start.config(state="normal"); self.btn_stop.config(state="disabled")

    def on_start(self):
        if not self.model.ready:
            return
        self.camera = CameraAnalyzer(self.model, on_event=self.on_event)
        self.camera.start()

    def on_stop(self):
        if self.camera:
            self.camera.stop()


def main():
    root = tk.Tk()
    RevereApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
