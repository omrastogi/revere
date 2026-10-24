#!/usr/bin/env python
# coding: utf-8
"""
revere_detector.py
==================
REVERE crime-detection desktop app.

Runs a Keras model over a video file or live camera, detects incidents, saves
a clip, and streams the clip + metadata (crime, location, locality, coordinates,
date, time) to the REVERE backend over a WebSocket.

Backend ingest endpoint (FastAPI):
    ws://127.0.0.1:8000/ws/ingest

Run:
    python revere_detector.py       (or: py revere_detector.py)

Dependencies: tensorflow, opencv-python, numpy, websockets  (tkinter ships with Python)

Environment overrides:
    REVERE_MODEL_PATH   path to model_8000.h5          (default: ./model_8000.h5)
    REVERE_CLIP_DIR     where clips are written         (default: ./clips)
    REVERE_WS_BACKEND   backend ingest websocket URL    (default: ws://127.0.0.1:8000/ws/ingest)
"""

import asyncio
import base64
import datetime
import json
import os
import threading
import time
from collections import deque
from urllib.request import urlopen

import cv2
import numpy as np
import tensorflow as tf

try:
    import websockets
except ImportError:
    websockets = None

import tkinter as tk
from tkinter import font as tkfont
from tkinter.filedialog import askopenfilename


# --------------------------------------------------------------------------- #
#  Configuration  (self-contained; resolves relative to this file)
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
WINDOW_SIZE = 30                 # rolling-average window (frames)
CONFIDENCE_THRESHOLD = 0.6       # min averaged probability to accept a label

# Clip is a CONTIGUOUS segment: from (first detection - PRE_PAD) to
# (last detection + POST_PAD), played back at the source's real frame rate.
PRE_PAD_SECONDS = 3.0            # seconds of context kept before first detection
POST_PAD_SECONDS = 3.0           # seconds of context kept after last detection
MAX_BUFFER_SECONDS = 120         # safety cap on how much footage we hold in RAM
DEFAULT_FPS = 15                 # fallback if the source doesn't report its fps


def _find_model():
    for c in [
        os.environ.get("REVERE_MODEL_PATH"),
        os.path.join(BASE_DIR, "model_8000.h5"),
        "C:/demo/model_8000.h5",
        "C:/demo/Ryuks-main/Ryuks-main/VigilAI/model_8000.h5",
    ]:
        if c and os.path.exists(c):
            return c
    return os.path.join(BASE_DIR, "model_8000.h5")


MODEL_PATH = _find_model()
CLIP_DIR = os.environ.get("REVERE_CLIP_DIR", os.path.join(BASE_DIR, "clips"))
CLIP_PATH = os.path.join(CLIP_DIR, "output_clip.webm")
WS_BACKEND = os.environ.get("REVERE_WS_BACKEND", "ws://127.0.0.1:8000/ws/ingest")

CLASSES_LIST = ["Fighting", "Shooting", "RoadAccidents", "Robbery", "Abuse",
                "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Normal"]

os.makedirs(CLIP_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Location + time helpers
# --------------------------------------------------------------------------- #
def get_location():
    """Return (city, locality, coordinates) via ipinfo.io, with safe fallbacks."""
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
#  Clip saving + websocket sending
# --------------------------------------------------------------------------- #
def save_clip(frames, fps=DEFAULT_FPS):
    """Write a list of frames to CLIP_PATH as webm/VP8 at the given fps."""
    if not frames:
        return False
    height, width, _ = frames[0].shape
    # VP80 (VP8) in a .webm container — this is what browsers can play in <video>.
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    writer = cv2.VideoWriter(CLIP_PATH, fourcc, float(fps), (width, height))
    for frame in frames:
        writer.write(frame)
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


def send_detection(crime, city, locality, coordinates, date, time_str, video_path):
    """Stream one detection (clip + metadata) to the backend. Returns reply str or None."""
    if websockets is None:
        return None
    payload = {
        "crime": crime, "location": city, "locality": locality,
        "coordinates": coordinates, "date": date, "time": time_str,
    }
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
    t.join(30)
    if result.get("error"):
        return None
    return result.get("reply")


# --------------------------------------------------------------------------- #
#  Detector  (reports progress back to the GUI through callbacks)
# --------------------------------------------------------------------------- #
class Detector:
    def __init__(self, video_file_path, on_event=None):
        self.video_file_path = video_file_path
        self.video_reader = cv2.VideoCapture(video_file_path)

        # read the source frame rate so the clip plays at real speed
        src_fps = self.video_reader.get(cv2.CAP_PROP_FPS)
        self.fps = src_fps if src_fps and src_fps > 1 else DEFAULT_FPS
        self.max_buffer = int(self.fps * MAX_BUFFER_SECONDS)

        # ring buffer of ALL recent frames (so we can slice a contiguous segment)
        self.buffer = deque(maxlen=self.max_buffer)   # holds (index, frame)
        self.first_detect_idx = None                  # index of first crime frame
        self.last_detect_idx = None                   # index of last crime frame
        self.crime_type = "NA"
        self.detect_count = 0

        self.is_processing = False
        self.lock = threading.Lock()
        self.deque = deque(maxlen=WINDOW_SIZE)         # rolling prob window
        self.on_event = on_event or (lambda *a, **k: None)
        self.start_time = time.time()
        self.frames_seen = 0

    def start_processing(self):
        self.is_processing = True
        threading.Thread(target=self.process_video, daemon=True).start()

    def stop_processing(self):
        self.is_processing = False

    def process_video(self):
        src = "camera" if self.video_file_path == 0 else os.path.basename(str(self.video_file_path))
        self.on_event("start", f"Analysing {src}… (source {self.fps:.0f} fps)")

        idx = 0
        while self.is_processing:
            status, frame = self.video_reader.read()
            if not status:
                break
            self.frames_seen += 1

            # keep every frame in the ring buffer, tagged with its index
            self.buffer.append((idx, frame))

            resized = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized = resized / 255.0
            threading.Thread(target=self.predict, args=(normalized, idx), daemon=True).start()

            cv2.imshow("REVERE — Live Feed (press q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            idx += 1

        self.is_processing = False
        self.video_reader.release()
        cv2.destroyAllWindows()
        self.on_event("stopped", "Feed ended. Finalising…")

        segment = self._build_segment()
        if segment:
            self.on_event("clip",
                          f"Clip: {len(segment)} frames (~{len(segment)/self.fps:.1f}s) "
                          f"with {PRE_PAD_SECONDS:.0f}s/{POST_PAD_SECONDS:.0f}s padding")
            if save_clip(segment, fps=self.fps):
                city, locality, coordinates = get_location()
                date, time_str = get_date_and_time()
                archive_path = CLIP_PATH
                try:
                    archive_path = os.path.join(CLIP_DIR, f"{city}_{date}+{time_str}.webm")
                    os.replace(CLIP_PATH, archive_path)
                except Exception:  # noqa: BLE001
                    archive_path = CLIP_PATH

                self.on_event("sending", f"Streaming '{self.crime_type}' @ {city}/{locality}…")
                reply = send_detection(self.crime_type, city, locality,
                                       coordinates, date, time_str, archive_path)
                if reply:
                    self.on_event("sent", f"Delivered to backend · {self.crime_type} @ {city}")
                else:
                    self.on_event("error", "Backend unreachable — is the server running?")
            else:
                self.on_event("error", "Failed to write clip.")
        else:
            self.on_event("none", "No incident detected; nothing sent.")

        self.on_event("done", f"Done in {time.time() - self.start_time:.1f}s")

    def _build_segment(self):
        """Slice a contiguous segment: [first_detect - pre_pad, last_detect + post_pad]."""
        if self.first_detect_idx is None or not self.buffer:
            return []

        pre = int(self.fps * PRE_PAD_SECONDS)
        post = int(self.fps * POST_PAD_SECONDS)
        start_idx = self.first_detect_idx - pre
        end_idx = self.last_detect_idx + post

        # buffer holds (index, frame); pull every frame whose index is in range.
        # Frames older than the ring buffer may have been evicted — that's the
        # MAX_BUFFER_SECONDS safety cap; we just include whatever is still held.
        segment = [f for (i, f) in self.buffer if start_idx <= i <= end_idx]
        return segment

    def predict(self, normalized_frame, idx):
        with self.lock:
            probs = model.predict(np.expand_dims(normalized_frame, axis=0), verbose=0)[0]
            self.deque.append(probs)
            if len(self.deque) == WINDOW_SIZE:
                averaged = np.array(self.deque).mean(axis=0)
                max_prob = float(np.max(averaged))
                label = int(np.argmax(averaged))
                if max_prob > CONFIDENCE_THRESHOLD and label < 10:
                    self.detect_count += 1
                    # worker threads may finish out of order -> use min/max
                    if self.first_detect_idx is None or idx < self.first_detect_idx:
                        self.first_detect_idx = idx
                        self.crime_type = CLASSES_LIST[label]
                        self.on_event("detect",
                                      f"⚠ {self.crime_type} detected ({max_prob*100:.0f}% conf.)")
                    if self.last_detect_idx is None or idx > self.last_detect_idx:
                        self.last_detect_idx = idx


# --------------------------------------------------------------------------- #
#  GUI  — dark REVERE-branded control panel
# --------------------------------------------------------------------------- #
# palette (matches the web UI)
BG      = "#0b1220"
PANEL   = "#131c2b"
PANEL2  = "#1a2538"
LINE    = "#243247"
TEXT    = "#e8eef6"
MUTED   = "#8aa0b8"
TEAL    = "#2dd4bf"
BLUE    = "#60a5fa"
OK      = "#22c55e"
WARN    = "#f59e0b"
DANGER  = "#ef4444"


class RevereApp:
    def __init__(self, root):
        self.root = root
        self.active = None
        root.title("REVERE — Detector")
        root.geometry("680x640")
        root.configure(bg=BG)
        root.minsize(620, 560)

        self.f_title = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        self.f_sub   = tkfont.Font(family="Segoe UI", size=10)
        self.f_lbl   = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.f_small = tkfont.Font(family="Segoe UI", size=9)
        self.f_mono  = tkfont.Font(family="Consolas", size=9)

        self._build_header()
        self._build_status()
        self._build_actions()
        self._build_log()
        self._build_footer()

        self.log("Ready.", MUTED)
        self.log(f"Model: {os.path.basename(MODEL_PATH)}", MUTED)
        self.log(f"Backend: {WS_BACKEND}", MUTED)
        if websockets is None:
            self.log("'websockets' not installed — detections can't be sent.", DANGER)

    # ---- sections ----
    def _build_header(self):
        bar = tk.Frame(self.root, bg="#0d1627", height=84)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        wrap = tk.Frame(bar, bg="#0d1627")
        wrap.pack(side="left", padx=24, pady=16)

        eye = tk.Canvas(wrap, width=44, height=44, bg="#0d1627", highlightthickness=0)
        eye.create_oval(4, 4, 40, 40, fill=TEAL, outline="")
        eye.create_oval(15, 15, 29, 29, fill="#0d1627", outline="")
        eye.pack(side="left", padx=(0, 14))

        t = tk.Frame(wrap, bg="#0d1627")
        t.pack(side="left")
        tk.Label(t, text="REVERE", font=self.f_title, fg=TEXT, bg="#0d1627").pack(anchor="w")
        tk.Label(t, text="Crime Detection · Desktop Agent", font=self.f_sub,
                 fg=MUTED, bg="#0d1627").pack(anchor="w")

        # live connection pill
        self.pill = tk.Frame(bar, bg="#0d1627")
        self.pill.pack(side="right", padx=24)
        self.dot = tk.Canvas(self.pill, width=12, height=12, bg="#0d1627", highlightthickness=0)
        self._dot_color("#555")
        self.dot.pack(side="left", padx=(0, 7), pady=4)
        self.conn_lbl = tk.Label(self.pill, text="idle", font=self.f_small, fg=MUTED, bg="#0d1627")
        self.conn_lbl.pack(side="left")

    def _build_status(self):
        card = self._card("CURRENT STATUS")
        self.status_var = tk.StringVar(value="Idle — choose a source to begin")
        tk.Label(card, textvariable=self.status_var, font=self.f_lbl, fg=TEXT, bg=PANEL,
                 anchor="w", justify="left", wraplength=600).pack(fill="x", padx=18, pady=(2, 10))

        row = tk.Frame(card, bg=PANEL)
        row.pack(fill="x", padx=18, pady=(0, 14))
        self.crime_chip = self._chip(row, "No incident", PANEL2, MUTED)
        self.crime_chip.pack(side="left")
        self.frames_lbl = tk.Label(row, text="", font=self.f_small, fg=MUTED, bg=PANEL)
        self.frames_lbl.pack(side="right")

    def _build_actions(self):
        card = self._card("DETECTION SOURCE")
        row = tk.Frame(card, bg=PANEL)
        row.pack(fill="x", padx=18, pady=(4, 16))

        self.btn_file = self._button(row, "📁  Analyze Video File", BLUE, self.on_file)
        self.btn_file.pack(side="left", expand=True, fill="x", padx=(0, 8), ipady=10)
        self.btn_cam = self._button(row, "📷  Live Camera", TEAL, self.on_camera)
        self.btn_cam.pack(side="left", expand=True, fill="x", padx=(8, 0), ipady=10)

        tk.Label(card, text="Tip: while a feed is running, press  q  in the video window to stop.",
                 font=self.f_small, fg=MUTED, bg=PANEL).pack(anchor="w", padx=18, pady=(0, 14))

    def _build_log(self):
        card = self._card("ACTIVITY LOG", grow=True)
        frame = tk.Frame(card, bg=PANEL2, highlightbackground=LINE, highlightthickness=1)
        frame.pack(fill="both", expand=True, padx=18, pady=(2, 16))
        self.logbox = tk.Text(frame, bg=PANEL2, fg=TEXT, font=self.f_mono, bd=0,
                              padx=12, pady=10, wrap="word", state="disabled", height=8)
        self.logbox.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(frame, command=self.logbox.yview)
        sb.pack(side="right", fill="y")
        self.logbox.config(yscrollcommand=sb.set)

    def _build_footer(self):
        f = tk.Frame(self.root, bg=BG)
        f.pack(fill="x", side="bottom")
        tk.Label(f, text="REVERE · detections are reviewed by officials before going public",
                 font=self.f_small, fg=MUTED, bg=BG).pack(pady=8)

    # ---- widget helpers ----
    def _card(self, title, grow=False):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both" if grow else "x", expand=grow, padx=20, pady=(14, 0))
        card = tk.Frame(outer, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        card.pack(fill="both", expand=True)
        tk.Label(card, text=title, font=self.f_small, fg=MUTED, bg=PANEL).pack(anchor="w", padx=18, pady=(12, 2))
        return card

    def _chip(self, parent, text, bg, fg):
        c = tk.Label(parent, text=text, font=self.f_small, fg=fg, bg=bg, padx=12, pady=5)
        return c

    def _button(self, parent, text, color, cmd):
        b = tk.Button(parent, text=text, command=cmd, font=self.f_lbl, fg="#06101f",
                      bg=color, activebackground=color, activeforeground="#06101f",
                      bd=0, relief="flat", cursor="hand2")
        return b

    def _dot_color(self, color):
        self.dot.delete("all")
        self.dot.create_oval(1, 1, 11, 11, fill=color, outline="")

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
        self.conn_lbl.config(text=text, fg=color)
        self._dot_color(color)

    # ---- detector event bridge (runs on worker thread -> marshal to UI) ----
    def on_event(self, kind, message):
        self.root.after(0, lambda: self._handle_event(kind, message))

    def _handle_event(self, kind, message):
        colors = {"start": BLUE, "detect": WARN, "clip": TEAL, "sending": BLUE,
                  "sent": OK, "error": DANGER, "none": MUTED, "done": OK,
                  "stopped": MUTED}
        self.log(message, colors.get(kind, TEXT))
        self.set_status(message)

        if kind == "start":
            self.set_conn("running", WARN)
            self._busy(True)
        elif kind == "detect":
            crime = message.split()[1]
            self.crime_chip.config(text=f"⚠ {crime}", bg="#2a1313", fg="#fca5a5")
        elif kind == "sent":
            self.set_conn("delivered", OK)
        elif kind == "error":
            self.set_conn("error", DANGER)
        elif kind == "done":
            self.set_conn("idle", MUTED)
            self._busy(False)
            if self.active:
                self.frames_lbl.config(text=f"{self.active.frames_seen} frames processed")

    def _busy(self, busy):
        state = "disabled" if busy else "normal"
        self.btn_file.config(state=state)
        self.btn_cam.config(state=state)

    # ---- actions ----
    def on_file(self):
        path = askopenfilename(title="Select a video file",
                               filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm")])
        if path:
            self._launch(path)

    def on_camera(self):
        self._launch(0)

    def _launch(self, source):
        self.crime_chip.config(text="No incident", bg=PANEL2, fg=MUTED)
        self.frames_lbl.config(text="")
        self.active = Detector(source, on_event=self.on_event)
        self.active.start_processing()


def main():
    global model
    # Build the window first so the user sees loading feedback
    root = tk.Tk()
    app = RevereApp(root)
    app.log("Loading model… (this can take a few seconds)", WARN)
    root.update()

    def load():
        global model
        try:
            if not os.path.exists(MODEL_PATH):
                app.on_event("error", f"Model not found: {MODEL_PATH}")
                return
            model = tf.keras.models.load_model(MODEL_PATH)
            app.root.after(0, lambda: (app.log("Model loaded. Ready to detect.", OK),
                                       app.set_status("Ready — choose a source to begin"),
                                       app.set_conn("ready", OK)))
        except Exception as e:  # noqa: BLE001
            app.on_event("error", f"Model load failed: {e}")

    threading.Thread(target=load, daemon=True).start()
    root.mainloop()


model = None

if __name__ == "__main__":
    main()
