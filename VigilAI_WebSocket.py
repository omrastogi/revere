#!/usr/bin/env python
# coding: utf-8
"""
VigilAI_WebSocket.py
====================
Standalone VigilAI crime-detection app that keeps the full detection
functionality of VigilAI_SourceCode_camButton.py and, on every detection,
streams the clip + metadata (crime type, location, locality, coordinates,
date, time) to the backend over a WebSocket.

Backend ingest endpoint (FastAPI, project at C:/demo/vigil_ws):
    ws://127.0.0.1:8000/ws/ingest

Run:
    py VigilAI_WebSocket.py

Dependencies: tensorflow, opencv-python, numpy, websockets  (tkinter ships with Python)
"""

import asyncio
import base64
import csv
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


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
WINDOW_SIZE = 30                 # rolling-average window (frames)
CONFIDENCE_THRESHOLD = 0.6       # min averaged probability to accept a label
MAX_CLIP_FRAMES = 450            # cap on frames collected per incident

MODEL_PATH = "C:/demo/Ryuks-main/Ryuks-main/VigilAI/model_8000.h5"
EYEBASE_DIR = "C:/demo/Ryuks-main/Ryuks-main/VigilAI/healthcare/healthcare/Eyebase"
CLIP_PATH = os.path.join(EYEBASE_DIR, "output_clip.avi")

WS_BACKEND = "ws://127.0.0.1:8000/ws/ingest"

CLASSES_LIST = ["Fighting", "Shooting", "RoadAccidents", "Robbery", "Abuse",
                "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Normal"]
MODEL_OUTPUT_SIZE = len(CLASSES_LIST)

os.makedirs(EYEBASE_DIR, exist_ok=True)

print(f"Classes: {MODEL_OUTPUT_SIZE}")

# --------------------------------------------------------------------------- #
#  Load model
# --------------------------------------------------------------------------- #
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------------------------------- #
#  Location helpers  (now returns city + locality/region + coordinates)
# --------------------------------------------------------------------------- #
def get_location():
    """Return (city, locality, coordinates) via ipinfo.io, with safe fallbacks.

    locality = region/area (ipinfo 'region'); falls back to the clip source.
    """
    try:
        data = json.load(urlopen("http://ipinfo.io/json", timeout=5))
        city = data.get("city", "NA")
        locality = data.get("region", data.get("org", "NA"))
        coordinates = data.get("loc", "NA")
        return city, locality, coordinates
    except Exception:
        return "NA", "NA", "NA"


def get_date_and_time():
    now = datetime.datetime.now()
    return now.strftime("%Y_%m_%d"), now.strftime("%H_%M_%S")


# --------------------------------------------------------------------------- #
#  Clip saving
# --------------------------------------------------------------------------- #
def save_clip(frames):
    """Write collected frames to CLIP_PATH as an .avi (XVID)."""
    if not frames:
        print("[clip] no frames to save")
        return False
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(CLIP_PATH, fourcc, 15, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"[clip] saved {len(frames)} frames -> {CLIP_PATH}")
    return True


# --------------------------------------------------------------------------- #
#  WebSocket sender  (thread-safe: own loop on a background thread)
# --------------------------------------------------------------------------- #
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
    """Stream one detection (clip + metadata) to the backend over WebSocket."""
    if websockets is None:
        print("[send] 'websockets' not installed: py -m pip install websockets")
        return None

    payload = {
        "crime": crime,
        "location": city,
        "locality": locality,
        "coordinates": coordinates,
        "date": date,
        "time": time_str,
    }
    if video_path and os.path.exists(video_path):
        try:
            with open(video_path, "rb") as f:
                payload["video_b64"] = base64.b64encode(f.read()).decode("ascii")
            payload["filename"] = os.path.basename(video_path)
        except Exception as e:  # noqa: BLE001
            print(f"[send] could not read clip: {e}")
    else:
        print(f"[send] clip not found ({video_path}); sending metadata only")

    result = {}
    t = threading.Thread(target=_ws_send_blocking, args=(payload, server := WS_BACKEND, result))
    t.start()
    t.join(30)
    if result.get("error"):
        print(f"[send] FAILED: {result['error']}")
        return None
    print(f"[send] OK  {crime} @ {city}/{locality} -> {result.get('reply')}")
    return result.get("reply")


# --------------------------------------------------------------------------- #
#  Detector
# --------------------------------------------------------------------------- #
class Detector:
    def __init__(self, video_file_path):
        self.video_file_path = video_file_path
        self.video_reader = cv2.VideoCapture(video_file_path)
        self.frames = []
        self.frame_count = 0
        self.crime_type = "NA"
        self.counter = 0
        self.is_processing = False
        self.lock = threading.Lock()
        self.deque = deque(maxlen=WINDOW_SIZE)
        self.processing_thread = None
        self.start_time = time.time()

    def start_processing(self):
        print(f"[detector] starting on source: {self.video_file_path}")
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.start()

    def stop_processing(self):
        self.is_processing = False

    def process_video(self):
        while self.is_processing:
            status, frame = self.video_reader.read()
            if not status:
                break

            resized = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized = resized / 255.0

            # run prediction on a worker thread (keeps capture loop responsive)
            worker = threading.Thread(target=self.predict, args=(normalized, frame))
            worker.start()

            cv2.imshow("CCTV FOOTAGE", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_reader.release()
        cv2.destroyAllWindows()

        # Save the collected incident clip, then stream it to the backend
        if save_clip(self.frames):
            city, locality, coordinates = get_location()
            date, time_str = get_date_and_time()

            # keep a descriptive local archive copy
            archive_path = CLIP_PATH
            try:
                archive_name = f"{city}_{date}+{time_str}.avi"
                archive_path = os.path.join(EYEBASE_DIR, archive_name)
                os.replace(CLIP_PATH, archive_path)
            except Exception as e:  # noqa: BLE001
                print(f"[detector] archive rename failed: {e}")
                archive_path = CLIP_PATH

            send_detection(
                crime=self.crime_type,
                city=city,
                locality=locality,
                coordinates=coordinates,
                date=date,
                time_str=time_str,
                video_path=archive_path,
            )
        else:
            print("[detector] nothing detected; no clip sent")

        elapsed = time.time() - self.start_time
        print(f"[detector] done in {elapsed:.1f}s")

    def predict(self, normalized_frame, original_frame):
        with self.lock:
            probs = model.predict(np.expand_dims(normalized_frame, axis=0), verbose=0)[0]
            self.deque.append(probs)

            if len(self.deque) == WINDOW_SIZE:
                averaged = np.array(self.deque).mean(axis=0)
                max_prob = float(np.max(averaged))
                label = int(np.argmax(averaged))

                if max_prob > CONFIDENCE_THRESHOLD and label < 10:
                    self.frame_count += 1
                    if self.frame_count < MAX_CLIP_FRAMES:
                        if self.counter == 0:
                            self.crime_type = CLASSES_LIST[label]
                            self.counter = 1
                            print(f"[predict] detected {self.crime_type} ({max_prob:.2f})")
                        self.frames.append(original_frame)
                    else:
                        if len(self.frames) > 100:
                            self.frame_count = 0
                else:
                    self.frame_count = 0


# --------------------------------------------------------------------------- #
#  Tkinter GUI
# --------------------------------------------------------------------------- #
import tkinter as tk
from tkinter.filedialog import askopenfilename


def start_video():
    file_path = askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm")],
    )
    if file_path:
        Detector(file_path).start_processing()


def start_camera():
    Detector(0).start_processing()


def build_gui():
    root = tk.Tk()
    root.title("VigilAI - WebSocket Detector")
    root.geometry("600x600")

    tk.Label(root, text="VigilAI", font="comicsans 30 bold").pack(pady=30)

    tk.Button(root, text="Analyze Video File", font="comicsans 20 bold",
              bg="yellow", padx=30, pady=20, command=start_video).place(x=150, y=120)

    tk.Button(root, text="Live Camera", font="comicsans 20 bold",
              bg="lightgreen", padx=30, pady=20, command=start_camera).place(x=150, y=380)

    root.mainloop()


if __name__ == "__main__":
    build_gui()
