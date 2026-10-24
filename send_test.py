"""
Test client — simulates the REVERE detector sending a detection to the backend.

Usage:
    python send_test.py                   # sends a metadata-only detection
    python send_test.py path/to/clip.avi  # sends a real video clip too

This is the exact pattern the real REVERE detector follows:
open a websocket to /ws/ingest and send one JSON message per detection.
"""

import asyncio
import base64
import json
import os
import sys
from datetime import datetime

import websockets

SERVER = "ws://127.0.0.1:8000/ws/ingest"


async def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else None

    now = datetime.now()
    payload = {
        "crime":       "Fighting",
        "location":    "Boston",
        "locality":    "Roxbury",
        "coordinates": "42.3251,-71.0956",
        "date":        now.strftime("%Y_%m_%d"),
        "time":        now.strftime("%H_%M_%S"),
    }

    if video_path and os.path.exists(video_path):
        with open(video_path, "rb") as f:
            payload["video_b64"] = base64.b64encode(f.read()).decode("ascii")
        payload["filename"] = os.path.basename(video_path)
        print(f"Attaching clip: {video_path} "
              f"({len(payload['video_b64'])} b64 chars)")
    else:
        print("Sending metadata-only detection (no video).")

    async with websockets.connect(SERVER, max_size=None) as ws:
        await ws.send(json.dumps(payload))
        reply = await ws.recv()
        print("Server replied:", reply)


if __name__ == "__main__":
    asyncio.run(main())
