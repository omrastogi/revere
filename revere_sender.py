"""
revere_sender.py — drop-in helper for the REVERE detector.

Import this in your detection script and call send_detection() whenever a
crime clip is ready. It handles base64 encoding and the websocket round-trip.

It is safe to call from any thread: the websocket work runs in a fresh
background thread with its own asyncio event loop, so it won't clash with
the detector's own threads or Tkinter's main loop.

Example:

    from revere_sender import send_detection

    send_detection(
        crime="Fighting",
        location="Boston",
        coordinates="42.32,-71.09",
        date="2026_10_24",
        time="19_34_41",
        video_path="./clips/output_clip.avi",
        server="ws://127.0.0.1:8000/ws/ingest",
    )
"""

import asyncio
import base64
import json
import os
import threading

try:
    import websockets
except ImportError:
    websockets = None


async def _send_async(payload, server):
    async with websockets.connect(server, max_size=None) as ws:
        await ws.send(json.dumps(payload))
        return await ws.recv()


def _run_in_new_loop(payload, server, result):
    """Run the async send in a brand-new event loop on this thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result["reply"] = loop.run_until_complete(_send_async(payload, server))
    except Exception as e:  # noqa: BLE001
        result["error"] = str(e)
    finally:
        loop.close()


def send_detection(crime, location, coordinates, date, time,
                   locality=None, video_path=None,
                   server="ws://127.0.0.1:8000/ws/ingest",
                   blocking=True, timeout=30):
    """Send one detection (metadata + optional clip) to the backend.

    Runs on a background thread so it is safe to call from a worker thread.
    If blocking=True, waits up to `timeout` seconds for the reply.

    Returns the server's JSON reply string, or None on failure.
    """
    if websockets is None:
        print("[revere_sender] 'websockets' package not installed; "
              "run: pip install websockets")
        return None

    payload = {
        "crime": crime,
        "location": location,
        "coordinates": coordinates,
        "date": date,
        "time": time,
    }
    if locality is not None:
        payload["locality"] = locality

    if video_path and os.path.exists(video_path):
        try:
            with open(video_path, "rb") as f:
                payload["video_b64"] = base64.b64encode(f.read()).decode("ascii")
            payload["filename"] = os.path.basename(video_path)
        except Exception as e:  # noqa: BLE001
            print(f"[revere_sender] could not read video: {e}")
    else:
        if video_path:
            print(f"[revere_sender] video not found: {video_path} (sending metadata only)")

    result = {}
    t = threading.Thread(target=_run_in_new_loop, args=(payload, server, result))
    t.start()

    if blocking:
        t.join(timeout)
        if result.get("error"):
            print(f"[revere_sender] send failed: {result['error']}")
            return None
        reply = result.get("reply")
        print(f"[revere_sender] sent '{crime}' @ {location} -> {reply}")
        return reply
    return None
