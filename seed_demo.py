"""
seed_demo.py — populate the dashboard with sample Boston detections.

Handy for demoing the UI without running the real detector.

    python seed_demo.py                 # 14 detections
    python seed_demo.py 30              # custom count
    python seed_demo.py 20 clip.webm   # attach a real clip to some detections

If a clip path is given (and exists), it's attached to roughly half the
detections. Otherwise detections are metadata-only.
"""
import asyncio, base64, json, os, random, sys
import websockets

WS = os.environ.get("REVERE_WS_BACKEND", "ws://127.0.0.1:8000/ws/ingest")

# Optional sample clip: 2nd CLI arg, or REVERE_SAMPLE_CLIP env var, else none.
CLIP = None
for cand in (sys.argv[2] if len(sys.argv) > 2 else None,
             os.environ.get("REVERE_SAMPLE_CLIP")):
    if cand and os.path.exists(cand):
        CLIP = cand
        break

HOODS = ["Roxbury", "Dorchester", "Mattapan", "Jamaica Plain", "South Boston",
         "Back Bay", "East Boston", "Roslindale", "Hyde Park", "Charlestown", "Allston"]
CRIMES = ["Shooting", "Robbery", "Assault", "Fighting", "Burglary",
          "RoadAccidents", "Arson", "Abuse"]

clip_b64 = None
if CLIP:
    with open(CLIP, "rb") as f:
        clip_b64 = base64.b64encode(f.read()).decode("ascii")
    print(f"Attaching sample clip: {CLIP}")


async def main(n):
    async with websockets.connect(WS, max_size=None) as ws:
        for i in range(n):
            p = {
                "crime": random.choice(CRIMES),
                "location": "Boston",
                "locality": random.choice(HOODS),
                "coordinates": "42.33,-71.08",
                "date": "2026_10_24",
                "time": f"{random.randint(10,22):02d}_{random.randint(0,59):02d}_00",
            }
            if clip_b64 and i % 2 == 0:
                p["video_b64"] = clip_b64
                p["filename"] = os.path.basename(CLIP)
            await ws.send(json.dumps(p))
            print(f"sent {i+1}/{n}: {p['crime']} @ {p['locality']} ->", await ws.recv())


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    asyncio.run(main(count))
