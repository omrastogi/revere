"""
seed_demo.py — populate the dashboard with sample Boston detections.

Handy for demoing the UI without running the real detector.

    py seed_demo.py           # 14 detections
    py seed_demo.py 30        # custom count
"""
import asyncio, base64, json, os, random, sys
import websockets

WS = "ws://127.0.0.1:8000/ws/ingest"
CLIP = "C:/demo/Ryuks-main/Ryuks-main/VigilAI/healthcare/healthcare/Eyebase/Lucknow_2023_06_18+18_33_21_233790.webm"

HOODS = ["Roxbury", "Dorchester", "Mattapan", "Jamaica Plain", "South Boston",
         "Back Bay", "East Boston", "Roslindale", "Hyde Park", "Charlestown", "Allston"]
CRIMES = ["Shooting", "Robbery", "Assault", "Fighting", "Burglary",
          "RoadAccidents", "Arson", "Abuse"]

clip_b64 = None
if os.path.exists(CLIP):
    with open(CLIP, "rb") as f:
        clip_b64 = base64.b64encode(f.read()).decode("ascii")


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
                p["filename"] = "clip.webm"
            await ws.send(json.dumps(p))
            print(f"sent {i+1}/{n}: {p['crime']} @ {p['locality']} ->", await ws.recv())


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    asyncio.run(main(count))
