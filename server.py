"""
REVERE Backend
==============
A FastAPI server powering a two-audience crime-intelligence platform:

  • Community portal  (public, no login)  -> /
      Residents see crime rates across Boston neighborhoods (Roxbury,
      Dorchester, ...), trends, and recent *verified* public incidents.

  • Officials portal  (login required)    -> /login  then  /officials
      Law-enforcement users see the live detection feed with video clips,
      full metadata, and can verify / dismiss incidents.

Realtime:
  • Detectors (VigilAI) push detections over  ws://.../ws/ingest
  • Dashboards receive live updates over       ws://.../ws/dashboard
      (channel is chosen per-connection: "public" or "officials")

Run:
    py -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import json
import os
import secrets
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set

from fastapi import Cookie, FastAPI, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------------------------------- #
#  Setup
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(CLIPS_DIR, exist_ok=True)

app = FastAPI(title="REVERE")
app.mount("/clips", StaticFiles(directory=CLIPS_DIR), name="clips")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --------------------------------------------------------------------------- #
#  Boston neighborhoods (localities).  Detections are mapped to one of these.
#  Each has a rough center so the community map can plot them.
# --------------------------------------------------------------------------- #
BOSTON_NEIGHBORHOODS = {
    "Roxbury":        {"lat": 42.3251, "lng": -71.0956},
    "Dorchester":     {"lat": 42.3016, "lng": -71.0661},
    "Mattapan":       {"lat": 42.2676, "lng": -71.0920},
    "Jamaica Plain":  {"lat": 42.3098, "lng": -71.1203},
    "Roslindale":     {"lat": 42.2843, "lng": -71.1300},
    "Hyde Park":      {"lat": 42.2565, "lng": -71.1242},
    "South Boston":   {"lat": 42.3340, "lng": -71.0493},
    "Back Bay":       {"lat": 42.3503, "lng": -71.0810},
    "Charlestown":    {"lat": 42.3782, "lng": -71.0602},
    "East Boston":    {"lat": 42.3702, "lng": -71.0389},
    "Allston":        {"lat": 42.3539, "lng": -71.1337},
    "Fenway":         {"lat": 42.3429, "lng": -71.0975},
    "Downtown":       {"lat": 42.3557, "lng": -71.0603},
    "Other":          {"lat": 42.3601, "lng": -71.0589},
}

# Severity weighting used to compute a neighborhood "crime score"
CRIME_SEVERITY = {
    "Shooting": 10, "Explosion": 10, "Arson": 9, "Robbery": 8, "Assault": 7,
    "Fighting": 6, "Abuse": 7, "Burglary": 6, "RoadAccidents": 5, "Arrest": 2,
    "Normal": 0, "Unknown": 3,
}

# --------------------------------------------------------------------------- #
#  Auth (demo-grade, in-memory).  Replace with a real user store for prod.
# --------------------------------------------------------------------------- #
OFFICIALS = {
    # username: password
    "officer":  "boston123",
    "admin":    "vigilai",
}
SESSIONS: Dict[str, str] = {}   # token -> username


def current_user(session: Optional[str]) -> Optional[str]:
    return SESSIONS.get(session) if session else None


# --------------------------------------------------------------------------- #
#  In-memory detection store
# --------------------------------------------------------------------------- #
# Each detection: {id, crime, location, locality, neighborhood, coordinates,
#                  date, time, video_url, received_at, status, severity}
DETECTIONS: List[Dict] = []

# Connected dashboards, split by audience channel
PUBLIC_CLIENTS: Set[WebSocket] = set()
OFFICIAL_CLIENTS: Set[WebSocket] = set()
_lock = asyncio.Lock()


def map_neighborhood(payload: Dict) -> str:
    """Map an incoming detection to a Boston neighborhood.

    Preference: explicit 'neighborhood' field, then 'locality', then
    nearest-center by coordinates, else 'Other'.
    """
    for key in ("neighborhood", "locality"):
        val = (payload.get(key) or "").strip()
        for name in BOSTON_NEIGHBORHOODS:
            if name.lower() == val.lower():
                return name

    # try nearest center from coordinates "lat,lng"
    coords = payload.get("coordinates", "")
    try:
        lat, lng = (float(x) for x in coords.split(","))
        best, best_d = "Other", 1e9
        for name, c in BOSTON_NEIGHBORHOODS.items():
            if name == "Other":
                continue
            d = (c["lat"] - lat) ** 2 + (c["lng"] - lng) ** 2
            if d < best_d:
                best, best_d = name, d
        # only accept if reasonably close (~within ~0.1 deg)
        return best if best_d < 0.02 else "Other"
    except Exception:
        return "Other"


def public_view(d: Dict) -> Dict:
    """Trimmed detection for the community page (no raw video, no precise coords)."""
    return {
        "id": d["id"],
        "crime": d["crime"],
        "neighborhood": d["neighborhood"],
        "date": d["date"],
        "time": d["time"],
        "severity": d["severity"],
        "status": d["status"],
    }


# --------------------------------------------------------------------------- #
#  Broadcast helpers
# --------------------------------------------------------------------------- #
async def _send_many(clients: Set[WebSocket], message: Dict):
    dead = []
    async with _lock:
        snapshot = list(clients)
    for ws in snapshot:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    if dead:
        async with _lock:
            for ws in dead:
                clients.discard(ws)


async def broadcast_detection(d: Dict):
    # officials get the full record; public gets only verified, trimmed records
    await _send_many(OFFICIAL_CLIENTS, {"type": "detection", "data": d})
    if d["status"] == "verified":
        await _send_many(PUBLIC_CLIENTS, {"type": "detection", "data": public_view(d)})
    # stats change on every new detection / status change
    await _send_many(PUBLIC_CLIENTS, {"type": "stats", "data": compute_stats()})
    await _send_many(OFFICIAL_CLIENTS, {"type": "stats", "data": compute_stats()})


# --------------------------------------------------------------------------- #
#  Stats / aggregation
# --------------------------------------------------------------------------- #
def compute_stats() -> Dict:
    by_hood_count = Counter()
    by_hood_score = defaultdict(int)
    by_crime = Counter()
    verified = 0

    for d in DETECTIONS:
        by_hood_count[d["neighborhood"]] += 1
        by_hood_score[d["neighborhood"]] += d["severity"]
        by_crime[d["crime"]] += 1
        if d["status"] == "verified":
            verified += 1

    neighborhoods = []
    max_score = max(by_hood_score.values(), default=0) or 1
    for name, c in BOSTON_NEIGHBORHOODS.items():
        if name == "Other" and by_hood_count[name] == 0:
            continue
        score = by_hood_score.get(name, 0)
        neighborhoods.append({
            "name": name,
            "lat": c["lat"],
            "lng": c["lng"],
            "count": by_hood_count.get(name, 0),
            "score": score,
            "intensity": round(score / max_score, 3),  # 0..1 for heat coloring
        })
    neighborhoods.sort(key=lambda x: x["score"], reverse=True)

    return {
        "total": len(DETECTIONS),
        "verified": verified,
        "pending": len(DETECTIONS) - verified,
        "neighborhoods": neighborhoods,
        "by_crime": dict(by_crime.most_common()),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# --------------------------------------------------------------------------- #
#  Ingest endpoint  (detector -> server)
# --------------------------------------------------------------------------- #
@app.websocket("/ws/ingest")
async def ws_ingest(websocket: WebSocket):
    await websocket.accept()
    print("[ingest] detector connected")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"ok": False, "error": "invalid JSON"})
                continue
            detection = await _store_detection(payload)
            await websocket.send_json({"ok": True, "id": detection["id"]})
            await broadcast_detection(detection)
            print(f"[ingest] {detection['id']} {detection['crime']} @ {detection['neighborhood']}")
    except WebSocketDisconnect:
        print("[ingest] detector disconnected")


async def _store_detection(payload: Dict) -> Dict:
    det_id = uuid.uuid4().hex[:12]

    video_url = None
    video_b64 = payload.get("video_b64")
    if video_b64:
        ext = os.path.splitext(payload.get("filename", "clip.avi"))[1] or ".avi"
        out_name = f"{det_id}{ext}"
        try:
            with open(os.path.join(CLIPS_DIR, out_name), "wb") as f:
                f.write(base64.b64decode(video_b64))
            video_url = f"/clips/{out_name}"
        except Exception as e:
            print(f"[ingest] clip save failed: {e}")

    crime = payload.get("crime", "Unknown")
    detection = {
        "id": det_id,
        "crime": crime,
        "location": payload.get("location", "NA"),
        "locality": payload.get("locality", "NA"),
        "neighborhood": map_neighborhood(payload),
        "coordinates": payload.get("coordinates", "NA"),
        "date": payload.get("date", ""),
        "time": payload.get("time", ""),
        "video_url": video_url,
        "received_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "pending",   # pending -> verified / dismissed
        "severity": CRIME_SEVERITY.get(crime, 3),
    }
    DETECTIONS.append(detection)
    return detection


# --------------------------------------------------------------------------- #
#  Dashboard endpoint  (browsers <- server).  ?channel=public|officials
# --------------------------------------------------------------------------- #
@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    channel = websocket.query_params.get("channel", "public")
    await websocket.accept()

    pool = OFFICIAL_CLIENTS if channel == "officials" else PUBLIC_CLIENTS
    async with _lock:
        pool.add(websocket)

    # initial payload
    if channel == "officials":
        await websocket.send_json({"type": "history", "data": DETECTIONS})
    else:
        public = [public_view(d) for d in DETECTIONS if d["status"] == "verified"]
        await websocket.send_json({"type": "history", "data": public})
    await websocket.send_json({"type": "stats", "data": compute_stats()})

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _lock:
            pool.discard(websocket)


# --------------------------------------------------------------------------- #
#  Officials actions (verify / dismiss)
# --------------------------------------------------------------------------- #
@app.post("/api/detections/{det_id}/status")
async def set_status(det_id: str, status: str = Form(...), session: Optional[str] = Cookie(None)):
    if not current_user(session):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    if status not in ("verified", "dismissed", "pending"):
        return JSONResponse({"ok": False, "error": "bad status"}, status_code=400)
    for d in DETECTIONS:
        if d["id"] == det_id:
            d["status"] = status
            await broadcast_detection(d)
            return {"ok": True, "id": det_id, "status": status}
    return JSONResponse({"ok": False, "error": "not found"}, status_code=404)


# --------------------------------------------------------------------------- #
#  REST
# --------------------------------------------------------------------------- #
@app.get("/api/stats")
async def api_stats():
    return JSONResponse(compute_stats())


@app.get("/api/detections")
async def api_detections(session: Optional[str] = Cookie(None)):
    if current_user(session):
        return JSONResponse(DETECTIONS)
    return JSONResponse([public_view(d) for d in DETECTIONS if d["status"] == "verified"])


# --------------------------------------------------------------------------- #
#  Auth routes
# --------------------------------------------------------------------------- #
@app.post("/login")
async def do_login(
    username: str = Form(...),
    password: str = Form(...),
    remember: Optional[str] = Form(None),
):
    if OFFICIALS.get(username) == password:
        token = secrets.token_urlsafe(24)
        SESSIONS[token] = username
        resp = RedirectResponse(url="/officials", status_code=303)
        # "Remember me" -> persistent 30-day cookie; otherwise a browser-session cookie.
        max_age = 60 * 60 * 24 * 30 if remember else None
        resp.set_cookie("session", token, httponly=True, samesite="lax", max_age=max_age)
        return resp
    return RedirectResponse(url="/login?error=1", status_code=303)


@app.get("/logout")
async def logout(session: Optional[str] = Cookie(None)):
    SESSIONS.pop(session, None)
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie("session")
    return resp


# --------------------------------------------------------------------------- #
#  Page routes
# --------------------------------------------------------------------------- #
def _page(name: str) -> str:
    with open(os.path.join(STATIC_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/", response_class=HTMLResponse)
async def community():
    return HTMLResponse(_page("community.html"))


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(_page("login.html"))


@app.get("/officials", response_class=HTMLResponse)
async def officials_page(session: Optional[str] = Cookie(None)):
    if not current_user(session):
        return RedirectResponse(url="/login", status_code=303)
    return HTMLResponse(_page("officials.html"))
