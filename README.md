# REVERE — Boston Crime Intelligence Platform

REVERE is a real-time, AI-assisted crime-awareness platform for Boston. A
computer-vision detector flags incidents from CCTV/camera feeds and streams them
— with the video clip and metadata — to a FastAPI backend over WebSockets. The
backend powers two experiences:

- **Community portal** (public, no login) — residents explore crime rates across
  Boston neighborhoods (Roxbury, Dorchester, Mattapan, …) on a live map, with
  severity rankings, a crime-type breakdown, and a feed of *verified* incidents.
- **Officials portal** (login required) — a live command center showing every AI
  detection with its video clip and full metadata, plus **verify / dismiss**
  triage. Verified incidents are what the public sees.

## Architecture

```
 Detector (VigilAI_WebSocket.py)        Backend (server.py)                Browsers
 ───────────────────────────────        ───────────────────                ────────
 runs the model, saves a clip,          store clip + map to a Boston       Community  ◄─ ws /ws/dashboard?channel=public
 sends it over  ─ws /ws/ingest─►        neighborhood, score severity,         (verified only, trimmed)
                                         broadcast per channel              Officials  ◄─ ws /ws/dashboard?channel=officials
                                                                               (full records + video)
```

## Tech stack

- **Backend:** FastAPI + Uvicorn, WebSockets, in-memory store
- **Frontend:** vanilla HTML/CSS/JS, [Leaflet](https://leafletjs.com) (keyless
  OpenStreetMap tiles), [Chart.js](https://www.chartjs.org)
- **Detector:** TensorFlow/Keras + OpenCV (optional — the platform runs fine with
  simulated data via `seed_demo.py`)

---

## Quick start

### 1. Prerequisites
- Python 3.9+ (3.11 recommended)
- Git

### 2. Clone & enter
```bash
git clone https://github.com/omrastogi/revere.git
cd revere
```

### 3. Install dependencies

Just the web platform (no model):
```bash
pip install fastapi "uvicorn[standard]" websockets python-multipart
```

Everything including the detector:
```bash
pip install -r requirements.txt
```

### 4. Run the backend
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

> On Windows, if `python` isn't found, use `py` instead of `python`.

### 5. Open it
- Community portal: <http://127.0.0.1:8000/>
- Officials login:   <http://127.0.0.1:8000/login>

### 6. See it with data

Without a camera, populate demo detections across Boston neighborhoods:
```bash
python seed_demo.py 20
```
Refresh the community page — the map, rankings, and charts update live. Then log
in as an official and verify a few; verified incidents appear on the public feed.

---

## Officials login (demo credentials)

| Username | Password   |
|----------|------------|
| officer  | boston123  |
| admin    | vigilai    |

Defined in the `OFFICIALS` dict in `server.py`. **Replace these and use a real
user store with hashed passwords before any real deployment.**

---

## Running the real detector (optional)

`VigilAI_WebSocket.py` loads a Keras model and runs live inference on a video
file or webcam, then streams detections to the backend.

1. Place the model file `model_8000.h5` next to the script (or set
   `VIGIL_MODEL_PATH`).
2. Make sure the backend is running.
3. Launch:
   ```bash
   python VigilAI_WebSocket.py
   ```
4. In the window that opens, choose **Analyze Video File** or **Live Camera**.
   Press `q` to stop the feed; the detected clip is saved and streamed to the
   backend, appearing instantly on the officials dashboard.

Environment overrides:

| Variable           | Default                         | Purpose                        |
|--------------------|---------------------------------|--------------------------------|
| `VIGIL_MODEL_PATH` | `./model_8000.h5`               | Path to the Keras model        |
| `VIGIL_CLIP_DIR`   | `./clips`                       | Where clips are written        |
| `VIGIL_WS_BACKEND` | `ws://127.0.0.1:8000/ws/ingest` | Backend ingest WebSocket URL   |

---

## HTTP & WebSocket API

| Route                              | Auth | Purpose                                  |
|------------------------------------|------|------------------------------------------|
| `GET /`                            | no   | Community portal                         |
| `GET /login`                       | no   | Officials login form                     |
| `POST /login`                      | no   | Authenticate, set session cookie         |
| `GET /logout`                      | —    | Clear session                            |
| `GET /officials`                   | yes  | Officials dashboard                      |
| `GET /api/stats`                   | no   | Neighborhood + crime aggregates          |
| `GET /api/detections`              | both | Public: verified+trimmed · Auth: full    |
| `POST /api/detections/{id}/status` | yes  | Set `verified` / `dismissed` / `pending` |
| `WS /ws/ingest`                    | —    | Detector pushes detections               |
| `WS /ws/dashboard?channel=…`       | —    | Live updates (`public` or `officials`)   |
| `GET /clips/<file>`                | no   | Saved video clips                        |

### Ingest message format

The detector sends one JSON text message per detection over `/ws/ingest`:

```json
{
  "crime":       "Fighting",
  "location":    "Boston",
  "locality":    "Roxbury",
  "coordinates": "42.32,-71.09",
  "date":        "2026_10_24",
  "time":        "19_34_41",
  "filename":    "clip.avi",
  "video_b64":   "<base64 of the video file bytes>"
}
```
The server replies `{"ok": true, "id": "<detection id>"}`.

---

## Project layout

| File                    | Purpose                                      |
|-------------------------|----------------------------------------------|
| `server.py`             | FastAPI app (auth, websockets, aggregation)  |
| `static/community.html` | Public portal (Leaflet map + Chart.js)       |
| `static/login.html`     | Officials login                              |
| `static/officials.html` | Officials dashboard                          |
| `vigil_sender.py`       | Reusable detector helper (`send_detection`)  |
| `send_test.py`          | Single-detection test client                 |
| `seed_demo.py`          | Bulk demo-data seeder                        |
| `VigilAI_WebSocket.py`  | The detector (loads the real model)          |
| `requirements.txt`      | Python dependencies                          |
| `clips/`                | Saved video clips (runtime; git-ignored)     |

---

## Notes & caveats

- **State is in memory.** Restarting the backend clears detections (clips on disk
  remain). Swap the `DETECTIONS` list for a database to persist.
- **Auth is demo-grade** (plain dict + in-memory sessions). Use real users,
  hashed passwords, and signed sessions for production.
- **Map tiles** use OpenStreetMap's public servers (keyless). For heavy traffic,
  use a paid tile provider or your own tile cache per the OSM fair-use policy.
- **Large clips** are base64'd into one WebSocket message. For production, prefer
  a chunked or HTTP multipart upload for video.
- REVERE reflects AI detections reviewed by officials and is **not** a substitute
  for official police statistics.
