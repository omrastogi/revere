# REVERE — Boston Crime Intelligence Platform

![REVERE detector on UCF-Crime clips: RED alerts for abuse, assault and armed robbery, GREEN for normal street footage](asset/preview_grid.jpg)

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

![Detection flow: camera stream → fine-tuned V-JEPA 2 → crime detection head → crime type (Head B) and what is happening + severity (Head C, dictionary) → alert](asset/flowchart.png)

```
 Detector (revere_websocket_new.py)     Backend (server.py)                Browsers
 ──────────────────────────────────     ───────────────────                ────────
 V-JEPA 2 + MIL heads, cuts a clip,     store clip + map to a Boston       Community  ◄─ ws /ws/dashboard?channel=public
 streams it  ─ws /ws/ingest─►           neighborhood, score severity,         (verified only, trimmed)
                                         broadcast per channel              Officials  ◄─ ws /ws/dashboard?channel=officials
                                                                               (full records + video)
```

## Tech stack

- **Backend:** FastAPI + Uvicorn, WebSockets, in-memory store
- **Frontend:** vanilla HTML/CSS/JS, [Leaflet](https://leafletjs.com) (keyless
  OpenStreetMap tiles), [Chart.js](https://www.chartjs.org)
- **Detector (newest, Phase 2):** frozen **V-JEPA 2** ViT-L + zone / type /
  explanation heads (`revere_websocket_camera.py`) — GREEN/YELLOW/RED zones, a
  natural-language description, and severity. See [`poc/training_phase2.md`](poc/training_phase2.md).
- **Detector (Phase 1):** V-JEPA 2 + MIL heads (`revere_websocket_new.py`) — see
  [`poc/training_phase1.md`](poc/training_phase1.md).
- **Detector (legacy):** TensorFlow/Keras CNN (`revere_detector.py`).

> **Model roadmap:** all detectors stream the same ingest message, so the
> backend and UI never change — only the detector swaps.

The platform also runs fine with **no detector at all** using simulated data via
`seed_demo.py`, which is the fastest way to see the dashboards.

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
> `--host 0.0.0.0` makes the backend reachable from other machines on your
> network (needed for the networked setup below). For purely local use,
> `127.0.0.1` also works.

### 5. Open it
- Community portal: <http://127.0.0.1:8000/>
- Officials login:   <http://127.0.0.1:8000/login>

The store is **in memory**, so it starts empty each time the backend starts.

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

## Running a detector

A detector flags incidents and streams them (clip + metadata) to the backend.
There are three, all with the same dark REVERE GUI — **Analyze Video File**,
**Live Camera**, a live activity log, and an editable **Server connection** field:

| Detector | Model | Notes |
|----------|-------|-------|
| `revere_websocket_camera.py` | **V-JEPA 2 + zone / type / explanation (Phase 2)** | **Newest.** GREEN/YELLOW/RED zones, a natural-language description, severity. |
| `revere_websocket_new.py`    | V-JEPA 2 + MIL heads (Phase 1) | Anomaly + crime type. |
| `revere_detector.py`         | TensorFlow/Keras CNN (legacy) | Needs `model_8000.h5`. Kept for reference. |

### Requirements (V-JEPA 2 detectors)

- **Python 3.11** recommended
- **`transformers >= 4.52`** and **`torch >= 2.1`** — V-JEPA 2 (`AutoVideoProcessor`)
  is not available in older versions. (An older environment fails with
  `cannot import name 'AutoVideoProcessor'`.)
- The trained heads shipped in the repo (`poc/heads/` for Phase 2, `poc/ckpt/` for Phase 1).
- The V-JEPA 2 weights `facebook/vjepa2-vitl-fpc64-256` download automatically on
  first run (~GB). GPU is much faster, but CPU works.

```bash
pip install -r requirements.txt
```

---

### A. Run everything on ONE machine (local)

1. Start the backend:
   ```bash
   python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```
2. In a second terminal, start a detector (defaults to `ws://127.0.0.1:8000`):
   ```bash
   python revere_websocket_camera.py
   ```
3. Choose **Analyze Video File** or **Live Camera**. Press `q` in the preview to
   stop a live feed. Detections appear on the officials dashboard; verify them to
   push them to the public map.

---

### B. Run the detector on a DIFFERENT PC on the same network

Run the **backend on one PC (the "server")** and a **detector on another PC**,
both on the same Wi-Fi/LAN.

**On the server PC:**

1. Start the backend bound to all interfaces:
   ```bash
   python -m uvicorn server:app --host 0.0.0.0 --port 8000
   ```
2. Find its LAN IPv4 (Windows):
   ```powershell
   ipconfig        # "IPv4 Address" under your Wi-Fi adapter, e.g. 10.20.0.54
   ```
3. Allow the port through the firewall **once** (admin PowerShell on the server PC):
   ```powershell
   New-NetFirewallRule -DisplayName "REVERE backend 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any
   ```

**On the detector PC:**

4. Clone + install, then point the detector at the server's IP (a bare IP is fine):
   ```bash
   python revere_websocket_camera.py --host 10.20.0.54
   # or the full URL:
   python revere_websocket_camera.py --backend ws://10.20.0.54:8000/ws/ingest
   ```
   You can also paste the IP into the GUI's **Server connection** field and click
   **Apply** → **Test** (it confirms "Backend reachable").

**Sanity check first:** on the detector PC, open `http://10.20.0.54:8000/` in a
browser. If the community page loads, the network path works. If not, it's almost
always the server PC's firewall (step 3) or a "guest" Wi-Fi that isolates clients.

---

### Environment overrides (all detectors)

| Variable            | Default                         | Purpose                           |
|---------------------|---------------------------------|-----------------------------------|
| `REVERE_WS_BACKEND` | `ws://127.0.0.1:8000/ws/ingest` | Backend ingest WebSocket URL      |
| `REVERE_CLIP_DIR`   | `./clips`                       | Where clips are written           |
| `REVERE_POC_DIR`    | `./poc`                         | Folder holding the heads + helpers|
| `REVERE_THRESHOLD`  | `0.5`                           | Anomaly-score threshold           |

(The `--host` / `--backend` flag takes priority over `REVERE_WS_BACKEND`.)
The legacy CNN detector uses `REVERE_MODEL_PATH` instead; place `model_8000.h5`
next to it or set that variable.

### POC details

The training methodology, benchmarks (video ROC AUC **0.948** on UCF-Crime), and
the full pipeline live in [`poc/`](poc/) — see
[`poc/training_phase1.md`](poc/training_phase1.md). `poc/demo.py` can also run the
model on a video file and write an annotated output, independent of the backend.

---

## Running the legacy detector (TensorFlow CNN)

`revere_detector.py` is the earlier detector (Keras `model_8000.h5`). It's kept
for reference; prefer the V-JEPA 2 detector above.

1. Place `model_8000.h5` next to the script (or set `REVERE_MODEL_PATH`).
2. Start the backend, then: `python revere_detector.py`
3. Choose **Analyze Video File** or **Live Camera**; press `q` to stop.

Env overrides: `REVERE_MODEL_PATH`, `REVERE_CLIP_DIR`, `REVERE_WS_BACKEND`.

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
  "crime":       "Violence",
  "location":    "Boston",
  "locality":    "Roxbury",
  "coordinates": "42.32,-71.09",
  "date":        "2026_10_24",
  "time":        "19_34_41",
  "confidence":  0.91,
  "description": "person fights with another person; weapon drawn",
  "severity":    4,
  "zone":        "RED",
  "filename":    "clip.webm",
  "video_b64":   "<base64 of the video file bytes>"
}
```
The server replies `{"ok": true, "id": "<detection id>"}`.
All of `confidence`, `description`, `severity`, and `zone` are **optional**
(the Phase-2 detector sends them). `severity` from the model is on a 1–5 scale;
the backend scales it to the app's 0–10 scale. The officials dashboard shows the
`description` and `confidence`; the public feed never includes them.

---

## Project layout

| File                      | Purpose                                           |
|---------------------------|---------------------------------------------------|
| `server.py`               | FastAPI app (auth, websockets, aggregation)       |
| `static/community.html`   | Public portal (Leaflet map + Chart.js)            |
| `static/login.html`       | Officials login                                   |
| `static/officials.html`   | Officials dashboard                               |
| `revere_websocket_camera.py` | **Newest detector** — V-JEPA 2 + zone/type/explanation (Phase 2) + GUI |
| `revere_websocket_new.py` | Phase-1 detector — V-JEPA 2 + MIL heads + GUI     |
| `revere_detector.py`      | Legacy detector — TensorFlow CNN + GUI            |
| `revere_sender.py`        | Reusable detector helper (`send_detection`)       |
| `send_test.py`            | Single-detection test client                      |
| `seed_demo.py`            | Bulk demo-data seeder                             |
| `poc/`                    | V-JEPA 2 training pipeline + checkpoints          |
| `requirements.txt`        | Python dependencies                               |
| `asset/`                  | README images (teaser, flowchart)                 |
| `clips/`                  | Saved video clips (runtime; git-ignored)          |

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
