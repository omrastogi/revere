# Revere — tech and workflow

## Run
- Python venv at `.venv` (Homebrew Python blocks global pip). Use `.venv/bin/python`.
- Web deps only: `.venv/bin/pip install fastapi "uvicorn[standard]" websockets python-multipart`
- Server: `.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload`
- Pages: `/` (community), `/login`, `/officials`, `/logout`; JSON: `/api/stats`, `/api/detections`.
- Demo logins: `officer / boston123`, `admin / vigilai` (demo only, hardcoded in `server.py`).

## Data is in memory
`DETECTIONS` lives in memory, so **any edit to `server.py` triggers a reload that wipes it.**
After editing `server.py`, reseed and verify some incidents so the public page has data:

```
.venv/bin/python seed_demo.py 20
```
then log in as `officer` and POST `status=verified` to `/api/detections/{id}/status` for a few ids.

## Verify before presenting
1. Syntax-check inline page scripts: `.venv/bin/python scripts/check_page_scripts.py static/<page>.html`
2. Visual check: headless Chrome screenshot, e.g.
   `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --window-size=1440,900 --virtual-time-budget=9000 --screenshot=/tmp/shot.png http://127.0.0.1:8000/`
   (For `/officials`, which needs a session, screenshot a temporary copy served from `/static/` and delete it afterward.)

## Conventions
- Plain HTML/CSS/vanilla JS in `static/`, no build step. Libraries via CDN: Leaflet 1.9.4, Chart.js 4.4.1.
- Keep all page JS free of new dependencies; prefer small helpers inside the page.
