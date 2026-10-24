# Revere — product rules

Revere is a Boston community-safety platform: a FastAPI + WebSocket backend
(`server.py`) ingests AI detections from the VigilAI detector, a public community
page (`static/community.html`) shows verified incidents, and a login-protected
officials portal (`static/officials.html`) is where reviewers verify or dismiss them.

## Privacy rules (never break these)
- The public page only ever shows incidents with `status == "verified"`.
- Public data goes through `public_view()` in `server.py`. It must never include
  `coordinates`, `location`, `locality` or `video_url`.
- Public map markers are placed at a stable, approximate offset (≤ ~600 m) from the
  neighborhood center, derived from the incident id. Never plot real coordinates publicly.
- Label approximate locations as approximate ("Map locations are approximate…").

## Domain definitions
- **Current** = incident time within the last 24 hours. **Historical** = older than 24 hours.
- Concern level from severity: `>= 8` High concern, `>= 6` Moderate, otherwise Low concern.
- All times are shown in Eastern Time (ET).
- Incident `date`/`time` arrive as `2026_10_24` / `18_22_00`; format them for display.

## Safety copy
- Every page keeps an emergency notice: "In an emergency, call 911. Revere is not an
  emergency dispatch service."
