# Design: Authenticated officials live feed

## Current behavior
`ws_dashboard` in `server.py` reads `channel` from the query string, calls
`websocket.accept()`, and adds the socket to `OFFICIAL_CLIENTS` whenever
`channel == "officials"`. There is no `current_user()` check.

## Approach
Browsers send cookies on same-origin WebSocket handshakes, so the existing `session`
cookie (set `httponly`, `samesite=lax` at `/login`) is available on the handshake. No
token needs to go in the URL.

```python
@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    channel = websocket.query_params.get("channel", "public")
    session = websocket.cookies.get("session")
    if channel == "officials" and not current_user(session):
        await websocket.close(code=1008)        # reject before accept(): no data sent
        return
    await websocket.accept()
    ...
```

To support Requirement 3, track which session opened each officials socket:
`OFFICIAL_SESSIONS: Dict[WebSocket, str]`. On `/logout`, close every socket whose
session matches the one being removed, then discard it from `OFFICIAL_CLIENTS`.

### Client change (`static/officials.html`)
In `ws.onclose`, if `event.code === 1008`, set `location.href = '/login'` and don't
schedule a reconnect.

## Data exposure check
`public_view()` stays the only serializer for the public channel. A test asserts its
output keys are exactly `{id, crime, neighborhood, date, time, received_at, severity, status}`.

## Testing strategy
- **Unit (FastAPI `TestClient`):** officials channel with no cookie, a bogus cookie, and a
  valid cookie; public channel with no cookie.
- **Property-based (Hypothesis):** for any generated detection dict, `public_view(d)` never
  contains `coordinates`, `location`, `locality` or `video_url`. For any random session
  string not in `SESSIONS`, an officials connection is rejected.

## Risks
- Closing before `accept()` returns HTTP 403 on the handshake in Starlette. Browsers
  surface that as close code 1006, not 1008, so the client also treats an immediate
  failure on first connect as "go to login". Verify this in Chrome.
