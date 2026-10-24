# Requirements: Authenticated officials live feed

## Introduction
The officials portal page (`/officials`) requires a login, but its live data feed,
`/ws/dashboard?channel=officials`, does not check the session. Anyone who knows the URL
can receive every detection, including pending ones with exact coordinates and clip
links. This feature closes that gap without changing the public feed.

## Requirement 1: Only signed-in officials receive the officials feed
**User story:** As a resident, I want unverified detections and exact locations to stay
private, so that unreviewed AI output is never exposed publicly.

### Acceptance criteria
1. WHEN a client connects with `channel=officials` AND its `session` cookie maps to a
   signed-in official, THEN the server SHALL accept the connection and send the full
   `history` and `stats` messages.
2. WHEN a client connects with `channel=officials` AND it has no valid session, THEN the
   server SHALL close the connection with WebSocket code 1008 (policy violation) before
   sending any detection data.
3. WHEN a client is rejected, THEN the server SHALL NOT add it to `OFFICIAL_CLIENTS`.

## Requirement 2: The public feed is unchanged
**User story:** As a resident, I want the community map to keep working without an account.

### Acceptance criteria
1. WHEN a client connects with `channel=public` or no channel, THEN the server SHALL
   accept it without a session and send only `public_view()` records of verified incidents.
2. WHEN any record is sent on the public channel, THEN it SHALL NOT contain `coordinates`,
   `location`, `locality` or `video_url`.

## Requirement 3: Logged-out officials are disconnected
**User story:** As an official, I want logging out to stop my live feed on that browser.

### Acceptance criteria
1. WHEN an official logs out, THEN any officials-channel sockets opened with that session
   SHALL be closed with code 1008.
2. WHEN the officials page receives close code 1008, THEN it SHALL redirect to `/login`
   instead of reconnecting in a loop.
