# Implementation plan: Authenticated officials live feed

- [ ] 1. Add test tooling
  - Add `pytest`, `httpx` and `hypothesis` (pinned versions) to a dev requirements file.
  - Create `tests/test_ws_auth.py` with a FastAPI `TestClient` fixture.
  - _Requirements: 1, 2_

- [ ] 2. Write failing tests first
  - 2.1 Officials channel with no cookie or a bogus cookie is rejected and receives no `history`.
  - 2.2 Officials channel with a valid session receives the full `history`.
  - 2.3 Public channel with no cookie receives only verified, trimmed records.
  - 2.4 Property: `public_view()` never returns `coordinates`, `location`, `locality`, `video_url`.
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2_

- [ ] 3. Gate the officials channel in `ws_dashboard`
  - Read the `session` cookie from the handshake; close with 1008 before `accept()` if invalid.
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 4. Disconnect officials sockets on logout
  - Track `OFFICIAL_SESSIONS`; close matching sockets in `/logout`.
  - _Requirements: 3.1_

- [ ] 5. Client handling in `static/officials.html`
  - On close code 1008 (or an immediate first-connect failure), redirect to `/login`.
  - _Requirements: 3.2_

- [ ] 6. Verify end to end
  - Run the tests. Log in and confirm the officials feed works, then log out and confirm
    it stops. Confirm the community page still loads verified incidents.
  - _Requirements: 1, 2, 3_
