# REVERE — 3-Minute Video Script  *(Director's Cut)*

**Runtime:** 3:00 · **Goal:** hook emotionally, prove it works, show we built it deeply with Kiro.
**Golden rule:** the live demo is the hero — give it half the runtime. Cut your own talking before you cut the demo.

**Format legend**
- **[SHOT]** = what the camera/screen shows
- **VO:** = voiceover (what you say)
- **[TEXT]** = on-screen caption/lower-third
- **[SFX/MUSIC]** = audio cue

---

## COLD OPEN — THE GAP  ·  0:00–0:22

*A hook grounded in a real, recent, local event. Start quiet and human, not techy.*

**[MUSIC]** Low, tense ambient pad. Single piano note.

**[SHOT]** Black screen. White text fades in, one line at a time, slowly:
> **[TEXT]** *March 2026. Northeastern University, Boston.*
> **[TEXT]** *A student is stabbed outside the campus rec center.*
> **[TEXT]** *The first safety alert reaches students 47 minutes later.*

**VO (calm, serious):**
> "In March 2026, a student was stabbed just outside Northeastern's rec center.
> The first alert didn't reach the campus for almost forty-seven minutes.
> Students rallied — not about the crime, but about how long it took to *know*."

**[SHOT]** Cut to a slow push-in on a Boston street at dusk — a surveillance camera on a pole.

**VO:**
> "The cameras were already watching. The information just wasn't moving."

**[MUSIC]** One beat of silence.

> *Sources: CBS Boston, Boston.com / The Huntington News — reported March–April 2026.*

---

## THE SCALE — 1,200 EYES  ·  0:22–0:40

*Turn the problem into an opportunity. This is the "aha."*

**[SHOT]** Map of Boston. Dots bloom across it — Roxbury, Dorchester, Downtown, East Boston — until there are over a thousand.

**[TEXT]** *Boston PD operates ~1,200 public surveillance cameras.*

**VO (building energy):**
> "Boston already runs around twelve hundred public cameras across the city.
> That's twelve hundred cameras that *record* everything — and *understand* nothing.
> What if they could?"

**[SHOT]** The REVERE title snaps in over the map.

**[TEXT]** **REVERE — turning Boston's cameras into real-time safety intelligence.**

**VO:**
> "That's Revere. We built it to sit on top of the cameras Boston already has."

> *Camera count per Boston Globe (2024): ~1,300 BPD cameras; ACLU/City report: 1,153 as of July 2024.*

---

## WHAT IT IS  ·  0:40–0:52

**[SHOT]** Slide 2 (two audiences).

**VO (quick, confident):**
> "One AI detector, two audiences.
> Residents get a live crime map of their neighborhood — no login.
> Officials review every detection and decide what's real.
> And it's private by design: only human-verified incidents ever go public."

---

## LIVE DEMO — THE HERO  ·  0:52–2:05  *(~70s, screen recording)*

*Narrate while you click. Keep momentum. Pre-seed verified incidents so the map is alive; run ONE fresh detection so judges watch it appear.*

**[SHOT]** The detector app.
**VO:** "Here's a camera feed running through our detector."

**[SHOT]** A detection fires — crime label + confidence.
**VO:** "It flags the incident, cuts a clip, and streams it — with the location — the moment it happens."

**[SHOT]** Officials dashboard; the card appears in the review queue.
**VO:** "On the officials dashboard it lands instantly: the clip, the neighborhood, the time. An officer watches it…"

**[SHOT]** Click **Verify**.
**VO:** "…and verifies it with one click."

**[SHOT]** Cut to the public community map — a marker appears in its neighborhood, color-coded; the ranking and chart tick up.
**VO:** "The instant it's verified, the public map updates — live, in the right Boston neighborhood, color-coded by severity."

**[SHOT]** Gesture across the map / rankings.
**VO:** "Residents see the pattern. Officials keep control. And it all runs over the network — a detector on any camera, anywhere in the city, streams in with just an IP."

---

## BUILT WITH KIRO  ·  2:05–2:42

**[SHOT]** Slide 5 (Built with Kiro).

**VO (this is your craft moment — be specific):**
> "We built all of this in Kiro — not as autocomplete, as our entire dev environment.
> We inherited a broken codebase, and Kiro read it and explained it before changing a line.
> It caught bugs we hadn't even noticed — our video clips weren't playable in a browser —
> fixed the codec, and *verified the fix itself* with ffprobe.
> For security, we went spec-driven: Kiro found our officials feed didn't require a login,
> and we locked it down with testable criteria before writing the fix.
> Hooks, steering files, and the Figma integration kept our quality and design consistent
> while we moved fast."

---

## IMPACT & CLOSE  ·  2:42–3:00

**[SHOT]** Slide 6 (impact). Numbers animate in.

**VO (lift the energy, then land it):**
> "Our detector hits 0.95 ROC AUC telling crime from normal footage — and everything is real-time.
> Boston already has the cameras. Revere gives them awareness —
> turning forty-seven minutes into the moment it happens."

**[SHOT]** Black. The logo and tagline.
**[TEXT]** **REVERE — see it, verify it, prevent it.**

**VO (quiet, final):**
> "Revere. Built with Kiro."

**[MUSIC]** Resolve the pad to a clean, hopeful chord. Hold two seconds. Cut.

---

## DIRECTOR'S NOTES

**Emotional arc:** dread (the gap) → possibility (1,200 cameras) → proof (demo) → credibility (Kiro) → hope (close). Ride that curve; don't stay flat.

**Pacing:** cold open is slow and deliberate; the "1,200 eyes" turn speeds up; the demo is brisk but legible; Kiro is measured and specific; the close slows again for impact.

**Visual motifs:** the camera, the map, the glowing dots. Reuse them so the film feels designed, not assembled.

**Audio:** one calm narration take beats fancy editing. Keep music under the voice. Let the one beat of silence after "wasn't moving" breathe.

**Captions:** burn in subtitles — judges often watch muted.

## PRODUCTION CHECKLIST
- [ ] Backend running with a few **verified** incidents (map not empty).
- [ ] One **fresh** detection staged to run live for the "it appears" beat.
- [ ] If V-JEPA 2 inference is slow on CPU, **pre-record and speed up 1.5–2×** the wait; keep the map-update real-time.
- [ ] Fallback: `python seed_demo.py` to populate data, then demo **verify → public map** only. 100% reliable.
- [ ] Rehearse once with a timer. The demo always runs long — trim narration, never the demo.
- [ ] Export 1080p, subtitles on.

## FACTS USED (verify before publishing)
- **Northeastern stabbing, March 2026** near Marino Recreation Center; first alert ~47 min later; student rally over communication. (CBS Boston; Boston.com / The Huntington News.)
- **Boston PD camera network ~1,200–1,300** public cameras. (Boston Globe, Jul 2024: ~1,300; ACLU/City report: 1,153 as of Jul 2024.)
- **0.95 video ROC AUC** — our V-JEPA 2 + MIL detector on the UCF-Crime test split (see `poc/training_phase1.md`).

> Keep the tone respectful: a real person was hurt in the opening event. Frame it as motivation to help, never as spectacle.
