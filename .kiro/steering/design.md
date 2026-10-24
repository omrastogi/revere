---
inclusion: fileMatch
fileMatchPattern: "static/**/*.html"
---

# Revere — UI design system

Apply these to every page in `static/`. They keep the community page and the
officials page visually consistent.

## Hard rules
- **Do not modify `static/login.html`.** It is the visual reference, not a target.
- **Square corners everywhere** (`border-radius:0`). Only true circles (status dots,
  map markers) keep `border-radius:50%`.
- Escape all detection data before inserting it into HTML (use the page's `esc()` helper).

## Palette
| Token | Value | Use |
|---|---|---|
| `--navy` | `#153D59` | header bar, primary buttons, selected states, chart bars |
| `--navy-dark` | `#0f2d42` | headings, hover on navy |
| `--navy-tint` | `#e8eef4` | pills, tags |
| `--bg` | `#f3f4f6` | page background |
| `--panel` | `#ffffff` | cards |
| `--line` | `#dfe3e8` | borders |
| `--low` / `--mid` / `--high` | `#2f7d4f` / `#b45309` / `#e8262b` | concern levels |

## Layout patterns
- Navy top bar (logo tile + "Revere", underlined active tab, white solid button right).
- White notice bar under it (emergency or authorized-personnel copy + live status).
- Hero: navy-dark `h1` ending in a period ("Review queue."), muted subtitle, update stamp right.
- Filters: uppercase 0.66rem muted labels; segmented toggles with a navy selected segment.

## Map
- Leaflet with **keyless** tiles only. Currently Esri Light Gray Canvas base + reference
  labels in a separate `labels` pane (z 450), tinted via CSS filter on `.leaflet-tile-pane`.
- CARTO basemaps now require an API key and OSM blocks non-browser tile requests; don't use them.
- Keep the attribution control on (license requirement).
