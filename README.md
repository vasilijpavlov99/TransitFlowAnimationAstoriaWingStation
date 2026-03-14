## Transit Flow Animation — Astoria Wing Station

A data visualization tool that generates an animated MP4 (or GIF fallback) illustrating the urban planning concept behind the proposed **Astoria Wing Station** — a new transit hub designed to decentralize NYC's Manhattan-centric rail network by enabling direct interborough connections.

### What it does

The 30-second, 1920×1080 animation walks through three narrative acts:

- **Act 1 — Current State:** Animated particles flow along Bezier-curved paths showing how all regional rail, subway, and airport connections funnel into Manhattan hubs (Penn Station, Grand Central).
- **Transition:** The Manhattan-centric network fades out as Astoria-Ditmars begins to glow, signaling the shift.
- **Act 2 — Proposed State:** Teal particles radiate to and from a new Astoria hub, connecting LGA Airport, Long Island, New Haven, Philadelphia, Newark, the Bronx, Brooklyn, and Queens — bypassing Manhattan entirely.

### Tech stack

- **Python** with `matplotlib` for rendering and animation
- `FuncAnimation` with `FFMpegWriter` (MP4) or `PillowWriter` (GIF fallback)
- Quadratic Bezier curves for smooth edge paths
- Ease-in/out and ease-out interpolation for cinematic transitions
- Particle system with per-frame alpha fading and positional animation

### Output

| File | Description |
|---|---|
| `transit_animation.mp4` | Full 30s HD animation (requires ffmpeg) |
| `transit_animation.gif` | Reduced-resolution fallback |
| `transit_keyframe.png` | Static snapshot of the proposed network (mid-Act 2) |

### Usage

```bash
pip install matplotlib numpy imageio-ffmpeg
python transit_animation.py
```

---
*Created by Vasilij Pavlov, MArch 2025, Parsons School of Design.*
