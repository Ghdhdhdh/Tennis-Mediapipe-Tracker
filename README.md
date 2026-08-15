# Tennis Stroke Analyzer

A web app that uses [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
to score the technique of a serve or groundstroke from a video, on a scale
from 1 to 16.5 — entirely in the browser, no server or upload required.

## How it works

1. **Pick your stroke** — Serve, Forehand Groundstroke, or Backhand
   Groundstroke — and your dominant (racket) hand.
2. **Upload a video** of yourself hitting the stroke. It's loaded locally
   in your browser and never sent anywhere.
3. **Trim the clip** down to the wind-up through the follow-through using
   the dual-handle range slider, and preview the trimmed result.
4. **Analyze** — MediaPipe Pose (running as WebAssembly, in-browser) scans
   the trimmed clip and tracks your body's joints. The app then scores six
   coaching-cue proxies visible in body pose:

   | Criterion | Points |
   |---|---|
   | Leg drive (knee load & extend) | 3.0 |
   | Rotation (shoulder-hip separation) | 3.0 |
   | Arm extension at contact | 3.0 |
   | Contact position (height for serves / reach for groundstrokes) | 3.0 |
   | Balance / head stability | 2.5 |
   | Follow-through | 2.0 |
   | **Total** | **16.5** |

MediaPipe Pose only sees the player's body — not the racket or ball — so
this can't measure spin, pace, or placement. Treat the score as a
consistent, relative read on technique, not a certified coaching rating.
For best results, film from the side with your full body in frame.

## Run it

This is a static, client-side web app (`index.html` + `app.js` +
`pose-scoring.js` + `style.css`) with no build step and no backend. It
needs to be served over HTTP (not opened as a `file://` URL) because it
uses ES modules and fetches the pose model at runtime:

```bash
python3 -m http.server 8000
# or: npx serve
```

Then open http://localhost:8000 in a recent Chrome, Edge, or Firefox.
The MediaPipe runtime (WASM) and pose model (~6 MB) are fetched from a
CDN/Google Cloud Storage on first use and cached by the browser, so you
need an internet connection at least once. Video files should be MP4
(H.264) or WebM — those are supported by all major browsers.

You can also deploy the four static files as-is to GitHub Pages, Netlify,
Vercel, or any static host.

## Project layout

- `index.html` — the 4-step wizard UI.
- `app.js` — wires up the UI, loads MediaPipe's `PoseLandmarker` (WASM),
  runs it over the trimmed clip, and renders the results.
- `pose-scoring.js` — pure scoring logic: turns a landmark time series
  into the 16.5-point breakdown. No DOM/MediaPipe dependencies.
- `style.css` — styling.

## Python/CLI version

The repo also includes an equivalent local/offline version built with
Streamlit and MediaPipe's Python SDK, useful if you'd rather run analysis
on a machine instead of in the browser: `app.py`, `pose_engine.py`,
`scoring.py`, and `video_utils.py`. It implements the exact same
4-step workflow and 16.5-point scoring model.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
