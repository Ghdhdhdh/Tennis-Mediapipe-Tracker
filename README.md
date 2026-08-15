# Tennis Stroke Analyzer

A Streamlit app that uses [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
to score the technique of a serve or groundstroke from a video, on a scale
from 1 to 16.5.

## How it works

1. **Pick your stroke** — Serve, Forehand Groundstroke, or Backhand
   Groundstroke — and your dominant (racket) hand.
2. **Upload a video** of yourself hitting the stroke.
3. **Trim the clip** down to the wind-up through the follow-through using
   the range slider, and preview the trimmed result.
4. **Analyze** — MediaPipe Pose runs over every sampled frame of the
   trimmed clip, tracking your body's joints. The app then scores six
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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pose model (~6 MB) is downloaded automatically into `models/` the
first time you run an analysis, so an internet connection is needed at
least once.

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Project layout

- `app.py` — Streamlit UI and the 4-step workflow.
- `pose_engine.py` — Wraps MediaPipe's `PoseLandmarker` to turn a video
  into a per-frame landmark time series.
- `scoring.py` — Turns a landmark time series into the 16.5-point score.
- `video_utils.py` — Video probing/trimming helpers (OpenCV + MoviePy).
