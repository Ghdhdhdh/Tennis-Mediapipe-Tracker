"""Tennis Stroke Analyzer — Streamlit app.

Walks the user through: pick a stroke -> upload a video -> trim to the
swing -> run MediaPipe Pose over the clip -> show a technique score out
of 16.5 with a per-criterion breakdown.

Run with: streamlit run app.py
"""
import os
import shutil

import streamlit as st

from pose_engine import analyze_video
from scoring import STROKES, score_stroke
from video_utils import make_workdir, probe_video, save_upload, trim_video

st.set_page_config(page_title="Tennis Stroke Analyzer", page_icon="🎾", layout="centered")


def _init_state():
    defaults = {
        "workdir": None,
        "stroke": None,
        "hand": "RIGHT",
        "source_path": None,
        "video_info": None,
        "trimmed_path": None,
        "trim_range": None,
        "result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state.workdir is None:
        st.session_state.workdir = make_workdir()


def _reset():
    if st.session_state.workdir and os.path.isdir(st.session_state.workdir):
        shutil.rmtree(st.session_state.workdir, ignore_errors=True)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _init_state()


_init_state()

st.title("🎾 Tennis Stroke Analyzer")
st.caption(
    "MediaPipe Pose looks at your body mechanics — leg drive, rotation, arm "
    "extension, contact position, balance, and follow-through — and scores "
    "the stroke out of 16.5. It can't see the racket or ball, so treat the "
    "score as a technique proxy, not a radar gun."
)

with st.sidebar:
    st.header("Session")
    if st.button("Start over", use_container_width=True):
        _reset()
        st.rerun()
    st.markdown("---")
    st.markdown(
        "**Tips for a good score**\n"
        "- Side-on camera angle, full body in frame\n"
        "- Good, even lighting\n"
        "- Trim tightly around a single stroke\n"
    )

# ---------------------------------------------------------------- Step 1 --
st.header("1. Choose your stroke")
col1, col2 = st.columns(2)
with col1:
    stroke = st.radio("Stroke type", STROKES, index=STROKES.index(st.session_state.stroke)
                       if st.session_state.stroke in STROKES else 0)
with col2:
    hand_label = st.radio("Dominant (racket) hand", ["Right-handed", "Left-handed"],
                           index=0 if st.session_state.hand == "RIGHT" else 1)
st.session_state.stroke = stroke
st.session_state.hand = "RIGHT" if hand_label == "Right-handed" else "LEFT"

# ---------------------------------------------------------------- Step 2 --
st.header("2. Upload your video")
uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "m4v", "webm"])

if uploaded is not None:
    if st.session_state.source_path is None or st.session_state.get("_last_upload_name") != uploaded.name:
        st.session_state.source_path = save_upload(uploaded, st.session_state.workdir)
        st.session_state.video_info = probe_video(st.session_state.source_path)
        st.session_state.trimmed_path = None
        st.session_state.result = None
        st.session_state["_last_upload_name"] = uploaded.name

if st.session_state.source_path:
    info = st.session_state.video_info
    st.video(st.session_state.source_path)
    st.caption(f"Duration: {info.duration:.1f}s · {info.width}x{info.height} · {info.fps:.0f} fps")

    # ------------------------------------------------------------ Step 3 --
    st.header("3. Trim to the stroke")
    st.write("Drag the range to cover the wind-up through the follow-through.")
    default_end = min(info.duration, max(1.0, info.duration))
    trim_range = st.slider(
        "Trim range (seconds)",
        min_value=0.0,
        max_value=max(0.5, info.duration),
        value=st.session_state.trim_range or (0.0, default_end),
        step=0.1,
    )
    st.session_state.trim_range = trim_range

    if st.button("Preview trimmed clip"):
        start, end = trim_range
        if end - start < 0.3:
            st.error("Select at least 0.3 seconds of video.")
        else:
            with st.spinner("Trimming..."):
                st.session_state.trimmed_path = trim_video(
                    st.session_state.source_path, start, end, st.session_state.workdir
                )
            st.session_state.result = None

    if st.session_state.trimmed_path:
        st.video(st.session_state.trimmed_path)

        # -------------------------------------------------------- Step 4 --
        st.header("4. Analyze")
        if st.button("Analyze stroke", type="primary"):
            progress = st.progress(0.0, text="Running pose detection...")

            def on_progress(frac: float):
                progress.progress(min(1.0, frac), text=f"Running pose detection... {frac:.0%}")

            with st.spinner("Analyzing..."):
                series = analyze_video(st.session_state.trimmed_path, progress_cb=on_progress)
                result = score_stroke(series, st.session_state.stroke, st.session_state.hand)
            progress.empty()
            st.session_state.result = result

    result = st.session_state.result
    if result:
        st.markdown("---")
        for w in result.warnings:
            st.warning(w)

        st.subheader(f"Score: {result.total:.1f} / {result.max_total:g}")
        st.progress(min(1.0, result.total / result.max_total))

        st.subheader("Breakdown")
        for c in result.criteria:
            st.write(f"**{c.name}** — {c.points:.2f} / {c.max_points:g}")
            st.progress(min(1.0, c.points / c.max_points) if c.max_points else 0.0)
            st.caption(c.detail)

        st.caption(
            f"Pose detected in {result.detection_rate:.0%} of sampled frames."
        )
else:
    st.info("Upload a video to continue.")
