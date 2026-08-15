"""Small helpers for inspecting and trimming an uploaded video file."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import cv2
from moviepy import VideoFileClip


@dataclass
class VideoInfo:
    duration: float
    fps: float
    width: int
    height: int


def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(duration=duration, fps=fps, width=width, height=height)


def save_upload(uploaded_file, workdir: str) -> str:
    """Persists a Streamlit UploadedFile to disk and returns the path."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    dest = os.path.join(workdir, f"source{suffix}")
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def trim_video(src_path: str, start: float, end: float, workdir: str) -> str:
    """Cuts [start, end] seconds out of src_path and returns the new file path."""
    if end <= start:
        raise ValueError("End time must be after start time.")
    dest = os.path.join(workdir, "trimmed.mp4")
    with VideoFileClip(src_path) as clip:
        end = min(end, clip.duration)
        sub = clip.subclipped(start, end)
        sub.write_videofile(
            dest,
            codec="libx264",
            audio=False,
            logger=None,
            temp_audiofile=os.path.join(workdir, "temp-audio.m4a"),
        )
    return dest


def make_workdir() -> str:
    return tempfile.mkdtemp(prefix="tennis_tracker_")
