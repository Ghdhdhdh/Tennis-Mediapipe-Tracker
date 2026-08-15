"""Runs MediaPipe Pose Landmarker over a video and returns a per-frame
time series of body landmarks, ready for stroke analysis.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_NAME = "pose_landmarker_lite.task"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Landmark indices we care about, keyed by readable name.
LM = {name: getattr(vision.PoseLandmark, name).value for name in (
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
)}


def ensure_model() -> str:
    """Downloads the pose landmarker model on first use and caches it locally."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return MODEL_PATH
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp_path = MODEL_PATH + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp_path)
    os.replace(tmp_path, MODEL_PATH)
    return MODEL_PATH


@dataclass
class FrameLandmarks:
    """Pixel-space (x, y) coordinates for the landmarks we track, one frame."""
    t_ms: int
    points: dict[str, tuple[float, float]]


@dataclass
class PoseSeries:
    frames: list[FrameLandmarks] = field(default_factory=list)
    width: int = 0
    height: int = 0
    fps: float = 30.0
    detected_frames: int = 0
    total_frames: int = 0

    @property
    def detection_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.detected_frames / self.total_frames


def analyze_video(
    video_path: str,
    max_fps: float = 15.0,
    progress_cb: Callable[[float], None] | None = None,
) -> PoseSeries:
    """Runs pose detection over a video and returns a landmark time series.

    Frames are subsampled to `max_fps` to keep processing fast; pose
    detection is run in MediaPipe's VIDEO mode so it can use temporal
    smoothing/tracking between frames.
    """
    model_path = ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    frame_stride = max(1, round(src_fps / max_fps)) if src_fps > 0 else 1
    effective_fps = src_fps / frame_stride if src_fps > 0 else max_fps

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    series = PoseSeries(width=width, height=height, fps=effective_fps)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        read_ok, frame = cap.read()
        while read_ok:
            if frame_idx % frame_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = _to_mp_image(rgb)
                t_ms = int((frame_idx / src_fps) * 1000) if src_fps > 0 else frame_idx
                result = landmarker.detect_for_video(mp_image, t_ms)

                series.total_frames += 1
                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    points = {
                        name: (lm[idx].x * width, lm[idx].y * height)
                        for name, idx in LM.items()
                    }
                    series.frames.append(FrameLandmarks(t_ms=t_ms, points=points))
                    series.detected_frames += 1

            frame_idx += 1
            if progress_cb and total_frames:
                progress_cb(min(1.0, frame_idx / total_frames))
            read_ok, frame = cap.read()

    cap.release()
    return series


def _to_mp_image(rgb_frame: np.ndarray):
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
