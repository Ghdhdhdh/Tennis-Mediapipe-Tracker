"""Turns a pose landmark time series into a stroke score out of 16.5.

This is a heuristic, body-mechanics-only scorer: MediaPipe Pose only sees
the player's body, not the racket or ball, so it can't judge spin, pace or
placement. Instead it grades six coaching-cue proxies that ARE visible in
body pose (leg drive, rotation, arm extension, contact position, balance,
follow-through). Treat the score as a consistent relative indicator of
technique, not a professional/certified rating.

Each of the six criteria is worth a slice of the 16.5-point total:
    leg drive / knee load-and-drive        3.0
    rotation (shoulder-hip separation)     3.0
    arm extension at contact               3.0
    contact position (height/reach)        3.0
    balance / head stability               2.5
    follow-through                         2.0
    ------------------------------------------
    total                                 16.5
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from pose_engine import PoseSeries

STROKE_SERVE = "Serve"
STROKE_FOREHAND = "Forehand Groundstroke"
STROKE_BACKHAND = "Backhand Groundstroke"
STROKES = (STROKE_SERVE, STROKE_FOREHAND, STROKE_BACKHAND)


@dataclass
class Criterion:
    name: str
    points: float
    max_points: float
    detail: str


@dataclass
class ScoreResult:
    total: float
    max_total: float
    criteria: list[Criterion]
    contact_frame_t_ms: int | None
    detection_rate: float
    warnings: list[str]


def _angle_at(a, b, c) -> float:
    """Angle (degrees) at point b formed by rays b->a and b->c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_theta = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def _line_angle(p1, p2) -> float:
    """Angle (degrees, 0-180) of the line p1-p2 relative to horizontal."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = math.degrees(math.atan2(dy, dx)) % 180
    return ang


def _rotation_separation(shoulder_l, shoulder_r, hip_l, hip_r) -> float:
    """Smallest-angle difference between the shoulder line and hip line."""
    shoulder_ang = _line_angle(shoulder_l, shoulder_r)
    hip_ang = _line_angle(hip_l, hip_r)
    diff = abs(shoulder_ang - hip_ang)
    return min(diff, 180 - diff)


def _clamped_linear(value: float, x0: float, x1: float) -> float:
    """Fraction 0..1 as value ramps from x0 (->0) to x1 (->1). Handles x1<x0."""
    if x1 == x0:
        return 1.0 if value >= x0 else 0.0
    frac = (value - x0) / (x1 - x0)
    return max(0.0, min(1.0, frac))


def _trapezoid(value: float, zero_lo: float, full_lo: float, full_hi: float, zero_hi: float) -> float:
    """0..1 score: ramps up zero_lo->full_lo, flat at 1 until full_hi, ramps
    down full_hi->zero_hi. Values outside [zero_lo, zero_hi] score 0."""
    if value < full_lo:
        return _clamped_linear(value, zero_lo, full_lo)
    if value <= full_hi:
        return 1.0
    return _clamped_linear(value, zero_hi, full_hi)


def _torso_length(points) -> float:
    sx = (points["LEFT_SHOULDER"][0] + points["RIGHT_SHOULDER"][0]) / 2
    sy = (points["LEFT_SHOULDER"][1] + points["RIGHT_SHOULDER"][1]) / 2
    hx = (points["LEFT_HIP"][0] + points["RIGHT_HIP"][0]) / 2
    hy = (points["LEFT_HIP"][1] + points["RIGHT_HIP"][1]) / 2
    return max(1.0, math.hypot(hx - sx, hy - sy))


def _shoulder_width(points) -> float:
    return max(
        1.0,
        math.hypot(
            points["LEFT_SHOULDER"][0] - points["RIGHT_SHOULDER"][0],
            points["LEFT_SHOULDER"][1] - points["RIGHT_SHOULDER"][1],
        ),
    )


def score_stroke(series: PoseSeries, stroke: str, dominant_hand: str) -> ScoreResult:
    """dominant_hand is 'RIGHT' or 'LEFT' (the hitting arm)."""
    warnings: list[str] = []
    frames = series.frames

    if series.detection_rate < 0.5:
        warnings.append(
            "Pose was only detected in "
            f"{series.detection_rate:.0%} of frames — make sure the full body "
            "is visible and well lit for more reliable scoring."
        )
    if len(frames) < 5:
        warnings.append(
            "Very few frames had a detected pose; the clip may be too short "
            "or the player too hard to see. Score may be unreliable."
        )
        return ScoreResult(0.0, 16.5, [], None, series.detection_rate, warnings)

    side = dominant_hand.upper()
    other = "LEFT" if side == "RIGHT" else "RIGHT"
    shoulder_key, elbow_key, wrist_key = f"{side}_SHOULDER", f"{side}_ELBOW", f"{side}_WRIST"
    hip_key = f"{side}_HIP"

    torso_ref = sum(_torso_length(f.points) for f in frames) / len(frames)
    shoulder_ref = sum(_shoulder_width(f.points) for f in frames) / len(frames)

    # ---- leg drive: how much the knees load (bend) and then extend ----
    knee_angles = []
    for f in frames:
        p = f.points
        left_knee = _angle_at(p["LEFT_HIP"], p["LEFT_KNEE"], p["LEFT_ANKLE"])
        right_knee = _angle_at(p["RIGHT_HIP"], p["RIGHT_KNEE"], p["RIGHT_ANKLE"])
        knee_angles.append((left_knee + right_knee) / 2)
    knee_range = max(knee_angles) - min(knee_angles)
    leg_frac = _trapezoid(knee_range, zero_lo=5, full_lo=20, full_hi=70, zero_hi=100)
    leg_points = round(leg_frac * 3.0, 2)

    # ---- rotation: peak shoulder/hip separation (X-factor proxy) ----
    separations = [
        _rotation_separation(f.points["LEFT_SHOULDER"], f.points["RIGHT_SHOULDER"],
                              f.points["LEFT_HIP"], f.points["RIGHT_HIP"])
        for f in frames
    ]
    peak_sep = max(separations)
    rot_frac = _trapezoid(peak_sep, zero_lo=3, full_lo=15, full_hi=55, zero_hi=80)
    rot_points = round(rot_frac * 3.0, 2)

    # ---- find the contact frame ----
    if stroke == STROKE_SERVE:
        # Highest point (smallest y) reached by the hitting wrist.
        contact_idx = min(range(len(frames)), key=lambda i: frames[i].points[wrist_key][1])
    else:
        # Frame where the hitting wrist is farthest in front of the hitting hip
        # in the swing direction (max horizontal distance from the hip).
        def horiz_reach(i):
            p = frames[i].points
            return abs(p[wrist_key][0] - p[hip_key][0])
        contact_idx = max(range(len(frames)), key=horiz_reach)
    contact_frame = frames[contact_idx]
    cp = contact_frame.points

    # ---- arm extension at contact ----
    elbow_angle = _angle_at(cp[shoulder_key], cp[elbow_key], cp[wrist_key])
    ext_frac = _trapezoid(elbow_angle, zero_lo=90, full_lo=155, full_hi=181, zero_hi=181)
    ext_points = round(ext_frac * 3.0, 2)

    # ---- contact position ----
    if stroke == STROKE_SERVE:
        # Wrist should be well above the hitting shoulder, normalized by torso length.
        rise = (cp[shoulder_key][1] - cp[wrist_key][1]) / torso_ref
        pos_frac = _trapezoid(rise, zero_lo=-0.2, full_lo=0.5, full_hi=2.0, zero_hi=3.0)
    else:
        # Wrist should be a healthy distance in front of the body, not jammed in.
        reach = abs(cp[wrist_key][0] - cp[hip_key][0]) / shoulder_ref
        pos_frac = _trapezoid(reach, zero_lo=0.1, full_lo=0.6, full_hi=2.2, zero_hi=3.0)
    pos_points = round(pos_frac * 3.0, 2)

    # ---- balance / head stability across the whole swing ----
    nose_x = [f.points["NOSE"][0] / shoulder_ref for f in frames]
    nose_y = [f.points["NOSE"][1] / shoulder_ref for f in frames]
    sway = (_std(nose_x) ** 2 + _std(nose_y) ** 2) ** 0.5
    bal_frac = 1.0 - _clamped_linear(sway, 0.15, 0.9)
    bal_points = round(bal_frac * 2.5, 2)

    # ---- follow-through: hitting-wrist travel after contact ----
    tail = frames[contact_idx:]
    path_len = 0.0
    for a, b in zip(tail, tail[1:]):
        ax, ay = a.points[wrist_key]
        bx, by = b.points[wrist_key]
        path_len += math.hypot(bx - ax, by - ay)
    path_len /= torso_ref
    ft_frac = _trapezoid(path_len, zero_lo=0.1, full_lo=0.8, full_hi=6.0, zero_hi=9.0)
    ft_points = round(ft_frac * 2.0, 2)

    criteria = [
        Criterion("Leg drive (knee load & extend)", leg_points, 3.0,
                  f"knee angle range {knee_range:.0f}°"),
        Criterion("Rotation (shoulder-hip separation)", rot_points, 3.0,
                  f"peak separation {peak_sep:.0f}°"),
        Criterion("Arm extension at contact", ext_points, 3.0,
                  f"elbow angle {elbow_angle:.0f}° at contact"),
        Criterion("Contact position", pos_points, 3.0,
                  ("wrist rise vs. torso" if stroke == STROKE_SERVE else "reach vs. shoulder width")),
        Criterion("Balance / head stability", bal_points, 2.5,
                  f"head sway {sway:.2f} (lower is steadier)"),
        Criterion("Follow-through", ft_points, 2.0,
                  f"wrist path after contact {path_len:.1f}x torso length"),
    ]

    raw_total = sum(c.points for c in criteria)
    total = round(raw_total * 2) / 2  # nearest 0.5
    total = max(1.0, min(16.5, total)) if raw_total > 0 else 0.0

    return ScoreResult(
        total=total,
        max_total=16.5,
        criteria=criteria,
        contact_frame_t_ms=contact_frame.t_ms,
        detection_rate=series.detection_rate,
        warnings=warnings,
    )


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return var ** 0.5
