// Turns a pose-landmark time series into a stroke score out of 16.5.
//
// This is a heuristic, body-mechanics-only scorer: MediaPipe Pose only sees
// the player's body, not the racket or ball, so it can't judge spin, pace or
// placement. Instead it grades six coaching-cue proxies that ARE visible in
// body pose (leg drive, rotation, arm extension, contact position, balance,
// follow-through). Treat the score as a consistent relative indicator of
// technique, not a professional/certified rating.
//
// Each of the six criteria is worth a slice of the 16.5-point total:
//   leg drive / knee load-and-drive        3.0
//   rotation (shoulder-hip separation)     3.0
//   arm extension at contact               3.0
//   contact position (height/reach)        3.0
//   balance / head stability               2.5
//   follow-through                         2.0
//   ------------------------------------------
//   total                                 16.5

export const STROKE_SERVE = "Serve";
export const STROKE_FOREHAND = "Forehand Groundstroke";
export const STROKE_BACKHAND = "Backhand Groundstroke";
export const STROKES = [STROKE_SERVE, STROKE_FOREHAND, STROKE_BACKHAND];

// Indices into MediaPipe's 33-point BlazePose landmark array.
export const LANDMARK_INDEX = {
  NOSE: 0,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
};

function angleAt(a, b, c) {
  const v1 = [a[0] - b[0], a[1] - b[1]];
  const v2 = [c[0] - b[0], c[1] - b[1]];
  const n1 = Math.hypot(v1[0], v1[1]);
  const n2 = Math.hypot(v2[0], v2[1]);
  if (n1 < 1e-6 || n2 < 1e-6) return 0;
  let cosTheta = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2);
  cosTheta = Math.max(-1, Math.min(1, cosTheta));
  return (Math.acos(cosTheta) * 180) / Math.PI;
}

function lineAngle(p1, p2) {
  const dx = p2[0] - p1[0];
  const dy = p2[1] - p1[1];
  let ang = (Math.atan2(dy, dx) * 180) / Math.PI;
  ang = ((ang % 180) + 180) % 180;
  return ang;
}

function rotationSeparation(shoulderL, shoulderR, hipL, hipR) {
  const shoulderAng = lineAngle(shoulderL, shoulderR);
  const hipAng = lineAngle(hipL, hipR);
  const diff = Math.abs(shoulderAng - hipAng);
  return Math.min(diff, 180 - diff);
}

function clampedLinear(value, x0, x1) {
  if (x1 === x0) return value >= x0 ? 1 : 0;
  const frac = (value - x0) / (x1 - x0);
  return Math.max(0, Math.min(1, frac));
}

// 0..1 score: ramps up zeroLo->fullLo, flat at 1 until fullHi, ramps down
// fullHi->zeroHi. Values outside [zeroLo, zeroHi] score 0.
function trapezoid(value, zeroLo, fullLo, fullHi, zeroHi) {
  if (value < fullLo) return clampedLinear(value, zeroLo, fullLo);
  if (value <= fullHi) return 1;
  return clampedLinear(value, zeroHi, fullHi);
}

function torsoLength(points) {
  const sx = (points.LEFT_SHOULDER[0] + points.RIGHT_SHOULDER[0]) / 2;
  const sy = (points.LEFT_SHOULDER[1] + points.RIGHT_SHOULDER[1]) / 2;
  const hx = (points.LEFT_HIP[0] + points.RIGHT_HIP[0]) / 2;
  const hy = (points.LEFT_HIP[1] + points.RIGHT_HIP[1]) / 2;
  return Math.max(1, Math.hypot(hx - sx, hy - sy));
}

function shoulderWidth(points) {
  return Math.max(
    1,
    Math.hypot(
      points.LEFT_SHOULDER[0] - points.RIGHT_SHOULDER[0],
      points.LEFT_SHOULDER[1] - points.RIGHT_SHOULDER[1]
    )
  );
}

function std(values) {
  if (values.length < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

/**
 * @param {{frames: Array<{tMs:number, points:Object}>, detectedFrames:number, totalFrames:number}} series
 * @param {string} stroke one of STROKES
 * @param {"RIGHT"|"LEFT"} dominantHand hitting arm
 */
export function scoreStroke(series, stroke, dominantHand) {
  const warnings = [];
  const frames = series.frames;
  const detectionRate =
    series.totalFrames > 0 ? series.detectedFrames / series.totalFrames : 0;

  if (detectionRate < 0.5) {
    warnings.push(
      `Pose was only detected in ${Math.round(
        detectionRate * 100
      )}% of frames — make sure the full body is visible and well lit for more reliable scoring.`
    );
  }
  if (frames.length < 5) {
    warnings.push(
      "Very few frames had a detected pose; the clip may be too short or the player too hard to see. Score may be unreliable."
    );
    return {
      total: 0,
      maxTotal: 16.5,
      criteria: [],
      contactFrameTMs: null,
      detectionRate,
      warnings,
    };
  }

  const side = dominantHand.toUpperCase();
  const shoulderKey = `${side}_SHOULDER`;
  const elbowKey = `${side}_ELBOW`;
  const wristKey = `${side}_WRIST`;
  const hipKey = `${side}_HIP`;

  const torsoRef =
    frames.reduce((a, f) => a + torsoLength(f.points), 0) / frames.length;
  const shoulderRef =
    frames.reduce((a, f) => a + shoulderWidth(f.points), 0) / frames.length;

  // ---- leg drive: how much the knees load (bend) and then extend ----
  const kneeAngles = frames.map((f) => {
    const p = f.points;
    const leftKnee = angleAt(p.LEFT_HIP, p.LEFT_KNEE, p.LEFT_ANKLE);
    const rightKnee = angleAt(p.RIGHT_HIP, p.RIGHT_KNEE, p.RIGHT_ANKLE);
    return (leftKnee + rightKnee) / 2;
  });
  const kneeRange = Math.max(...kneeAngles) - Math.min(...kneeAngles);
  const legFrac = trapezoid(kneeRange, 5, 20, 70, 100);
  const legPoints = round2(legFrac * 3.0);

  // ---- rotation: peak shoulder/hip separation (X-factor proxy) ----
  const separations = frames.map((f) =>
    rotationSeparation(
      f.points.LEFT_SHOULDER,
      f.points.RIGHT_SHOULDER,
      f.points.LEFT_HIP,
      f.points.RIGHT_HIP
    )
  );
  const peakSep = Math.max(...separations);
  const rotFrac = trapezoid(peakSep, 3, 15, 55, 80);
  const rotPoints = round2(rotFrac * 3.0);

  // ---- find the contact frame ----
  let contactIdx;
  if (stroke === STROKE_SERVE) {
    // Highest point (smallest y) reached by the hitting wrist.
    contactIdx = frames.reduce(
      (best, f, i) => (f.points[wristKey][1] < frames[best].points[wristKey][1] ? i : best),
      0
    );
  } else {
    // Frame where the hitting wrist is farthest in front of the hitting hip.
    contactIdx = frames.reduce((best, f, i) => {
      const reach = Math.abs(f.points[wristKey][0] - f.points[hipKey][0]);
      const bestReach = Math.abs(
        frames[best].points[wristKey][0] - frames[best].points[hipKey][0]
      );
      return reach > bestReach ? i : best;
    }, 0);
  }
  const contactFrame = frames[contactIdx];
  const cp = contactFrame.points;

  // ---- arm extension at contact ----
  const elbowAngle = angleAt(cp[shoulderKey], cp[elbowKey], cp[wristKey]);
  const extFrac = trapezoid(elbowAngle, 90, 155, 181, 181);
  const extPoints = round2(extFrac * 3.0);

  // ---- contact position ----
  let posFrac;
  if (stroke === STROKE_SERVE) {
    const rise = (cp[shoulderKey][1] - cp[wristKey][1]) / torsoRef;
    posFrac = trapezoid(rise, -0.2, 0.5, 2.0, 3.0);
  } else {
    const reach = Math.abs(cp[wristKey][0] - cp[hipKey][0]) / shoulderRef;
    posFrac = trapezoid(reach, 0.1, 0.6, 2.2, 3.0);
  }
  const posPoints = round2(posFrac * 3.0);

  // ---- balance / head stability across the whole swing ----
  const noseX = frames.map((f) => f.points.NOSE[0] / shoulderRef);
  const noseY = frames.map((f) => f.points.NOSE[1] / shoulderRef);
  const sway = Math.hypot(std(noseX), std(noseY));
  const balFrac = 1 - clampedLinear(sway, 0.15, 0.9);
  const balPoints = round2(balFrac * 2.5);

  // ---- follow-through: hitting-wrist travel after contact ----
  const tail = frames.slice(contactIdx);
  let pathLen = 0;
  for (let i = 0; i < tail.length - 1; i++) {
    const [ax, ay] = tail[i].points[wristKey];
    const [bx, by] = tail[i + 1].points[wristKey];
    pathLen += Math.hypot(bx - ax, by - ay);
  }
  pathLen /= torsoRef;
  const ftFrac = trapezoid(pathLen, 0.1, 0.8, 6.0, 9.0);
  const ftPoints = round2(ftFrac * 2.0);

  const criteria = [
    {
      name: "Leg drive (knee load & extend)",
      points: legPoints,
      maxPoints: 3.0,
      detail: `knee angle range ${kneeRange.toFixed(0)}°`,
    },
    {
      name: "Rotation (shoulder-hip separation)",
      points: rotPoints,
      maxPoints: 3.0,
      detail: `peak separation ${peakSep.toFixed(0)}°`,
    },
    {
      name: "Arm extension at contact",
      points: extPoints,
      maxPoints: 3.0,
      detail: `elbow angle ${elbowAngle.toFixed(0)}° at contact`,
    },
    {
      name: "Contact position",
      points: posPoints,
      maxPoints: 3.0,
      detail:
        stroke === STROKE_SERVE
          ? "wrist rise vs. torso"
          : "reach vs. shoulder width",
    },
    {
      name: "Balance / head stability",
      points: balPoints,
      maxPoints: 2.5,
      detail: `head sway ${sway.toFixed(2)} (lower is steadier)`,
    },
    {
      name: "Follow-through",
      points: ftPoints,
      maxPoints: 2.0,
      detail: `wrist path after contact ${pathLen.toFixed(1)}x torso length`,
    },
  ];

  const rawTotal = criteria.reduce((a, c) => a + c.points, 0);
  let total = Math.round(rawTotal * 2) / 2; // nearest 0.5
  total = rawTotal > 0 ? Math.max(1.0, Math.min(16.5, total)) : 0;

  return {
    total,
    maxTotal: 16.5,
    criteria,
    contactFrameTMs: contactFrame.tMs,
    detectionRate,
    warnings,
  };
}

function round2(v) {
  return Math.round(v * 100) / 100;
}
