import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";
import { scoreStroke, LANDMARK_INDEX } from "./pose-scoring.js";

const WASM_BASE_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const ANALYSIS_FPS = 15;

const el = (id) => document.getElementById(id);

const video = el("sourceVideo");
const fileInput = el("fileInput");
const step3 = el("step3");
const step4 = el("step4");
const resultsSection = el("resultsSection");
const trimStart = el("trimStart");
const trimEnd = el("trimEnd");
const trimStartLabel = el("trimStartLabel");
const trimEndLabel = el("trimEndLabel");
const previewBtn = el("previewBtn");
const analyzeBtn = el("analyzeBtn");
const progressWrap = el("progressWrap");
const progressBar = el("progressBar");
const progressLabel = el("progressLabel");
const resetBtn = el("resetBtn");

let landmarkerPromise = null;
function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const fileset = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
      return PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    })();
  }
  return landmarkerPromise;
}

function selectedStroke() {
  return document.querySelector('input[name="stroke"]:checked').value;
}

function selectedHand() {
  return document.querySelector('input[name="hand"]:checked').value;
}

function fmtTime(s) {
  return `${s.toFixed(2)}s`;
}

// ---- Step 2: file upload ----
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  resultsSection.classList.add("hidden");
  const url = URL.createObjectURL(file);
  video.src = url;
  video.style.display = "block";
  video.setAttribute("controls", "true");
  video.load();
  video.addEventListener(
    "error",
    () => {
      el("videoMeta").textContent =
        "This browser couldn't play that video file. Try an MP4 (H.264) or WebM file.";
    },
    { once: true }
  );
  video.addEventListener(
    "loadedmetadata",
    () => {
      const duration = video.duration;
      el("videoMeta").textContent =
        `Duration: ${duration.toFixed(1)}s · ${video.videoWidth}x${video.videoHeight}`;
      trimStart.min = 0;
      trimStart.max = duration;
      trimStart.step = 0.05;
      trimStart.value = 0;
      trimEnd.min = 0;
      trimEnd.max = duration;
      trimEnd.step = 0.05;
      trimEnd.value = duration;
      trimStartLabel.textContent = fmtTime(0);
      trimEndLabel.textContent = fmtTime(duration);
      step3.classList.remove("hidden");
      step4.classList.add("hidden");
      resetBtn.classList.remove("hidden");
    },
    { once: true }
  );
});

// ---- Step 3: trim range ----
function syncTrimHandles() {
  const minGap = 0.2;
  if (Number(trimStart.value) > Number(trimEnd.value) - minGap) {
    trimStart.value = Math.max(0, Number(trimEnd.value) - minGap);
  }
  trimStartLabel.textContent = fmtTime(Number(trimStart.value));
  trimEndLabel.textContent = fmtTime(Number(trimEnd.value));
}
function syncTrimHandlesEnd() {
  const minGap = 0.2;
  if (Number(trimEnd.value) < Number(trimStart.value) + minGap) {
    trimEnd.value = Math.min(Number(trimEnd.max), Number(trimStart.value) + minGap);
  }
  trimStartLabel.textContent = fmtTime(Number(trimStart.value));
  trimEndLabel.textContent = fmtTime(Number(trimEnd.value));
}
trimStart.addEventListener("input", syncTrimHandles);
trimEnd.addEventListener("input", syncTrimHandlesEnd);

previewBtn.addEventListener("click", async () => {
  const start = Number(trimStart.value);
  const end = Number(trimEnd.value);
  video.currentTime = start;
  await video.play();
  const onTime = () => {
    if (video.currentTime >= end) {
      video.pause();
      video.removeEventListener("timeupdate", onTime);
    }
  };
  video.addEventListener("timeupdate", onTime);
  step4.classList.remove("hidden");
});

// ---- Step 4: analyze ----
function seekTo(t) {
  return new Promise((resolve) => {
    const onSeeked = () => {
      video.removeEventListener("seeked", onSeeked);
      resolve();
    };
    video.addEventListener("seeked", onSeeked);
    video.currentTime = t;
  });
}

async function extractSeries(start, end, onProgress) {
  const landmarker = await getLandmarker();
  const step = 1 / ANALYSIS_FPS;
  const frames = [];
  let detectedFrames = 0;
  let totalFrames = 0;
  video.pause();

  const width = video.videoWidth;
  const height = video.videoHeight;
  let t = start;
  let lastTsMs = -1;
  while (t <= end + 1e-6) {
    await seekTo(t);
    let tsMs = Math.round(t * 1000);
    if (tsMs <= lastTsMs) tsMs = lastTsMs + 1;
    lastTsMs = tsMs;

    const result = landmarker.detectForVideo(video, tsMs);
    totalFrames++;
    if (result.landmarks && result.landmarks.length > 0) {
      const lm = result.landmarks[0];
      const points = {};
      for (const [name, idx] of Object.entries(LANDMARK_INDEX)) {
        points[name] = [lm[idx].x * width, lm[idx].y * height];
      }
      frames.push({ tMs: tsMs, points });
      detectedFrames++;
    }
    onProgress(Math.min(1, (t - start) / (end - start || 1)));
    t += step;
  }
  return { frames, detectedFrames, totalFrames };
}

function renderResults(result) {
  resultsSection.classList.remove("hidden");
  const warningsEl = el("warnings");
  warningsEl.innerHTML = "";
  for (const w of result.warnings) {
    const div = document.createElement("div");
    div.className = "warning";
    div.textContent = w;
    warningsEl.appendChild(div);
  }

  el("scoreTotal").textContent = result.total.toFixed(1);
  el("scoreBar").style.width = `${Math.min(100, (result.total / result.maxTotal) * 100)}%`;

  const list = el("criteriaList");
  list.innerHTML = "";
  for (const c of result.criteria) {
    const div = document.createElement("div");
    div.className = "criterion";
    const pct = c.maxPoints ? Math.min(100, (c.points / c.maxPoints) * 100) : 0;
    div.innerHTML = `
      <div class="crit-head"><span>${c.name}</span><span>${c.points.toFixed(2)} / ${c.maxPoints}</span></div>
      <div class="progress"><div class="progress-bar" style="width:${pct}%"></div></div>
      <p class="detail">${c.detail}</p>
    `;
    list.appendChild(div);
  }

  el("detectionRateLabel").textContent =
    `Pose detected in ${Math.round(result.detectionRate * 100)}% of sampled frames.`;

  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

analyzeBtn.addEventListener("click", async () => {
  const start = Number(trimStart.value);
  const end = Number(trimEnd.value);
  if (end - start < 0.3) {
    alert("Select at least 0.3 seconds of video to analyze.");
    return;
  }

  analyzeBtn.disabled = true;
  progressWrap.classList.remove("hidden");
  progressBar.style.width = "0%";
  progressLabel.textContent = "Loading pose model...";

  try {
    await getLandmarker();
    progressLabel.textContent = "Running pose detection...";
    const series = await extractSeries(start, end, (frac) => {
      progressBar.style.width = `${frac * 100}%`;
      progressLabel.textContent = `Running pose detection... ${Math.round(frac * 100)}%`;
    });
    const stroke = selectedStroke();
    const hand = selectedHand();
    const result = scoreStroke(series, stroke, hand);
    renderResults(result);
  } catch (err) {
    console.error(err);
    alert(`Analysis failed: ${err.message || err}`);
  } finally {
    progressWrap.classList.add("hidden");
    analyzeBtn.disabled = false;
  }
});

// ---- Reset ----
resetBtn.addEventListener("click", () => {
  location.reload();
});
