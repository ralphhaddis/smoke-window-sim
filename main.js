import { SmokeFluidSolver } from './solver.js';
import { FluidRenderer } from './renderer.js';

const canvas = document.getElementById('simCanvas');
const solver = new SmokeFluidSolver(160, 96);
const renderer = new FluidRenderer(canvas, solver);

let running = true;
let stepsPerFrame = 2;
let toolMode = 'emitter'; // 'emitter' | 'wall' | 'erase'
let isPointerDown = false;

// --- Presets showcasing distinct fluid dynamics regimes ---
const PRESETS = {
  coldDay: {
    label: 'Cold Day Exchange',
    params: {
      tempIndoor: 22, tempOutdoor: 4, tempSmoke: 45,
      smokeDensityFactor: 0.0, pressureDiff: 0.0,
      windSpeedX: 0.0, windSpeedY: 0.0,
      emitterX: 0.45, emitterY: 0.42, jetSpeed: 0.25, jetAngleDeg: 25,
      windowSillFrac: 0.30, windowTopFrac: 0.72, windowBaffle: 0.0,
      rearVentOpen: false,
    },
  },
  windIn: {
    label: 'Wind Blowing In',
    params: {
      tempIndoor: 21, tempOutdoor: 14, tempSmoke: 38,
      smokeDensityFactor: 0.0, pressureDiff: 0.0,
      windSpeedX: -1.1, windSpeedY: 0.0,
      emitterX: 0.50, emitterY: 0.44, jetSpeed: 0.3, jetAngleDeg: 10,
      windowSillFrac: 0.32, windowTopFrac: 0.68, windowBaffle: 0.0,
      rearVentOpen: false,
    },
  },
  crossVent: {
    label: 'Cross-Ventilation (ΔP > 0)',
    params: {
      tempIndoor: 22, tempOutdoor: 18, tempSmoke: 36,
      smokeDensityFactor: 0.0, pressureDiff: 1.8,
      windSpeedX: 0.3, windSpeedY: 0.0,
      emitterX: 0.32, emitterY: 0.42, jetSpeed: 0.2, jetAngleDeg: 45,
      windowSillFrac: 0.32, windowTopFrac: 0.68, windowBaffle: 0.0,
      rearVentOpen: true,
    },
  },
  heavyFog: {
    label: 'Cold Dense Smoke',
    params: {
      tempIndoor: 22, tempOutdoor: 22, tempSmoke: 10,
      smokeDensityFactor: 0.06, pressureDiff: 0.0,
      windSpeedX: 0.0, windSpeedY: 0.0,
      emitterX: 0.48, emitterY: 0.58, jetSpeed: 0.15, jetAngleDeg: 0,
      windowSillFrac: 0.28, windowTopFrac: 0.68, windowBaffle: 0.0,
      rearVentOpen: false,
    },
  },
  hotSummer: {
    label: 'Hot Summer Day (Reversed)',
    params: {
      tempIndoor: 19, tempOutdoor: 36, tempSmoke: 34,
      smokeDensityFactor: 0.0, pressureDiff: 0.0,
      windSpeedX: 0.0, windSpeedY: 0.0,
      emitterX: 0.46, emitterY: 0.48, jetSpeed: 0.2, jetAngleDeg: 15,
      windowSillFrac: 0.30, windowTopFrac: 0.72, windowBaffle: 0.0,
      rearVentOpen: false,
    },
  },
};

// --- Bind sliders & controls to solver.params ---
const sliderBindings = [
  { id: 'tempIndoor', key: 'tempIndoor', fmt: v => `${v.toFixed(0)} °C` },
  { id: 'tempOutdoor', key: 'tempOutdoor', fmt: v => `${v.toFixed(0)} °C` },
  { id: 'tempSmoke', key: 'tempSmoke', fmt: v => `${v.toFixed(0)} °C` },
  { id: 'smokeDensityFactor', key: 'smokeDensityFactor', fmt: v => `${v.toFixed(3)}` },
  { id: 'pressureDiff', key: 'pressureDiff', fmt: v => `${v.toFixed(1)} Pa` },
  { id: 'windSpeedX', key: 'windSpeedX', fmt: v => `${v.toFixed(2)} m/s` },
  { id: 'windSpeedY', key: 'windSpeedY', fmt: v => `${v.toFixed(2)} m/s` },
  { id: 'jetSpeed', key: 'jetSpeed', fmt: v => `${v.toFixed(2)} m/s` },
  { id: 'jetAngleDeg', key: 'jetAngleDeg', fmt: v => `${v.toFixed(0)}°` },
  { id: 'viscosity', key: 'viscosity', fmt: v => v.toExponential(1) },
  { id: 'vorticityConf', key: 'vorticityConf', fmt: v => v.toFixed(2) },
  { id: 'windowSillFrac', key: 'windowSillFrac', fmt: v => `${Math.round(v * 100)}%`, geom: true },
  { id: 'windowTopFrac', key: 'windowTopFrac', fmt: v => `${Math.round(v * 100)}%`, geom: true },
  { id: 'wallXFrac', key: 'wallXFrac', fmt: v => `${Math.round(v * 100)}%`, geom: true },
  { id: 'roomCeilFrac', key: 'roomCeilFrac', fmt: v => `${Math.round(v * 100)}%`, geom: true },
  { id: 'windowBaffle', key: 'windowBaffle', fmt: v => v.toFixed(2), geom: true },
  { id: 'poissonIters', key: 'poissonIters', fmt: v => `${Math.round(v)}` },
];

function syncUIFromParams() {
  for (const b of sliderBindings) {
    const el = document.getElementById(b.id);
    const valEl = document.getElementById(`${b.id}Val`);
    if (el) el.value = solver.params[b.key];
    if (valEl) valEl.textContent = b.fmt(solver.params[b.key]);
  }
  document.getElementById('rearVentOpen').checked = solver.params.rearVentOpen;
  document.getElementById('emitterEnabled').checked = solver.params.emitterEnabled;
  document.getElementById('advectionOrder').value = String(solver.params.advectionOrder);
}

for (const b of sliderBindings) {
  const el = document.getElementById(b.id);
  const valEl = document.getElementById(`${b.id}Val`);
  if (!el) continue;
  el.addEventListener('input', () => {
    const val = parseFloat(el.value);
    solver.params[b.key] = val;
    if (b.key === 'windowSillFrac' && val >= solver.params.windowTopFrac - 0.06) {
      solver.params.windowTopFrac = Math.min(0.85, val + 0.06);
      syncUIFromParams();
    }
    if (b.key === 'windowTopFrac' && val <= solver.params.windowSillFrac + 0.06) {
      solver.params.windowSillFrac = Math.max(0.10, val - 0.06);
      syncUIFromParams();
    }
    if (valEl) valEl.textContent = b.fmt(solver.params[b.key]);
    if (b.geom) solver.rebuildGeometry();
  });
}

document.getElementById('rearVentOpen').addEventListener('change', (e) => {
  solver.params.rearVentOpen = e.target.checked;
  solver.rebuildGeometry();
});

document.getElementById('emitterEnabled').addEventListener('change', (e) => {
  solver.params.emitterEnabled = e.target.checked;
});

document.getElementById('advectionOrder').addEventListener('change', (e) => {
  solver.params.advectionOrder = parseInt(e.target.value, 10);
});

document.getElementById('showVectors').addEventListener('change', (e) => {
  renderer.showVectors = e.target.checked;
});

// View mode tabs
document.querySelectorAll('[data-view]').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderer.viewMode = btn.dataset.view;
  });
});

// Canvas interaction tool tabs
document.querySelectorAll('[data-tool]').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('[data-tool]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    toolMode = btn.dataset.tool;
  });
});

// Play / Pause / Step / Reset / Clear walls
const playBtn = document.getElementById('playPauseBtn');
playBtn.addEventListener('click', () => {
  running = !running;
  playBtn.textContent = running ? 'Pause' : 'Play';
});

document.getElementById('stepBtn').addEventListener('click', () => {
  running = false;
  playBtn.textContent = 'Play';
  solver.step();
  renderer.render();
  updateDiagnostics();
});

document.getElementById('resetBtn').addEventListener('click', () => {
  solver.resetFields();
});

document.getElementById('clearWallsBtn').addEventListener('click', () => {
  solver.clearObstacles();
});

// Presets
document.getElementById('presetSelect').addEventListener('change', (e) => {
  const preset = PRESETS[e.target.value];
  if (!preset) return;
  Object.assign(solver.params, preset.params);
  solver.rebuildGeometry();
  solver.resetFields();
  syncUIFromParams();
});

// --- Pointer interaction on Canvas (drag emitter or paint/erase walls) ---
function handleCanvasPointer(e) {
  const rect = canvas.getBoundingClientRect();
  const normX = Math.max(0.02, Math.min(0.98, (e.clientX - rect.left) / rect.width));
  const normY = Math.max(0.02, Math.min(0.98, 1 - (e.clientY - rect.top) / rect.height));

  if (toolMode === 'emitter') {
    solver.params.emitterX = normX;
    solver.params.emitterY = normY;
  } else if (toolMode === 'wall') {
    solver.paintObstacle(normX, normY, 2, true);
  } else if (toolMode === 'erase') {
    solver.paintObstacle(normX, normY, 3, false);
  }
}

canvas.addEventListener('pointerdown', (e) => {
  isPointerDown = true;
  canvas.setPointerCapture(e.pointerId);
  handleCanvasPointer(e);
});
canvas.addEventListener('pointermove', (e) => {
  if (isPointerDown) handleCanvasPointer(e);
});
canvas.addEventListener('pointerup', () => {
  isPointerDown = false;
});

// --- Live Diagnostics ---
const diagTime = document.getElementById('diagTime');
const diagCfl = document.getElementById('diagCfl');
const diagDiv = document.getElementById('diagDiv');
const diagSmoke = document.getElementById('diagSmoke');

function updateDiagnostics() {
  const st = solver.stats;
  diagTime.textContent = `${st.time.toFixed(1)} s`;
  diagCfl.textContent = st.cfl.toFixed(2);
  diagDiv.textContent = st.maxDivAfter.toExponential(1);
  diagSmoke.textContent = `${st.smokeInside.toFixed(0)}% in / ${st.smokeOutside.toFixed(0)}% out`;
}

// --- Main Loop ---
function loop() {
  if (running) {
    for (let k = 0; k < stepsPerFrame; k++) {
      solver.step();
    }
  }
  renderer.render();
  updateDiagnostics();
  requestAnimationFrame(loop);
}

syncUIFromParams();
loop();
