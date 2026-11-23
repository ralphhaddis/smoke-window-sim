/* Bare-Bones Scientific UI Controller: Binary Frame Streamer & Controls */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const textDecoder = new TextDecoder("utf-8");

  function decodeBinaryFrame(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    const headerLen = view.getUint32(0, true);
    const headerBytes = new Uint8Array(arrayBuffer, 4, headerLen);
    const header = JSON.parse(textDecoder.decode(headerBytes));

    const payloadStart = 4 + headerLen;
    const chunks = {};
    for (const [name, info] of Object.entries(header.chunks || {})) {
      const byteOff = payloadStart + info.offset;
      if (info.dtype === "f32") {
        chunks[name] = new Float32Array(arrayBuffer, byteOff, info.bytes / 4);
      } else {
        chunks[name] = new Uint8Array(arrayBuffer, byteOff, info.bytes);
      }
    }
    return { header, chunks };
  }

  const state = {
    running: true,
    timeScale: 1.0,
    grid: "fast",
    preset: "cold_day",
    sliceField: "smoke",
    volField: "smoke",
    vectors: true,
    xz_j: null,
    xy_k: null,
    yz_i: null,
    geom: null,
    params: null,
    inFlight: false,
    dirtyFrame: true,
    pollTimer: null,
    lastStep: null,
    lastStateVer: null,
    lastFrameKey: null,
    paramTimer: null,
    pendingParams: {},
  };

  const SLIDER_SPECS = [
    { id: "inp-angle",    out: "out-angle",    key: "win_angle_deg",      fmt: (v) => `${Math.round(v)}°` },
    { id: "inp-win-w",    out: "out-win-w",    key: "win_width",          fmt: (v) => `${v.toFixed(2)}m` },
    { id: "inp-win-h",    out: "out-win-h",    key: "win_height",         fmt: (v) => `${v.toFixed(2)}m` },
    { id: "inp-win-sill", out: "out-win-sill", key: "win_sill",           fmt: (v) => `${v.toFixed(2)}m` },
    { id: "inp-tin",      out: "out-tin",      key: "temp_in_c",          fmt: (v) => `${v.toFixed(1)}°C` },
    { id: "inp-tout",     out: "out-tout",     key: "temp_out_c",         fmt: (v) => `${v.toFixed(1)}°C` },
    { id: "inp-tsmoke",   out: "out-tsmoke",   key: "smoke_temp_c",       fmt: (v) => `${v.toFixed(1)}°C` },
    { id: "inp-alpha",    out: "out-alpha",    key: "smoke_alpha_s",      fmt: (v) => `${v.toFixed(3)}` },
    { id: "inp-sx",       out: "out-sx",       key: "smoke_x",            fmt: (v) => `${v.toFixed(2)}m` },
    { id: "inp-sy",       out: "out-sy",       key: "smoke_y",            fmt: (v) => `${v.toFixed(2)}m` },
    { id: "inp-sz",       out: "out-sz",       key: "smoke_z",            fmt: (v) => `${v.toFixed(2)}m` },
    { id: "inp-srate",    out: "out-srate",    key: "smoke_rate",         fmt: (v) => `${v.toFixed(3)}` },
    { id: "inp-sjet",     out: "out-sjet",     key: "smoke_jet_speed",    fmt: (v) => `${v.toFixed(2)}m/s` },
    { id: "inp-sazim",    out: "out-sazim",    key: "smoke_jet_azim_deg", fmt: (v) => `${Math.round(v)}°` },
    { id: "inp-selev",    out: "out-selev",    key: "smoke_jet_elev_deg", fmt: (v) => `${Math.round(v)}°` },
    { id: "inp-dp",       out: "out-dp",       key: "delta_p_pa",         fmt: (v) => `${v.toFixed(1)}Pa` },
    { id: "inp-wspd",     out: "out-wspd",     key: "wind_speed",         fmt: (v) => `${v.toFixed(1)}m/s` },
    { id: "inp-wdir",     out: "out-wdir",     key: "wind_dir_deg",       fmt: (v) => `${Math.round(v)}°` },
  ];

  const CHECK_SPECS = [
    { id: "inp-vent",     key: "vent_enabled" },
    { id: "inp-smoke-on", key: "smoke_enabled" },
    { id: "inp-wabl",     key: "wind_use_abl" },
  ];

  function applyEmitterPatchUI(patch) {
    for (const s of SLIDER_SPECS) {
      if (patch[s.key] !== undefined) {
        $(s.id).value = String(patch[s.key]);
        $(s.out).textContent = s.fmt(patch[s.key]);
      }
    }
    if (patch.smoke_enabled !== undefined) {
      $("inp-smoke-on").checked = Boolean(patch.smoke_enabled);
    }
    queueParamUpdate(patch);
  }

  const viewer3d = new window.Viewer3D($("canvas-3d"), $("webgl-fallback"));
  const slices2d = new window.Slices2DManager({
    canvasXZ: $("canvas-xz"),
    canvasXY: $("canvas-xy"),
    canvasYZ: $("canvas-yz"),
    canvasColorbar: $("canvas-colorbar"),
    cbMinEl: $("cb-min"),
    cbMaxEl: $("cb-max"),
    cbUnitEl: $("cb-unit"),
    probeEl: $("probe-readout"),
    canvasSpark: null,
    onMoveEmitter: (patch) => {
      patch.smoke_enabled = 1;
      applyEmitterPatchUI(patch);
    },
  });

  async function postJSON(url, body) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`POST ${url} failed (${resp.status}): ${txt}`);
    }
    return resp.json();
  }

  function queueParamUpdate(patch) {
    Object.assign(state.pendingParams, patch);
    if (state.paramTimer) clearTimeout(state.paramTimer);
    state.paramTimer = setTimeout(async () => {
      const payload = { ...state.pendingParams };
      state.pendingParams = {};
      state.paramTimer = null;
      try {
        const res = await postJSON("/api/params", payload);
        syncStateFromServer(res, false);
        requestFrame();
      } catch (err) {
        console.error(err);
      }
    }, 30);
  }

  async function sendControl(payload) {
    try {
      const res = await postJSON("/api/control", payload);
      const resetSlices = payload.action === "grid" || payload.action === "preset";
      const updateInputs = payload.action === "preset" || payload.action === "grid";
      syncStateFromServer(res, updateInputs, resetSlices);
      requestFrame();
    } catch (err) {
      console.error(err);
    }
  }

  function syncSliceSliders(geom, resetToDefault) {
    if (!geom) return;
    const rXZ = $("rng-xz-j");
    const rXY = $("rng-xy-k");
    const rYZ = $("rng-yz-i");

    rXZ.max = String(geom.ny - 1);
    rXY.max = String(geom.nz - 1);
    rYZ.max = String(geom.nx - 1);

    if (resetToDefault || state.xz_j === null || state.xz_j >= geom.ny) {
      state.xz_j = Math.floor((geom.j_w0 + geom.j_w1) / 2);
    }
    if (resetToDefault || state.xy_k === null || state.xy_k >= geom.nz) {
      state.xy_k = Math.floor((geom.k_w0 + geom.k_w1) / 2);
    }
    if (resetToDefault || state.yz_i === null || state.yz_i >= geom.nx) {
      state.yz_i = geom.ix_wall;
    }

    rXZ.value = String(state.xz_j);
    rXY.value = String(state.xy_k);
    rYZ.value = String(state.yz_i);
    updateSlicePosLabels(geom);
  }

  function updateSlicePosLabels(geom) {
    if (!geom) return;
    $("lbl-xz-pos").textContent = `y=${((state.xz_j + 0.5) * geom.h).toFixed(2)}m`;
    $("lbl-xy-pos").textContent = `z=${((state.xy_k + 0.5) * geom.h).toFixed(2)}m`;
    $("lbl-yz-pos").textContent = `x=${((state.yz_i + 0.5) * geom.h).toFixed(2)}m`;
  }

  function syncControlsFromParams(p, geom) {
    if (!p) return;
    $("inp-hinge").value = String(p.win_hinge_mode);

    if (geom) {
      $("inp-sx").max = Math.max(4.5, geom.lx - 0.15).toFixed(2);
      $("inp-sy").max = Math.max(2.5, (geom.room_y1 - geom.room_y0) - 0.1).toFixed(2);
      $("inp-sz").max = Math.max(2.3, geom.lz - 0.15).toFixed(2);
    }

    for (const s of SLIDER_SPECS) {
      const val = Number(p[s.key]);
      if (!Number.isFinite(val)) continue;
      $(s.id).value = String(val);
      $(s.out).textContent = s.fmt(val);
    }
    for (const c of CHECK_SPECS) {
      $(c.id).checked = Boolean(p[c.key]);
    }
  }

  function updateStatusUI(running, timeScale, achievedRt, grid, preset, m) {
    state.running = running;
    state.grid = grid;
    state.preset = preset;
    if (timeScale !== undefined) {
      state.timeScale = timeScale;
      const selSpeed = $("sel-speed");
      const targetStr = timeScale === 0 ? "0" : String(timeScale);
      if ([...selSpeed.options].some((o) => o.value === targetStr)) {
        selSpeed.value = targetStr;
      }
    }

    $("btn-play").textContent = running ? "Pause" : "Play";
    $("sel-grid").value = grid;
    $("sel-preset").value = preset;
    if (achievedRt !== undefined) {
      $("top-rt").textContent = running ? `${Number(achievedRt).toFixed(1)}×` : "paused";
    }

    if (m) {
      $("top-time").textContent = m.sim_time.toFixed(2);
      $("top-step-ms").textContent = m.step_ms.toFixed(1);
      $("m-div").textContent = m.max_div.toExponential(1);
      $("m-qout").textContent = Math.round(m.q_out_ls).toString();
      $("m-qin").textContent = Math.round(m.q_in_ls).toString();
      $("m-znpp").textContent = m.z_npp > 0 ? `${m.z_npp.toFixed(2)}m` : "—";
      $("m-maxu").textContent = m.max_speed.toFixed(2);
      $("m-tin").textContent = m.temp_mean_in_c.toFixed(1);
    }
  }

  function syncStateFromServer(res, updateInputs, resetSlices) {
    state.geom = res.geom;
    state.params = res.params;
    syncSliceSliders(res.geom, Boolean(resetSlices));
    if (updateInputs) {
      syncControlsFromParams(res.params, res.geom);
    }
    updateStatusUI(
      res.running,
      res.time_scale,
      res.achieved_rt,
      res.grid,
      res.preset,
      res.metrics
    );
  }

  // --- UI Event Bindings ---
  $("btn-play").addEventListener("click", () => {
    sendControl({ action: state.running ? "pause" : "play" });
  });
  $("btn-step").addEventListener("click", () => {
    sendControl({ action: "step", n: 1 });
  });
  $("btn-reset").addEventListener("click", () => {
    sendControl({ action: "reset" });
  });
  $("sel-speed").addEventListener("change", (e) => {
    sendControl({ action: "speed", scale: parseFloat(e.target.value) });
  });
  $("sel-preset").addEventListener("change", (e) => {
    sendControl({ action: "preset", preset: e.target.value });
  });
  $("sel-grid").addEventListener("change", (e) => {
    sendControl({ action: "grid", grid: e.target.value });
  });

  $("sel-vol-field").addEventListener("change", (e) => {
    state.volField = e.target.value;
    viewer3d.setOptions({ volField: state.volField });
    requestFrame();
  });
  $("rng-vol-density").addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    $("out-gain").textContent = v.toFixed(2);
    viewer3d.setOptions({ densityGain: v });
  });
  $("chk-cutplanes").addEventListener("change", (e) => {
    viewer3d.setOptions({ showCutPlanes: e.target.checked });
  });
  $("chk-masonry").addEventListener("change", (e) => {
    viewer3d.setOptions({ showMasonry: e.target.checked });
  });

  document.querySelectorAll("[data-cam]").forEach((btn) => {
    btn.addEventListener("click", () => viewer3d.setCameraPreset(btn.dataset.cam));
  });

  $("sel-slice-field").addEventListener("change", (e) => {
    state.sliceField = e.target.value;
    slices2d.setOptions({ field: state.sliceField });
    requestFrame();
  });
  $("chk-vectors").addEventListener("change", (e) => {
    state.vectors = e.target.checked;
    slices2d.setOptions({ showVectors: state.vectors });
    requestFrame();
  });
  $("chk-autoscale").addEventListener("change", (e) => {
    slices2d.setOptions({ autoScale: e.target.checked });
  });

  $("rng-xz-j").addEventListener("input", (e) => {
    state.xz_j = parseInt(e.target.value, 10);
    updateSlicePosLabels(state.geom);
    requestFrame();
  });
  $("rng-xy-k").addEventListener("input", (e) => {
    state.xy_k = parseInt(e.target.value, 10);
    updateSlicePosLabels(state.geom);
    requestFrame();
  });
  $("rng-yz-i").addEventListener("input", (e) => {
    state.yz_i = parseInt(e.target.value, 10);
    updateSlicePosLabels(state.geom);
    requestFrame();
  });

  $("inp-hinge").addEventListener("change", (e) => {
    queueParamUpdate({ win_hinge_mode: parseInt(e.target.value, 10) });
  });

  for (const s of SLIDER_SPECS) {
    $(s.id).addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      $(s.out).textContent = s.fmt(v);
      queueParamUpdate({ [s.key]: v });
    });
  }

  for (const c of CHECK_SPECS) {
    $(c.id).addEventListener("change", (e) => {
      queueParamUpdate({ [c.key]: e.target.checked ? 1 : 0 });
    });
  }

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT") {
      e.preventDefault();
      sendControl({ action: state.running ? "pause" : "play" });
    }
  });

  window.addEventListener("resize", () => {
    if (document.hidden) return;
    viewer3d.render();
    if (slices2d.lastFrame) slices2d.render(slices2d.lastFrame);
  });

  function clearPollTimer() {
    if (state.pollTimer !== null) {
      clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function scheduleNextPoll(delayMs = 55) {
    clearPollTimer();
    if (document.hidden || state.inFlight) return;
    if (state.dirtyFrame) {
      state.pollTimer = setTimeout(() => {
        state.pollTimer = null;
        fetchFrameOnce();
      }, 0);
      return;
    }
    if (state.running) {
      state.pollTimer = setTimeout(() => {
        state.pollTimer = null;
        fetchFrameOnce();
      }, delayMs);
    }
  }

  function requestFrame() {
    state.dirtyFrame = true;
    if (document.hidden || state.inFlight) return;
    clearPollTimer();
    fetchFrameOnce();
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      clearPollTimer();
    } else if (state.running || state.dirtyFrame) {
      requestFrame();
    }
  });

  async function fetchFrameOnce() {
    if (state.inFlight || document.hidden) return;
    state.inFlight = true;
    state.dirtyFrame = false;
    let nextDelay = 55;

    try {
      const qs = new URLSearchParams({
        field: state.sliceField,
        vol_field: state.volField,
        vectors: state.vectors ? "1" : "0",
        vol: "1",
      });
      if (state.xz_j !== null) qs.set("xz_j", String(state.xz_j));
      if (state.xy_k !== null) qs.set("xy_k", String(state.xy_k));
      if (state.yz_i !== null) qs.set("yz_i", String(state.yz_i));
      if (state.lastStep !== null && state.lastStateVer !== null) {
        qs.set("last_step", String(state.lastStep));
        qs.set("last_ver", String(state.lastStateVer));
        if (state.lastFrameKey !== null) {
          qs.set("last_key", state.lastFrameKey);
        }
      }

      const resp = await fetch(`/api/frame?${qs.toString()}`);
      if (resp.status === 204) {
        return;
      }
      if (!resp.ok) throw new Error(`Frame HTTP ${resp.status}`);
      const buf = await resp.arrayBuffer();
      if (document.hidden) {
        state.dirtyFrame = true;
        return;
      }
      const frame = decodeBinaryFrame(buf);
      const hdr = frame.header;

      state.lastStep = hdr.metrics ? hdr.metrics.step_count : null;
      state.lastStateVer = hdr.state_ver !== undefined ? hdr.state_ver : null;
      state.lastFrameKey = hdr.frame_key || null;

      state.geom = hdr.geom;
      state.params = hdr.params;
      state.xz_j = hdr.xz_j;
      state.xy_k = hdr.xy_k;
      state.yz_i = hdr.yz_i;
      updateSlicePosLabels(hdr.geom);
      updateStatusUI(
        hdr.running,
        hdr.time_scale,
        hdr.achieved_rt,
        hdr.grid,
        hdr.preset,
        hdr.metrics
      );

      const volInfo = hdr.chunks.volume;
      viewer3d.updateScene(
        hdr.geom,
        hdr.params,
        { xz_j: hdr.xz_j, xy_k: hdr.xy_k, yz_i: hdr.yz_i },
        frame.chunks.volume || null,
        volInfo ? volInfo.shape : null,
        hdr.vol_vmin,
        hdr.vol_vmax
      );
      slices2d.render(frame);
    } catch (err) {
      console.error("Frame fetch error:", err);
      nextDelay = 250;
    } finally {
      state.inFlight = false;
      scheduleNextPoll(nextDelay);
    }
  }

  async function init() {
    try {
      const resp = await fetch("/api/state");
      const initState = await resp.json();
      syncStateFromServer(initState, true, true);
      if (!document.hidden) {
        requestFrame();
      }
    } catch (err) {
      console.error("Failed to connect to 3D CFD server:", err);
    }
  }

  init();
})();
