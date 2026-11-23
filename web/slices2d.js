/* Quantitative 2D Cross-Section Slices (X-Z, X-Y, Y-Z), Colorbar, Probe & Sparkline */

(function (global) {
  "use strict";

  // --- Scientific Colormap LUT Builders (256 x RGBA) ---
  function buildLUT(stops) {
    const lut = new Uint8ClampedArray(256 * 4);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      let s0 = stops[0];
      let s1 = stops[stops.length - 1];
      for (let k = 0; k < stops.length - 1; k++) {
        if (t >= stops[k][0] && t <= stops[k + 1][0]) {
          s0 = stops[k];
          s1 = stops[k + 1];
          break;
        }
      }
      const f = s1[0] > s0[0] ? (t - s0[0]) / (s1[0] - s0[0]) : 0;
      lut[i * 4 + 0] = Math.round(s0[1] + f * (s1[1] - s0[1]));
      lut[i * 4 + 1] = Math.round(s0[2] + f * (s1[2] - s0[2]));
      lut[i * 4 + 2] = Math.round(s0[3] + f * (s1[3] - s0[3]));
      lut[i * 4 + 3] = 255;
    }
    return lut;
  }

  const COLORMAPS = {
    smoke: buildLUT([
      [0.00, 12, 17, 26],
      [0.12, 27, 41, 62],
      [0.32, 68, 96, 128],
      [0.60, 148, 176, 204],
      [0.85, 218, 231, 244],
      [1.00, 248, 251, 255],
    ]),
    temp: buildLUT([
      [0.00, 22, 62, 168],
      [0.25, 42, 152, 226],
      [0.50, 224, 232, 240],
      [0.75, 242, 148, 42],
      [1.00, 198, 30, 38],
    ]),
    speed: buildLUT([
      [0.00, 68, 1, 84],
      [0.25, 59, 82, 139],
      [0.50, 33, 145, 140],
      [0.75, 94, 201, 98],
      [1.00, 253, 231, 37],
    ]),
    pressure: buildLUT([
      [0.00, 30, 74, 158],
      [0.25, 92, 164, 222],
      [0.50, 232, 236, 242],
      [0.75, 234, 128, 96],
      [1.00, 178, 24, 43],
    ]),
    vorticity: buildLUT([
      [0.00, 11, 8, 28],
      [0.25, 86, 20, 110],
      [0.50, 182, 54, 121],
      [0.75, 241, 134, 45],
      [1.00, 252, 250, 168],
    ]),
    nu_sgs: buildLUT([
      [0.00, 13, 8, 135],
      [0.33, 126, 3, 168],
      [0.66, 218, 84, 108],
      [1.00, 240, 249, 33],
    ]),
  };

  const FIELD_META = {
    smoke:     { unit: "",      fmt: (v) => v.toFixed(2), fixed: [0, 0.85],  gamma: 0.72, symmetric: false },
    temp:      { unit: "°C",    fmt: (v) => v.toFixed(1), fixed: [0, 35],    gamma: 1.0,  symmetric: false },
    speed:     { unit: "m/s",   fmt: (v) => v.toFixed(2), fixed: [0, 1.5],   gamma: 1.0,  symmetric: false },
    pressure:  { unit: "Pa",    fmt: (v) => v.toFixed(2), fixed: [-2.0, 2.0],gamma: 1.0,  symmetric: true },
    vorticity: { unit: "1/s",   fmt: (v) => v.toFixed(1), fixed: [0, 15.0],  gamma: 0.8,  symmetric: false },
    nu_sgs:    { unit: "m²/s",  fmt: (v) => v.toExponential(1), fixed: [0, 1e-3], gamma: 0.75, symmetric: false },
  };

  // Exact intersection segment of 3D quad corners[0..3] with axis-aligned plane coord[axis] == val
  function intersectQuadWithPlane(corners, axis, val) {
    if (!corners || corners.length !== 4) return null;
    const pts = [];
    const eps = 1e-6;
    const edges = [[0, 1], [1, 2], [2, 3], [3, 0]];
    for (const [ia, ib] of edges) {
      const a = corners[ia];
      const b = corners[ib];
      const da = a[axis] - val;
      const db = b[axis] - val;
      if (Math.abs(da) <= eps && Math.abs(db) <= eps) {
        pts.push(a, b);
      } else if ((da <= 0 && db >= 0) || (da >= 0 && db <= 0)) {
        const denom = da - db;
        if (Math.abs(denom) > 1e-12) {
          const t = da / denom;
          pts.push([
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
          ]);
        }
      }
    }
    if (pts.length < 2) return null;
    // Find the two most distant points
    let bestD = -1;
    let pA = null, pB = null;
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const d = Math.hypot(
          pts[i][0] - pts[j][0],
          pts[i][1] - pts[j][1],
          pts[i][2] - pts[j][2]
        );
        if (d > bestD) {
          bestD = d;
          pA = pts[i];
          pB = pts[j];
        }
      }
    }
    return bestD > 1e-4 ? [pA, pB] : null;
  }

  class Slices2DManager {
    constructor(opts) {
      this.canvases = {
        xz: opts.canvasXZ,
        xy: opts.canvasXY,
        yz: opts.canvasYZ,
      };
      this.colorbarCanvas = opts.canvasColorbar;
      this.cbMinEl = opts.cbMinEl;
      this.cbMaxEl = opts.cbMaxEl;
      this.cbUnitEl = opts.cbUnitEl;
      this.probeEl = opts.probeEl;
      this.sparkCanvas = opts.canvasSpark;

      this.showVectors = true;
      this.autoScale = false;
      this.field = "smoke";
      this.onMoveEmitter = opts.onMoveEmitter || null;
      this.dragPlane = null;

      // Reusable offscreen canvases, 2D contexts & ImageData buffers per plane (zero per-frame alloc)
      this.offscreens = {
        xz: document.createElement("canvas"),
        xy: document.createElement("canvas"),
        yz: document.createElement("canvas"),
      };
      this.offCtxs = {
        xz: this.offscreens.xz.getContext("2d"),
        xy: this.offscreens.xy.getContext("2d"),
        yz: this.offscreens.yz.getContext("2d"),
      };
      this.offImgs = { xz: null, xy: null, yz: null };
      this._cbImg = null;
      this._lastColorbarField = null;

      // Last frame data & plot transforms for hover probing & click-to-place emitter
      this.lastFrame = null;
      this.plotRects = { xz: null, xy: null, yz: null };
      this.hoverState = { plane: null, mx: 0, my: 0 };

      this._initHover();
    }

    _emitFromPointer(plane, mx, my) {
      if (!this.onMoveEmitter || !this.lastFrame) return;
      const pr = this.plotRects[plane];
      if (!pr) return;
      const u = (mx - pr.ox) / pr.plotW;
      const v = 1.0 - (my - pr.oy) / pr.plotH;
      if (u < 0 || u > 1 || v < 0 || v > 1) return;

      const g = this.lastFrame.header.geom;
      const c1 = u * pr.len1;
      const c2 = v * pr.len2;
      const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
      const roomW = g.room_y1 - g.room_y0;

      let patch = null;
      if (plane === "xz") {
        patch = {
          smoke_x: Number(clamp(c1, 0.15, g.lx - 0.15).toFixed(2)),
          smoke_z: Number(clamp(c2, 0.15, g.lz - 0.15).toFixed(2)),
        };
      } else if (plane === "xy") {
        patch = {
          smoke_x: Number(clamp(c1, 0.15, g.lx - 0.15).toFixed(2)),
          smoke_y: Number(clamp(c2 - g.room_y0, 0.10, roomW - 0.10).toFixed(2)),
        };
      } else {
        patch = {
          smoke_y: Number(clamp(c1 - g.room_y0, 0.10, roomW - 0.10).toFixed(2)),
          smoke_z: Number(clamp(c2, 0.15, g.lz - 0.15).toFixed(2)),
        };
      }
      if (patch) this.onMoveEmitter(patch);
    }

    _initHover() {
      for (const plane of ["xz", "xy", "yz"]) {
        const cv = this.canvases[plane];
        cv.addEventListener("pointerdown", (e) => {
          if (e.button !== 0) return;
          const rect = cv.getBoundingClientRect();
          this.dragPlane = plane;
          cv.setPointerCapture(e.pointerId);
          this._emitFromPointer(plane, e.clientX - rect.left, e.clientY - rect.top);
        });
        cv.addEventListener("pointermove", (e) => {
          const rect = cv.getBoundingClientRect();
          const mx = e.clientX - rect.left;
          const my = e.clientY - rect.top;
          this.hoverState = { plane, mx, my };
          if (this.dragPlane === plane) {
            this._emitFromPointer(plane, mx, my);
          }
          this._updateProbeText();
        });
        cv.addEventListener("pointerup", () => {
          if (this.dragPlane === plane) this.dragPlane = null;
        });
        cv.addEventListener("pointercancel", () => {
          if (this.dragPlane === plane) this.dragPlane = null;
        });
        cv.addEventListener("mouseleave", () => {
          if (this.hoverState.plane === plane) {
            this.hoverState.plane = null;
            if (this.probeEl) {
              this.probeEl.textContent = "";
            }
          }
        });
      }
    }

    setOptions(opts) {
      if (opts.showVectors !== undefined) this.showVectors = opts.showVectors;
      if (opts.autoScale !== undefined) this.autoScale = opts.autoScale;
      if (opts.field !== undefined) this.field = opts.field;
      if (this.lastFrame) {
        this.render(this.lastFrame);
      }
    }

    _computeRange(frame) {
      const f = frame.header.field;
      const meta = FIELD_META[f] || FIELD_META.smoke;
      const p = frame.header.params;

      if (!this.autoScale) {
        if (f === "temp" && p) {
          // Fixed physical temperature scale: stays anchored at [0, 35] C unless inputs exceed it,
          // so raising indoor T always moves smoothly toward warmer red instead of shifting blue.
          const lo = Math.min(0.0, p.temp_in_c - 2.0, p.temp_out_c - 2.0);
          const hi = Math.max(35.0, p.temp_in_c + 2.0, p.temp_out_c + 2.0, p.smoke_temp_c + 2.0);
          return [lo, hi];
        }
        return meta.fixed.slice();
      }

      const mm = frame.header.slices_minmax.global;
      let vmin = mm[0];
      let vmax = mm[1];

      if (f === "smoke") {
        vmin = 0.0;
        vmax = Math.max(0.25, vmax);
      } else if (f === "speed") {
        vmin = 0.0;
        vmax = Math.max(0.25, vmax);
      } else if (f === "vorticity") {
        vmin = 0.0;
        vmax = Math.max(2.0, vmax);
      } else if (f === "nu_sgs") {
        vmin = 0.0;
        vmax = Math.max(1e-5, vmax);
      } else if (meta.symmetric) {
        const a = Math.max(0.15, Math.abs(vmin), Math.abs(vmax));
        vmin = -a;
        vmax = a;
      } else if (f === "temp") {
        if (vmax - vmin < 1.0) {
          const mid = 0.5 * (vmin + vmax);
          vmin = mid - 0.5;
          vmax = mid + 0.5;
        }
      }
      return [vmin, vmax];
    }

    _drawColorbar(field, vmin, vmax) {
      const cv = this.colorbarCanvas;
      if (!cv) return;
      if (this._lastColorbarField !== field) {
        const ctx = cv.getContext("2d");
        const w = cv.width;
        const h = cv.height;
        const lut = COLORMAPS[field] || COLORMAPS.smoke;
        if (!this._cbImg || this._cbImg.width !== w || this._cbImg.height !== h) {
          this._cbImg = ctx.createImageData(w, h);
        }
        const data = this._cbImg.data;
        for (let x = 0; x < w; x++) {
          const idx = Math.round((x / (w - 1)) * 255) * 4;
          const r = lut[idx + 0];
          const g = lut[idx + 1];
          const b = lut[idx + 2];
          for (let y = 0; y < h; y++) {
            const p = (y * w + x) * 4;
            data[p + 0] = r;
            data[p + 1] = g;
            data[p + 2] = b;
            data[p + 3] = 255;
          }
        }
        ctx.putImageData(this._cbImg, 0, 0);
        this._lastColorbarField = field;
      }

      const meta = FIELD_META[field] || FIELD_META.smoke;
      if (this.cbMinEl) this.cbMinEl.textContent = meta.fmt(vmin);
      if (this.cbMaxEl) this.cbMaxEl.textContent = meta.fmt(vmax);
      if (this.cbUnitEl) this.cbUnitEl.textContent = meta.unit;
    }

    render(frame) {
      this.lastFrame = frame;
      const hdr = frame.header;
      const field = hdr.field;
      const [vmin, vmax] = this._computeRange(frame);
      this._drawColorbar(field, vmin, vmax);

      this._renderPlane("xz", frame, vmin, vmax);
      this._renderPlane("xy", frame, vmin, vmax);
      this._renderPlane("yz", frame, vmin, vmax);
      if (this.sparkCanvas) this._renderSparkline(hdr.history || []);
      this._updateProbeText();
    }

    _renderPlane(plane, frame, vmin, vmax) {
      const cv = this.canvases[plane];
      if (!cv) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      const cssW = Math.max(1, cv.clientWidth);
      const cssH = Math.max(1, cv.clientHeight);
      const cw = Math.floor(cssW * dpr);
      const ch = Math.floor(cssH * dpr);
      if (cv.width !== cw || cv.height !== ch) {
        cv.width = cw;
        cv.height = ch;
      }

      const ctx = cv.getContext("2d");
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.fillStyle = "#0b1018";
      ctx.fillRect(0, 0, cssW, cssH);

      const hdr = frame.header;
      const g = hdr.geom;
      const p = hdr.params;
      const field = hdr.field;
      const meta = FIELD_META[field] || FIELD_META.smoke;
      const lut = COLORMAPS[field] || COLORMAPS.smoke;

      const vals = frame.chunks[plane + "_val"];
      const solid = frame.chunks[plane + "_solid"];
      if (!vals || !solid) {
        ctx.restore();
        return;
      }

      // Plane dimensions in cells & physical meters
      let n1, n2, len1, len2, label1, label2;
      if (plane === "xz") {
        n1 = g.nx; n2 = g.nz; len1 = g.lx; len2 = g.lz;
        label1 = "x (m)"; label2 = "z (m)";
      } else if (plane === "xy") {
        n1 = g.nx; n2 = g.ny; len1 = g.lx; len2 = g.ly;
        label1 = "x (m)"; label2 = "y (m)";
      } else {
        n1 = g.ny; n2 = g.nz; len1 = g.ly; len2 = g.lz;
        label1 = "y (m)"; label2 = "z (m)";
      }

      // Compute 1:1 physical aspect ratio plot rect inside margins
      const padL = 32, padR = 10, padT = 8, padB = 22;
      const availW = Math.max(20, cssW - padL - padR);
      const availH = Math.max(20, cssH - padT - padB);
      const scale = Math.min(availW / len1, availH / len2);
      const plotW = len1 * scale;
      const plotH = len2 * scale;
      const ox = padL + (availW - plotW) * 0.5;
      const oy = padT + (availH - plotH) * 0.5;

      this.plotRects[plane] = { ox, oy, plotW, plotH, len1, len2, n1, n2 };

      // Helper mapping physical (c1, c2) -> canvas pixel (px, py) with c2 increasing upward
      const toPx = (c1) => ox + (c1 / len1) * plotW;
      const toPy = (c2) => oy + plotH - (c2 / len2) * plotH;

      // 1. Render scalar raster into reusable offscreen ImageData (row 0 = top = n2 - 1)
      const off = this.offscreens[plane];
      const octx = this.offCtxs[plane];
      let img = this.offImgs[plane];
      if (off.width !== n1 || off.height !== n2 || !img || img.width !== n1 || img.height !== n2) {
        off.width = n1;
        off.height = n2;
        img = octx.createImageData(n1, n2);
        this.offImgs[plane] = img;
      }
      const imgData = img.data;
      const invSpan = vmax > vmin ? 1.0 / (vmax - vmin) : 0.0;
      const gamma = meta.gamma || 1.0;

      for (let r = 0; r < n2; r++) {
        const srcRow = (n2 - 1 - r) * n1;
        const dstRow = r * n1;
        for (let c = 0; c < n1; c++) {
          const sIdx = srcRow + c;
          const dIdx = (dstRow + c) * 4;
          if (solid[sIdx]) {
            // Architectural masonry solid shading with subtle checker hatch
            const hatch = ((r + c) & 1) ? 4 : 0;
            imgData[dIdx + 0] = 34 + hatch;
            imgData[dIdx + 1] = 43 + hatch;
            imgData[dIdx + 2] = 58 + hatch;
            imgData[dIdx + 3] = 255;
          } else {
            let t = (vals[sIdx] - vmin) * invSpan;
            if (t < 0) t = 0;
            else if (t > 1) t = 1;
            if (gamma !== 1.0) t = Math.pow(t, gamma);
            const li = Math.round(t * 255) * 4;
            imgData[dIdx + 0] = lut[li + 0];
            imgData[dIdx + 1] = lut[li + 1];
            imgData[dIdx + 2] = lut[li + 2];
            imgData[dIdx + 3] = 255;
          }
        }
      }
      octx.putImageData(img, 0, 0);

      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, ox, oy, plotW, plotH);

      // 2. Velocity Vector Overlay (strided cleanly over fluid cells)
      const u1 = frame.chunks[plane + "_u1"];
      const u2 = frame.chunks[plane + "_u2"];
      if (this.showVectors && u1 && u2) {
        const stride1 = Math.max(3, Math.round(n1 / 20));
        const stride2 = Math.max(3, Math.round(n2 / 14));
        const refSpeed = Math.max(0.35, Math.min(3.5, hdr.metrics.max_speed || 0.8));
        const maxArrowPx = Math.min(plotW / (n1 / stride1), plotH / (n2 / stride2)) * 0.85;

        ctx.strokeStyle = "rgba(235, 245, 255, 0.68)";
        ctx.lineWidth = 1.0;
        ctx.beginPath();

        for (let r = Math.floor(stride2 / 2); r < n2; r += stride2) {
          for (let c = Math.floor(stride1 / 2); c < n1; c += stride1) {
            const idx = r * n1 + c;
            if (solid[idx]) continue;
            const vx = u1[idx];
            const vy = u2[idx];
            const spd = Math.hypot(vx, vy);
            if (spd < 0.015) continue;

            const frac = Math.min(1.0, Math.sqrt(spd / refSpeed));
            const len = Math.max(2.5, frac * maxArrowPx);
            const ux = vx / spd;
            const uy = -vy / spd; // canvas Y points down

            const px = ox + ((c + 0.5) / n1) * plotW;
            const py = oy + plotH - ((r + 0.5) / n2) * plotH;
            const ex = px + ux * len;
            const ey = py + uy * len;

            ctx.moveTo(px, py);
            ctx.lineTo(ex, ey);
            // Arrowhead
            const hl = Math.min(4.2, len * 0.42);
            const ang = Math.atan2(uy, ux);
            ctx.moveTo(ex, ey);
            ctx.lineTo(ex - hl * Math.cos(ang - 0.52), ey - hl * Math.sin(ang - 0.52));
            ctx.moveTo(ex, ey);
            ctx.lineTo(ex - hl * Math.cos(ang + 0.52), ey - hl * Math.sin(ang + 0.52));
          }
        }
        ctx.stroke();
      }

      // 3. Synchronized Cross-Reference Slice Lines (where the other 2 planes cut this view)
      const ySlice = (hdr.xz_j + 0.5) * g.h;
      const zSlice = (hdr.xy_k + 0.5) * g.h;
      const xSlice = (hdr.yz_i + 0.5) * g.h;

      ctx.save();
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.0;
      if (plane === "xz") {
        // Horizontal XY plane line (amber)
        ctx.strokeStyle = "rgba(245, 158, 11, 0.55)";
        ctx.beginPath();
        ctx.moveTo(ox, toPy(zSlice));
        ctx.lineTo(ox + plotW, toPy(zSlice));
        ctx.stroke();
        // Vertical YZ plane line (violet)
        ctx.strokeStyle = "rgba(167, 139, 250, 0.55)";
        ctx.beginPath();
        ctx.moveTo(toPx(xSlice), oy);
        ctx.lineTo(toPx(xSlice), oy + plotH);
        ctx.stroke();
      } else if (plane === "xy") {
        // Horizontal XZ plane line (cyan)
        ctx.strokeStyle = "rgba(56, 189, 248, 0.55)";
        ctx.beginPath();
        ctx.moveTo(ox, toPy(ySlice));
        ctx.lineTo(ox + plotW, toPy(ySlice));
        ctx.stroke();
        // Vertical YZ plane line (violet)
        ctx.strokeStyle = "rgba(167, 139, 250, 0.55)";
        ctx.beginPath();
        ctx.moveTo(toPx(xSlice), oy);
        ctx.lineTo(toPx(xSlice), oy + plotH);
        ctx.stroke();
      } else {
        // Vertical XZ plane line (cyan)
        ctx.strokeStyle = "rgba(56, 189, 248, 0.55)";
        ctx.beginPath();
        ctx.moveTo(toPx(ySlice), oy);
        ctx.lineTo(toPx(ySlice), oy + plotH);
        ctx.stroke();
        // Horizontal XY plane line (amber)
        ctx.strokeStyle = "rgba(245, 158, 11, 0.55)";
        ctx.beginPath();
        ctx.moveTo(ox, toPy(zSlice));
        ctx.lineTo(ox + plotW, toPy(zSlice));
        ctx.stroke();
      }
      ctx.restore();

      // 4. Crisp Structural Overlays: Facade Reveal, Angled Casement Sash, Neutral Plane & Emitter
      // Facade inner/outer wall guide lines on XZ & XY
      if (plane === "xz" || plane === "xy") {
        ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
        ctx.lineWidth = 1;
        ctx.strokeRect(
          toPx(g.room_x0),
          plane === "xz" ? toPy(g.room_z1) : toPy(g.room_y1),
          toPx(g.room_x1) - toPx(g.room_x0),
          plane === "xz" ? (toPy(g.room_z0) - toPy(g.room_z1)) : (toPy(g.room_y0) - toPy(g.room_y1))
        );
      } else {
        // Window opening frame on YZ plane
        ctx.strokeStyle = "rgba(226, 232, 240, 0.85)";
        ctx.lineWidth = 1.4;
        ctx.strokeRect(
          toPx(g.win_y0),
          toPy(g.win_z1),
          toPx(g.win_y1) - toPx(g.win_y0),
          toPy(g.win_z0) - toPy(g.win_z1)
        );
        // Neutral Pressure Plane z_NPP across window opening
        const znpp = hdr.metrics.z_npp;
        if (znpp > g.win_z0 && znpp < g.win_z1) {
          ctx.save();
          ctx.strokeStyle = "#22c55e";
          ctx.lineWidth = 1.4;
          ctx.setLineDash([5, 3]);
          ctx.beginPath();
          ctx.moveTo(toPx(g.win_y0) - 6, toPy(znpp));
          ctx.lineTo(toPx(g.win_y1) + 6, toPy(znpp));
          ctx.stroke();
          ctx.restore();
        }
      }

      // Exact Analytic Angled UK Casement Sash Intersection!
      if (g.sash_corners && g.sash_corners.length === 4) {
        let seg = null;
        if (plane === "xz") {
          seg = intersectQuadWithPlane(g.sash_corners, 1, ySlice);
        } else if (plane === "xy") {
          seg = intersectQuadWithPlane(g.sash_corners, 2, zSlice);
        } else {
          seg = intersectQuadWithPlane(g.sash_corners, 0, xSlice);
        }

        if (seg) {
          const [a, b] = seg;
          const xA = plane === "yz" ? toPx(a[1]) : toPx(a[0]);
          const yA = plane === "xy" ? toPy(a[1]) : toPy(a[2]);
          const xB = plane === "yz" ? toPx(b[1]) : toPx(b[0]);
          const yB = plane === "xy" ? toPy(b[1]) : toPy(b[2]);

          ctx.save();
          ctx.strokeStyle = "#38bdf8";
          ctx.lineWidth = 3.0;
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(xA, yA);
          ctx.lineTo(xB, yB);
          ctx.stroke();
          ctx.restore();
        }
      }

      // Smoke Emitter Marker (crisp crosshair + ring; click/drag slice to move)
      if (p && p.smoke_enabled) {
        const ex = p.smoke_x;
        const ey = g.room_y0 + p.smoke_y;
        const ez = p.smoke_z;
        let pxE, pyE, dist;
        if (plane === "xz") { pxE = toPx(ex); pyE = toPy(ez); dist = Math.abs(ey - ySlice); }
        else if (plane === "xy") { pxE = toPx(ex); pyE = toPy(ey); dist = Math.abs(ez - zSlice); }
        else { pxE = toPx(ey); pyE = toPy(ez); dist = Math.abs(ex - xSlice); }

        ctx.save();
        ctx.strokeStyle = dist < 0.45 ? "#e5e7eb" : "rgba(229, 231, 235, 0.50)";
        ctx.lineWidth = dist < 0.45 ? 1.4 : 1.0;
        ctx.beginPath();
        ctx.arc(pxE, pyE, 5.0, 0, Math.PI * 2);
        ctx.moveTo(pxE - 8, pyE); ctx.lineTo(pxE - 3, pyE);
        ctx.moveTo(pxE + 3, pyE); ctx.lineTo(pxE + 8, pyE);
        ctx.moveTo(pxE, pyE - 8); ctx.lineTo(pxE, pyE - 3);
        ctx.moveTo(pxE, pyE + 3); ctx.lineTo(pxE, pyE + 8);
        ctx.stroke();
        ctx.restore();
      }

      // 5. Plot Border & Physical Meter Axis Ticks
      ctx.strokeStyle = "#2d3d57";
      ctx.lineWidth = 1;
      ctx.strokeRect(ox, oy, plotW, plotH);

      ctx.fillStyle = "#8a9bb3";
      ctx.font = '9.5px "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace';

      // Horizontal ticks (every 1.0 m)
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      const step1 = len1 > 5.0 ? 1.5 : 1.0;
      for (let v = 0; v <= len1 + 0.01; v += step1) {
        const tx = toPx(Math.min(v, len1));
        ctx.beginPath();
        ctx.moveTo(tx, oy + plotH);
        ctx.lineTo(tx, oy + plotH + 3);
        ctx.stroke();
        ctx.fillText(v.toFixed(v % 1 === 0 ? 0 : 1), tx, oy + plotH + 5);
      }

      // Vertical ticks
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      const step2 = len2 > 4.5 ? 1.5 : 1.0;
      for (let v = 0; v <= len2 + 0.01; v += step2) {
        const ty = toPy(Math.min(v, len2));
        ctx.beginPath();
        ctx.moveTo(ox - 3, ty);
        ctx.lineTo(ox, ty);
        ctx.stroke();
        ctx.fillText(v.toFixed(v % 1 === 0 ? 0 : 1), ox - 5, ty);
      }

      // Small axis name badges
      ctx.fillStyle = "#5c6e87";
      ctx.textAlign = "right";
      ctx.textBaseline = "bottom";
      ctx.fillText(label1, ox + plotW - 2, oy + plotH - 2);
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(label2, ox + 4, oy + 3);

      ctx.restore();
    }

    _updateProbeText() {
      if (!this.probeEl || !this.lastFrame || !this.hoverState.plane) return;
      const plane = this.hoverState.plane;
      const pr = this.plotRects[plane];
      if (!pr) return;

      const u = (this.hoverState.mx - pr.ox) / pr.plotW;
      const v = 1.0 - (this.hoverState.my - pr.oy) / pr.plotH;
      if (u < 0 || u > 1 || v < 0 || v > 1) return;

      const hdr = this.lastFrame.header;
      const g = hdr.geom;
      const c = Math.max(0, Math.min(pr.n1 - 1, Math.floor(u * pr.n1)));
      const r = Math.max(0, Math.min(pr.n2 - 1, Math.floor(v * pr.n2)));
      const idx = r * pr.n1 + c;

      const solid = this.lastFrame.chunks[plane + "_solid"];
      const vals = this.lastFrame.chunks[plane + "_val"];
      const u1 = this.lastFrame.chunks[plane + "_u1"];
      const u2 = this.lastFrame.chunks[plane + "_u2"];

      let wx, wy, wz;
      if (plane === "xz") {
        wx = (c + 0.5) * g.h; wy = (hdr.xz_j + 0.5) * g.h; wz = (r + 0.5) * g.h;
      } else if (plane === "xy") {
        wx = (c + 0.5) * g.h; wy = (r + 0.5) * g.h; wz = (hdr.xy_k + 0.5) * g.h;
      } else {
        wx = (hdr.yz_i + 0.5) * g.h; wy = (c + 0.5) * g.h; wz = (r + 0.5) * g.h;
      }

      if (solid && solid[idx]) {
        this.probeEl.textContent = `(${wx.toFixed(2)}, ${wy.toFixed(2)}, ${wz.toFixed(2)}) m · SOLID`;
        return;
      }

      const meta = FIELD_META[hdr.field] || FIELD_META.smoke;
      const valStr = vals ? `${meta.fmt(vals[idx])} ${meta.unit}`.trim() : "—";
      let vecStr = "";
      if (u1 && u2) {
        const compNames = plane === "xz" ? ["u", "w"] : (plane === "xy" ? ["u", "v"] : ["v", "w"]);
        vecStr = ` · ${compNames[0]}=${u1[idx].toFixed(2)}, ${compNames[1]}=${u2[idx].toFixed(2)} m/s`;
      }
      this.probeEl.textContent = `(${wx.toFixed(2)}, ${wy.toFixed(2)}, ${wz.toFixed(2)}) m · ${hdr.field}=${valStr}${vecStr}`;
    }

    _renderSparkline(history) {
      const cv = this.sparkCanvas;
      if (!cv) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      const cssW = Math.max(1, cv.clientWidth);
      const cssH = Math.max(1, cv.clientHeight);
      const cw = Math.floor(cssW * dpr);
      const ch = Math.floor(cssH * dpr);
      if (cv.width !== cw || cv.height !== ch) {
        cv.width = cw;
        cv.height = ch;
      }

      const ctx = cv.getContext("2d");
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, cssW, cssH);

      // Subtle horizontal guide lines
      ctx.strokeStyle = "rgba(51, 70, 99, 0.35)";
      ctx.lineWidth = 1;
      for (const frac of [0.25, 0.5, 0.75]) {
        const y = Math.round(cssH * frac) + 0.5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(cssW, y);
        ctx.stroke();
      }

      if (!history || history.length < 2) {
        ctx.restore();
        return;
      }

      let maxQ = 50;
      for (const pt of history) {
        if (pt.q_out_ls > maxQ) maxQ = pt.q_out_ls;
        if (pt.q_in_ls > maxQ) maxQ = pt.q_in_ls;
      }
      maxQ *= 1.1;

      const n = history.length;
      const padY = 3;
      const h = Math.max(4, cssH - padY * 2);

      const drawSeries = (getter, maxVal, color, lineWidth) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const x = (i / (n - 1)) * cssW;
          const norm = Math.max(0, Math.min(1, getter(history[i]) / maxVal));
          const y = padY + (1 - norm) * h;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      };

      drawSeries((d) => d.smoke_frac, 100.0, "rgba(203, 213, 225, 0.55)", 1.1);
      drawSeries((d) => d.q_in_ls, maxQ, "#f59e0b", 1.35);
      drawSeries((d) => d.q_out_ls, maxQ, "#38bdf8", 1.5);

      ctx.restore();
    }
  }

  global.Slices2DManager = Slices2DManager;
})(window);
