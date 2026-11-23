import { CELL_SOLID, CELL_OPEN_OUTDOOR, CELL_OPEN_VENT } from './solver.js';

/**
 * Renders the MAC-grid fields onto an HTML5 Canvas with scientific colormaps,
 * optional velocity vectors, room/window annotations, and draggable emitter handle.
 */
export class FluidRenderer {
  constructor(canvas, solver) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.solver = solver;

    // Offscreen pixel buffer matching grid resolution (scaled smoothly onto main canvas)
    this.offscreen = document.createElement('canvas');
    this.offscreen.width = solver.nx;
    this.offscreen.height = solver.ny;
    this.offCtx = this.offscreen.getContext('2d');
    this.imgData = this.offCtx.createImageData(solver.nx, solver.ny);

    this.viewMode = 'smoke'; // 'smoke' | 'temperature' | 'velocity' | 'vorticity' | 'pressure'
    this.showVectors = false;
    this.showLabels = true;
  }

  resizeGrid(nx, ny) {
    this.offscreen.width = nx;
    this.offscreen.height = ny;
    this.imgData = this.offCtx.createImageData(nx, ny);
  }

  render() {
    const { solver, ctx, canvas, offscreen, offCtx, imgData } = this;
    const { nx, ny, cellType, s, T, p, curl, u, v, params } = solver;
    const data = imgData.data;

    const tMin = Math.min(params.tempIndoor, params.tempOutdoor, params.tempSmoke) - 2;
    const tMax = Math.max(params.tempIndoor, params.tempOutdoor, params.tempSmoke) + 2;
    const tSpan = Math.max(4, tMax - tMin);

    // Fill pixel buffer (note: physics j=0 is bottom, canvas py=0 is top -> flip y)
    for (let j = 0; j < ny; j++) {
      const py = ny - 1 - j;
      const rowOffset = py * nx;
      for (let i = 0; i < nx; i++) {
        const c = i + j * nx;
        const pIdx = (rowOffset + i) * 4;
        const type = cellType[c];

        if (type === CELL_SOLID) {
          // Architectural dark slate wall with subtle hatch
          const hatch = ((i + j) & 3) === 0 ? 10 : 0;
          data[pIdx] = 44 + hatch;
          data[pIdx + 1] = 52 + hatch;
          data[pIdx + 2] = 66 + hatch;
          data[pIdx + 3] = 255;
          continue;
        }

        let r = 14, g = 18, b = 26;

        if (this.viewMode === 'smoke') {
          // Subtle indoor/outdoor ambient tint + bright wispy smoke
          const isInside = i < solver.geom.wallI;
          const baseR = isInside ? 20 : 12;
          const baseG = isInside ? 24 : 20;
          const baseB = isInside ? 34 : 36;

          const sm = Math.max(0, Math.min(1.2, s[c]));
          const alpha = 1 - Math.exp(-2.6 * sm);
          // Warm smoke gets slightly amber/white, cold smoke gets cool blue-white
          const tempWarmth = Math.max(0, Math.min(1, (T[c] - params.tempOutdoor) / 35));
          const smR = 215 + 35 * tempWarmth;
          const smG = 222 + 12 * tempWarmth;
          const smB = 238 - 22 * tempWarmth;

          r = baseR + (smR - baseR) * alpha;
          g = baseG + (smG - baseG) * alpha;
          b = baseB + (smB - baseB) * alpha;
        } else if (this.viewMode === 'temperature') {
          const norm = Math.max(0, Math.min(1, (T[c] - tMin) / tSpan));
          [r, g, b] = this.thermalColor(norm);
          // Faint smoke contour overlay
          const sm = Math.min(1, s[c]) * 0.25;
          r = r * (1 - sm) + 255 * sm;
          g = g * (1 - sm) + 255 * sm;
          b = b * (1 - sm) + 255 * sm;
        } else if (this.viewMode === 'velocity') {
          const uC = 0.5 * (u[solver.idxU(i, j)] + u[solver.idxU(i + 1, j)]);
          const vC = 0.5 * (v[solver.idxV(i, j)] + v[solver.idxV(i, j + 1)]);
          const spd = Math.hypot(uC, vC);
          const norm = Math.min(1, spd / 1.8);
          [r, g, b] = this.viridisColor(norm);
        } else if (this.viewMode === 'vorticity') {
          const w = Math.max(-1, Math.min(1, curl[c] / 12.0));
          [r, g, b] = this.divergingColor(w);
        } else if (this.viewMode === 'pressure') {
          const pPhys = (p[c] * 1.2) / Math.max(1e-4, params.dt); // Pascals
          const norm = Math.max(-1, Math.min(1, pPhys / 2.5));
          [r, g, b] = this.divergingColor(norm);
        }

        // Highlight open vent slightly
        if (type === CELL_OPEN_VENT) {
          r = Math.min(255, r + 18);
          g = Math.min(255, g + 35);
          b = Math.min(255, b + 20);
        }

        data[pIdx] = r;
        data[pIdx + 1] = g;
        data[pIdx + 2] = b;
        data[pIdx + 3] = 255;
      }
    }

    offCtx.putImageData(imgData, 0, 0);

    // Draw scaled field onto main canvas
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);

    // Draw window frame highlight
    this.drawWindowFrame();

    // Optional velocity vectors
    if (this.showVectors) {
      this.drawVelocityVectors();
    }

    // Emitter handle
    this.drawEmitterHandle();

    // Room / Outdoor labels
    if (this.showLabels) {
      this.drawAnnotations();
    }
  }

  drawWindowFrame() {
    const { ctx, canvas, solver } = this;
    const { nx, ny, geom } = solver;
    const sx = canvas.width / nx;
    const sy = canvas.height / ny;

    const x = geom.wallI * sx;
    const w = 2 * sx;
    const yTop = (ny - geom.topJ) * sy;
    const yBot = (ny - geom.sillJ) * sy;

    ctx.save();
    ctx.strokeStyle = 'rgba(148, 210, 255, 0.55)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(x, yTop, w, yBot - yTop);
    ctx.restore();
  }

  drawVelocityVectors() {
    const { ctx, canvas, solver } = this;
    const { nx, ny, u, v, cellType } = solver;
    const sx = canvas.width / nx;
    const sy = canvas.height / ny;
    const stride = Math.max(4, Math.round(nx / 28));
    const scale = 14;

    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.65)';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    ctx.lineWidth = 1.1;

    for (let j = stride; j < ny - 1; j += stride) {
      for (let i = stride; i < nx - 1; i += stride) {
        if (cellType[i + j * nx] === CELL_SOLID) continue;
        const uC = 0.5 * (u[solver.idxU(i, j)] + u[solver.idxU(i + 1, j)]);
        const vC = 0.5 * (v[solver.idxV(i, j)] + v[solver.idxV(i, j + 1)]);
        const spd = Math.hypot(uC, vC);
        if (spd < 0.03) continue;

        const px = (i + 0.5) * sx;
        const py = (ny - (j + 0.5)) * sy;
        const dx = Math.max(-22, Math.min(22, uC * scale));
        const dy = Math.max(-22, Math.min(22, -vC * scale)); // canvas y is inverted

        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + dx, py + dy);
        ctx.stroke();

        // Arrowhead
        const ang = Math.atan2(dy, dx);
        const head = 4;
        ctx.beginPath();
        ctx.moveTo(px + dx, py + dy);
        ctx.lineTo(px + dx - head * Math.cos(ang - 0.5), py + dy - head * Math.sin(ang - 0.5));
        ctx.lineTo(px + dx - head * Math.cos(ang + 0.5), py + dy - head * Math.sin(ang + 0.5));
        ctx.closePath();
        ctx.fill();
      }
    }
    ctx.restore();
  }

  drawEmitterHandle() {
    const { ctx, canvas, solver } = this;
    const { params } = solver;
    const x = params.emitterX * canvas.width;
    const y = (1 - params.emitterY) * canvas.height;
    const r = Math.max(7, params.emitterRadius * canvas.height);

    ctx.save();
    // Outer ring
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.strokeStyle = params.emitterEnabled ? '#fbbf24' : '#64748b';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Inner dot
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = params.emitterEnabled ? '#fef08a' : '#94a3b8';
    ctx.fill();

    // Jet direction arrow
    if (params.emitterEnabled && params.jetSpeed > 0.02) {
      const ang = (params.jetAngleDeg * Math.PI) / 180;
      const len = 12 + params.jetSpeed * 18;
      const ex = x + Math.cos(ang) * len;
      const ey = y - Math.sin(ang) * len;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(ex, ey);
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.restore();
  }

  drawAnnotations() {
    const { ctx, canvas, solver } = this;
    const { nx, ny, geom, params } = solver;
    const sx = canvas.width / nx;
    const sy = canvas.height / ny;

    ctx.save();
    ctx.font = '600 12px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillStyle = 'rgba(226, 232, 240, 0.72)';

    const roomCenterX = ((geom.leftI + geom.wallI) * 0.5) * sx;
    const roomTopY = (ny - geom.ceilJ) * sy + 20;
    ctx.textAlign = 'center';
    ctx.fillText(`INDOOR ROOM (${params.tempIndoor.toFixed(0)}°C)`, roomCenterX, roomTopY);

    const outCenterX = ((geom.wallI + nx) * 0.5) * sx;
    ctx.fillText(`OUTDOORS (${params.tempOutdoor.toFixed(0)}°C)`, outCenterX, 24);

    if (params.rearVentOpen) {
      const vy = (ny - 0.5 * (geom.ventBotJ + geom.ventTopJ)) * sy;
      ctx.textAlign = 'left';
      ctx.fillStyle = 'rgba(134, 239, 172, 0.85)';
      ctx.fillText('VENT', geom.leftI * sx + 6, vy);
    }
    ctx.restore();
  }

  // Scientific colormaps
  thermalColor(t) {
    // Deep blue (cold) -> cyan -> dark neutral -> orange -> bright yellow (hot)
    if (t < 0.25) {
      const u = t / 0.25;
      return [15 + 15 * u, 40 + 95 * u, 130 + 100 * u];
    } else if (t < 0.5) {
      const u = (t - 0.25) / 0.25;
      return [30 + 50 * u, 135 + 45 * u, 230 - 60 * u];
    } else if (t < 0.75) {
      const u = (t - 0.5) / 0.25;
      return [80 + 155 * u, 180 - 50 * u, 170 - 130 * u];
    } else {
      const u = (t - 0.75) / 0.25;
      return [235 + 20 * u, 130 + 115 * u, 40 + 90 * u];
    }
  }

  viridisColor(t) {
    const r = 68 + 185 * Math.pow(t, 1.5);
    const g = 1 + 230 * Math.pow(t, 0.85);
    const b = 84 + 90 * Math.sin(t * Math.PI);
    return [Math.min(255, r), Math.min(255, g), Math.min(255, b)];
  }

  divergingColor(v) {
    // -1 = blue, 0 = dark slate, +1 = red/orange
    if (v >= 0) {
      return [22 + 230 * v, 26 + 70 * v, 38 - 10 * v];
    } else {
      const u = -v;
      return [22 + 20 * u, 26 + 135 * u, 38 + 210 * u];
    }
  }
}
