/**
 * 2D Incompressible Navier-Stokes + Boussinesq Thermal Buoyancy Solver
 * Discretisation: Marker-and-Cell (MAC) Staggered Finite-Volume Grid
 * Time Integration: Chorin's Projection Method (Operator Splitting)
 *
 * Coordinate convention (standard physics):
 *   i in [0, nx-1] : x-axis (left -> right; left = inside room, right = outdoors)
 *   j in [0, ny-1] : y-axis (bottom -> top; +y is UP, gravity acts in -y)
 *
 * Staggered MAC Grid Layout for cell (i, j):
 *   - Scalars (p, T, s, div, solid) live at cell CENTERS: (i + 0.5, j + 0.5)
 *   - Horizontal velocity u(i, j) lives on LEFT vertical face of cell (i, j): (i, j + 0.5)
 *       u has dimensions (nx + 1) x ny
 *   - Vertical velocity   v(i, j) lives on BOTTOM horizontal face of cell (i, j): (i + 0.5, j)
 *       v has dimensions nx x (ny + 1)
 */

// Cell types for boundary conditions
export const CELL_FLUID = 0;
export const CELL_SOLID = 1;
export const CELL_OPEN_OUTDOOR = 2; // Dirichlet pressure = 0, outdoor T, zero-gradient / inflow wind
export const CELL_OPEN_VENT = 3;    // Rear room vent: Dirichlet pressure = deltaP, indoor T

export class SmokeFluidSolver {
  constructor(nx = 160, ny = 96) {
    this.nx = nx;
    this.ny = ny;
    // Physical domain height = 3.0 meters (typical room + exterior slice)
    this.domainHeight = 3.0;
    this.dx = this.domainHeight / ny; // uniform square cells: dx = dy = h
    this.domainWidth = this.dx * nx;

    // Physical & numerical parameters (all tweakable live from UI)
    this.params = {
      dt: 0.015,               // Base timestep (seconds), auto-clamped if needed
      tempIndoor: 22.0,        // Indoor ambient temperature (°C)
      tempOutdoor: 5.0,        // Outdoor ambient temperature (°C)
      tempSmoke: 45.0,         // Temperature of emitted smoke (°C)
      buoyancyBeta: 0.0034,    // Thermal expansion coeff β ≈ 1/293 K^-1 for ideal gas
      smokeDensityFactor: 0.0, // Excess density κ of smoke particulates (>0 heavy, <0 light)
      gravity: 9.81,           // m/s^2
      viscosity: 0.0005,       // Effective kinematic viscosity ν (m^2/s)
      thermalDiffusivity: 0.0006, // α (m^2/s)
      smokeDiffusivity: 0.0002,   // D (m^2/s)
      pressureDiff: 0.0,       // Indoor-minus-outdoor ΔP (Pascals, scaled for ρ=1.2 kg/m^3)
      windSpeedX: 0.0,         // Outdoor horizontal wind (m/s, negative blows INTO window)
      windSpeedY: 0.0,         // Outdoor vertical draft along facade (m/s)
      vorticityConf: 0.15,     // Vorticity confinement ε (counters numerical dissipation)
      advectionOrder: 2,       // 1 = Semi-Lagrangian, 2 = MacCormack (2nd-order)
      poissonIters: 60,        // Red-Black SOR iterations per step
      sorOmega: 1.78,          // Over-relaxation factor ω in (1, 2)

      // Smoke emitter properties (normalized domain coords 0..1)
      emitterX: 0.42,
      emitterY: 0.36,
      emitterRadius: 0.025,    // fraction of domain height
      emissionRate: 1.0,       // target smoke concentration
      jetSpeed: 0.35,          // m/s initial velocity of smoke plume
      jetAngleDeg: 35,         // degrees (0 = right toward window, 90 = straight up)
      emitterEnabled: true,

      // Room & Window Geometry (normalized fractions)
      wallXFrac: 0.56,         // x position of the window wall
      roomLeftFrac: 0.04,      // left wall of room
      roomFloorFrac: 0.06,     // floor level
      roomCeilFrac: 0.88,      // ceiling level
      windowSillFrac: 0.32,    // bottom of window opening
      windowTopFrac: 0.68,     // top of window opening
      windowBaffle: 0.0,       // 0 = none, >0 = angled awning/casement sash length
      rearVentOpen: false,     // rear door/vent on left wall (allows cross-ventilation & ΔP)
      rearVentBottomFrac: 0.10,
      rearVentTopFrac: 0.45,
    };

    this.allocateFields();
    this.userSolidMask = new Uint8Array(nx * ny); // user-painted custom obstacles
    this.rebuildGeometry();
    this.resetFields();
  }

  allocateFields() {
    const { nx, ny } = this;
    const nCells = nx * ny;
    const nU = (nx + 1) * ny;
    const nV = nx * (ny + 1);

    // Primary staggered velocity fields
    this.u = new Float32Array(nU);
    this.v = new Float32Array(nV);
    this.uTmp = new Float32Array(nU);
    this.vTmp = new Float32Array(nV);
    this.uAux = new Float32Array(nU); // For MacCormack backward step
    this.vAux = new Float32Array(nV);

    // Cell-centered scalar fields
    this.p = new Float32Array(nCells);     // Pressure (kinematic p/ρ)
    this.div = new Float32Array(nCells);   // Velocity divergence
    this.T = new Float32Array(nCells);     // Temperature (°C)
    this.TTmp = new Float32Array(nCells);
    this.TAux = new Float32Array(nCells);
    this.s = new Float32Array(nCells);     // Smoke concentration [0, 1+]
    this.sTmp = new Float32Array(nCells);
    this.sAux = new Float32Array(nCells);
    this.curl = new Float32Array(nCells);  // Vorticity ω = ∂v/∂x - ∂u/∂y

    // Geometry / boundary classification
    this.cellType = new Uint8Array(nCells);

    // Diagnostics updated each step
    this.stats = {
      time: 0,
      stepCount: 0,
      maxSpeed: 0,
      cfl: 0,
      maxDivBefore: 0,
      maxDivAfter: 0,
      smokeInside: 0,
      smokeOutside: 0,
    };
  }

  // Indexing helpers
  idx(i, j) { return i + j * this.nx; }
  idxU(i, j) { return i + j * (this.nx + 1); }
  idxV(i, j) { return i + j * this.nx; }

  /**
   * Builds the discrete geometry mask (solid walls, window opening, rear vent, open outdoors).
   */
  rebuildGeometry() {
    const { nx, ny, params } = this;
    const wallI = Math.max(4, Math.min(nx - 10, Math.round(params.wallXFrac * nx)));
    const leftI = Math.max(1, Math.min(wallI - 6, Math.round(params.roomLeftFrac * nx)));
    const floorJ = Math.max(1, Math.min(ny - 10, Math.round(params.roomFloorFrac * ny)));
    const ceilJ = Math.max(floorJ + 8, Math.min(ny - 2, Math.round(params.roomCeilFrac * ny)));
    const sillJ = Math.max(floorJ + 1, Math.min(ceilJ - 2, Math.round(params.windowSillFrac * ny)));
    const topJ = Math.max(sillJ + 2, Math.min(ceilJ - 1, Math.round(params.windowTopFrac * ny)));
    const ventBotJ = Math.max(floorJ + 1, Math.round(params.rearVentBottomFrac * ny));
    const ventTopJ = Math.min(ceilJ - 1, Math.round(params.rearVentTopFrac * ny));

    this.geom = { wallI, leftI, floorJ, ceilJ, sillJ, topJ, ventBotJ, ventTopJ };

    const wallThick = 2;
    this.cellType.fill(CELL_FLUID);

    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = this.idx(i, j);

        // 1. User-painted solid obstacles
        if (this.userSolidMask[c]) {
          this.cellType[c] = CELL_SOLID;
          continue;
        }

        // 2. Room envelope (left of wallI + wallThick)
        const inRoomX = i >= leftI && i < wallI + wallThick;
        const inWallX = i >= wallI && i < wallI + wallThick;

        // Exterior space outside the room's left/bottom/top shell
        if (i < leftI) {
          if (params.rearVentOpen && j >= ventBotJ && j < ventTopJ) {
            this.cellType[c] = CELL_OPEN_VENT;
          } else {
            this.cellType[c] = CELL_SOLID;
          }
          continue;
        }

        // Floor & ground outside
        if (j < floorJ) {
          this.cellType[c] = CELL_SOLID;
          continue;
        }

        // Roof / ceiling above room
        if (inRoomX && j >= ceilJ) {
          this.cellType[c] = CELL_SOLID;
          continue;
        }

        // Left wall of room
        if (i === leftI && j >= floorJ && j < ceilJ) {
          if (params.rearVentOpen && j >= ventBotJ && j < ventTopJ) {
            this.cellType[c] = CELL_OPEN_VENT;
          } else {
            this.cellType[c] = CELL_SOLID;
          }
          continue;
        }

        // Window wall (separating inside room from outdoors)
        if (inWallX && j >= floorJ && j < ceilJ) {
          if (j >= sillJ && j < topJ) {
            this.cellType[c] = CELL_FLUID; // Window opening!
          } else {
            this.cellType[c] = CELL_SOLID;
          }
          continue;
        }

        // Optional angled awning / tilted sash baffle extending outward from window top
        if (params.windowBaffle > 0.01) {
          const baffleLen = Math.round(params.windowBaffle * (topJ - sillJ));
          const di = i - (wallI + wallThick);
          if (di >= 0 && di < baffleLen) {
            const baffleJ = topJ - Math.round(di * 0.65);
            if (Math.abs(j - baffleJ) <= 0) {
              this.cellType[c] = CELL_SOLID;
              continue;
            }
          }
        }

        // Open outdoor domain boundaries (right edge, top edge outside room)
        if (i === nx - 1 || (j === ny - 1 && i >= wallI + wallThick)) {
          this.cellType[c] = CELL_OPEN_OUTDOOR;
          continue;
        }
      }
    }
  }

  /**
   * Resets velocities, pressure, smoke, and initializes indoor/outdoor temperatures.
   */
  resetFields() {
    const { nx, ny, params, geom } = this;
    this.u.fill(0);
    this.v.fill(0);
    this.p.fill(0);
    this.s.fill(0);
    this.curl.fill(0);

    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = this.idx(i, j);
        // Smooth transition right at the window wall
        if (i <= geom.wallI) {
          this.T[c] = params.tempIndoor;
        } else {
          this.T[c] = params.tempOutdoor;
        }
      }
    }

    // Initialize outdoor wind velocity
    if (Math.abs(params.windSpeedX) > 1e-4 || Math.abs(params.windSpeedY) > 1e-4) {
      for (let j = geom.floorJ; j < ny; j++) {
        for (let i = geom.wallI + 2; i <= nx; i++) {
          this.u[this.idxU(i, j)] = params.windSpeedX;
        }
        for (let i = geom.wallI + 2; i < nx; i++) {
          this.v[this.idxV(i, j)] = params.windSpeedY;
        }
      }
    }

    this.stats.time = 0;
    this.stats.stepCount = 0;
  }

  clearObstacles() {
    this.userSolidMask.fill(0);
    this.rebuildGeometry();
  }

  paintObstacle(normX, normY, radiusCells, isSolid) {
    const ci = Math.round(normX * this.nx);
    const cj = Math.round(normY * this.ny);
    const r2 = radiusCells * radiusCells;
    for (let j = Math.max(1, cj - radiusCells); j <= Math.min(this.ny - 2, cj + radiusCells); j++) {
      for (let i = Math.max(1, ci - radiusCells); i <= Math.min(this.nx - 2, ci + radiusCells); i++) {
        if ((i - ci) * (i - ci) + (j - cj) * (j - cj) <= r2) {
          this.userSolidMask[this.idx(i, j)] = isSolid ? 1 : 0;
        }
      }
    }
    this.rebuildGeometry();
  }

  /**
   * Advance the simulation by one fractional-step (Chorin projection) timestep.
   */
  step() {
    const dt = this.params.dt;

    // 1. Inject smoke, heat, and momentum from the smoke source
    this.applyEmitter(dt);

    // 2. Apply body forces: Boussinesq thermal buoyancy + smoke excess weight + vorticity confinement
    this.applyBuoyancy(dt);
    if (this.params.vorticityConf > 0.001) {
      this.applyVorticityConfinement(dt);
    }

    // 3. Enforce boundary conditions before advection
    this.enforceVelocityBC();

    // 4. Advection: transport u, v, T, s along the velocity field
    this.advectAll(dt);
    this.enforceVelocityBC();

    // 5. Diffusion: viscous momentum damping + thermal & smoke conduction/diffusion
    if (this.params.viscosity > 1e-6) {
      this.diffuseVelocity(dt, this.params.viscosity, 10);
    }
    if (this.params.thermalDiffusivity > 1e-6) {
      this.diffuseScalar(this.T, this.TTmp, dt, this.params.thermalDiffusivity, 8);
    }
    if (this.params.smokeDiffusivity > 1e-6) {
      this.diffuseScalar(this.s, this.sTmp, dt, this.params.smokeDiffusivity, 8);
    }

    // Re-apply scalar BCs (outdoor & vent temperatures/smoke)
    this.enforceScalarBC();
    this.enforceVelocityBC();

    // 6. Pressure Projection: solve ∇²φ = ∇·u* and subtract ∇φ to enforce ∇·u = 0
    this.project(dt);
    this.enforceVelocityBC();

    // 7. Compute vorticity & diagnostics for visualization
    this.computeVorticityAndStats(dt);
  }

  /**
   * Stage 1: Smoke Emitter (Gaussian-smoothed source for concentration, temperature, and jet momentum)
   */
  applyEmitter(dt) {
    const { nx, ny, params } = this;
    if (!params.emitterEnabled) return;

    const ex = params.emitterX * nx;
    const ey = params.emitterY * ny;
    const r = Math.max(1.5, params.emitterRadius * ny);
    const r2 = r * r;

    const angleRad = (params.jetAngleDeg * Math.PI) / 180;
    const jetU = params.jetSpeed * Math.cos(angleRad);
    const jetV = params.jetSpeed * Math.sin(angleRad);

    const iMin = Math.max(1, Math.floor(ex - r - 1));
    const iMax = Math.min(nx - 2, Math.ceil(ex + r + 1));
    const jMin = Math.max(1, Math.floor(ey - r - 1));
    const jMax = Math.min(ny - 2, Math.ceil(ey + r + 1));

    const relax = Math.min(1.0, dt * 14.0);

    for (let j = jMin; j <= jMax; j++) {
      for (let i = iMin; i <= iMax; i++) {
        const c = this.idx(i, j);
        if (this.cellType[c] !== CELL_FLUID) continue;

        const dx = (i + 0.5) - ex;
        const dy = (j + 0.5) - ey;
        const d2 = dx * dx + dy * dy;
        if (d2 > r2) continue;

        const w = Math.exp(-2.5 * (d2 / r2));
        // Relax smoke & temperature toward emitter values
        this.s[c] += (params.emissionRate - this.s[c]) * w * relax;
        this.T[c] += (params.tempSmoke - this.T[c]) * w * relax;

        // Inject jet velocity on adjacent faces
        if (params.jetSpeed > 1e-4) {
          const uIdx = this.idxU(i, j);
          const uIdxR = this.idxU(i + 1, j);
          const vIdx = this.idxV(i, j);
          const vIdxT = this.idxV(i, j + 1);
          this.u[uIdx] += (jetU - this.u[uIdx]) * w * relax * 0.7;
          this.u[uIdxR] += (jetU - this.u[uIdxR]) * w * relax * 0.7;
          this.v[vIdx] += (jetV - this.v[vIdx]) * w * relax * 0.7;
          this.v[vIdxT] += (jetV - this.v[vIdxT]) * w * relax * 0.7;
        }
      }
    }
  }

  /**
   * Stage 2a: Boussinesq Buoyancy Force
   *   f_y = g * [ β * (T - T_ref) - κ * s ]
   * Applied directly to vertical velocity faces v(i, j) between cells (i, j-1) and (i, j).
   */
  applyBuoyancy(dt) {
    const { nx, ny, params } = this;
    const Tref = 0.5 * (params.tempIndoor + params.tempOutdoor);
    const g = params.gravity;
    const beta = params.buoyancyBeta;
    const kappa = params.smokeDensityFactor;

    for (let j = 1; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const cBot = this.idx(i, j - 1);
        const cTop = this.idx(i, j);
        if (this.cellType[cBot] === CELL_SOLID || this.cellType[cTop] === CELL_SOLID) continue;

        const Tface = 0.5 * (this.T[cBot] + this.T[cTop]);
        const sFace = 0.5 * (this.s[cBot] + this.s[cTop]);

        // Positive acceleration = upward (+y)
        const fy = g * (beta * (Tface - Tref) - kappa * sFace);
        this.v[this.idxV(i, j)] += dt * fy;
      }
    }
  }

  /**
   * Stage 2b: Vorticity Confinement (Fedkiw et al. 2001)
   * Re-injects small-scale rotational energy lost to numerical dissipation on coarse grids:
   *   f_conf = ε * h * (N × ω), where N = ∇|ω| / |∇|ω||
   */
  applyVorticityConfinement(dt) {
    const { nx, ny, dx, params } = this;
    const eps = params.vorticityConf;

    // Compute cell-centered vorticity ω = ∂v/∂x - ∂u/∂y
    const inv2h = 0.5 / dx;
    for (let j = 1; j < ny - 1; j++) {
      for (let i = 1; i < nx - 1; i++) {
        const c = this.idx(i, j);
        if (this.cellType[c] !== CELL_FLUID) {
          this.curl[c] = 0;
          continue;
        }
        const vR = 0.5 * (this.v[this.idxV(i + 1, j)] + this.v[this.idxV(i + 1, j + 1)]);
        const vL = 0.5 * (this.v[this.idxV(i - 1, j)] + this.v[this.idxV(i - 1, j + 1)]);
        const uT = 0.5 * (this.u[this.idxU(i, j + 1)] + this.u[this.idxU(i + 1, j + 1)]);
        const uB = 0.5 * (this.u[this.idxU(i, j - 1)] + this.u[this.idxU(i + 1, j - 1)]);
        this.curl[c] = (vR - vL - (uT - uB)) * inv2h;
      }
    }

    for (let j = 2; j < ny - 2; j++) {
      for (let i = 2; i < nx - 2; i++) {
        const c = this.idx(i, j);
        if (this.cellType[c] !== CELL_FLUID) continue;

        const dwDx = (Math.abs(this.curl[this.idx(i + 1, j)]) - Math.abs(this.curl[this.idx(i - 1, j)])) * inv2h;
        const dwDy = (Math.abs(this.curl[this.idx(i, j + 1)]) - Math.abs(this.curl[this.idx(i, j - 1)])) * inv2h;
        const mag = Math.hypot(dwDx, dwDy) + 1e-6;
        const Nx = dwDx / mag;
        const Ny = dwDy / mag;
        const w = this.curl[c];

        // 2D cross product (N × ω z-hat) = (Ny * w, -Nx * w)
        const fx = eps * dx * (Ny * w);
        const fy = eps * dx * (-Nx * w);

        this.u[this.idxU(i, j)] += 0.5 * dt * fx;
        this.u[this.idxU(i + 1, j)] += 0.5 * dt * fx;
        this.v[this.idxV(i, j)] += 0.5 * dt * fy;
        this.v[this.idxV(i, j + 1)] += 0.5 * dt * fy;
      }
    }
  }

  /**
   * Velocity interpolation on the MAC staggered grid at continuous grid coords (gx, gy).
   * Cell (i, j) center is at (i + 0.5, j + 0.5).
   *   u(i, j) is at (i, j + 0.5)
   *   v(i, j) is at (i + 0.5, j)
   */
  sampleU(gx, gy) {
    const { nx, ny, u } = this;
    const x = Math.max(0, Math.min(nx - 1e-4, gx));
    const y = Math.max(0, Math.min(ny - 1 - 1e-4, gy - 0.5));
    const i0 = Math.floor(x);
    const j0 = Math.floor(y);
    const tx = x - i0;
    const ty = y - j0;
    const i1 = Math.min(nx, i0 + 1);
    const j1 = Math.min(ny - 1, j0 + 1);

    const w00 = u[this.idxU(i0, j0)];
    const w10 = u[this.idxU(i1, j0)];
    const w01 = u[this.idxU(i0, j1)];
    const w11 = u[this.idxU(i1, j1)];
    return (1 - ty) * ((1 - tx) * w00 + tx * w10) + ty * ((1 - tx) * w01 + tx * w11);
  }

  sampleV(gx, gy) {
    const { nx, ny, v } = this;
    const x = Math.max(0, Math.min(nx - 1 - 1e-4, gx - 0.5));
    const y = Math.max(0, Math.min(ny - 1e-4, gy));
    const i0 = Math.floor(x);
    const j0 = Math.floor(y);
    const tx = x - i0;
    const ty = y - j0;
    const i1 = Math.min(nx - 1, i0 + 1);
    const j1 = Math.min(ny, j0 + 1);

    const w00 = v[this.idxV(i0, j0)];
    const w10 = v[this.idxV(i1, j0)];
    const w01 = v[this.idxV(i0, j1)];
    const w11 = v[this.idxV(i1, j1)];
    return (1 - ty) * ((1 - tx) * w00 + tx * w10) + ty * ((1 - tx) * w01 + tx * w11);
  }

  sampleScalar(field, gx, gy) {
    const { nx, ny } = this;
    const x = Math.max(0, Math.min(nx - 1 - 1e-4, gx - 0.5));
    const y = Math.max(0, Math.min(ny - 1 - 1e-4, gy - 0.5));
    const i0 = Math.floor(x);
    const j0 = Math.floor(y);
    const tx = x - i0;
    const ty = y - j0;
    const i1 = Math.min(nx - 1, i0 + 1);
    const j1 = Math.min(ny - 1, j0 + 1);

    const w00 = field[this.idx(i0, j0)];
    const w10 = field[this.idx(i1, j0)];
    const w01 = field[this.idx(i0, j1)];
    const w11 = field[this.idx(i1, j1)];
    return (1 - ty) * ((1 - tx) * w00 + tx * w10) + ty * ((1 - tx) * w01 + tx * w11);
  }

  /**
   * 2nd-order Runge-Kutta (midpoint) backtrace in grid-index coordinates.
   */
  traceBack(gx, gy, dt) {
    const invH = 1.0 / this.dx;
    const u0 = this.sampleU(gx, gy);
    const v0 = this.sampleV(gx, gy);
    const midX = gx - 0.5 * dt * u0 * invH;
    const midY = gy - 0.5 * dt * v0 * invH;
    const u1 = this.sampleU(midX, midY);
    const v1 = this.sampleV(midX, midY);
    return [gx - dt * u1 * invH, gy - dt * v1 * invH];
  }

  /**
   * Stage 4: Advection (Semi-Lagrangian 1st-order or clamped MacCormack 2nd-order)
   */
  advectAll(dt) {
    const useMacCormack = this.params.advectionOrder === 2;
    this.advectU(dt, useMacCormack);
    this.advectV(dt, useMacCormack);
    this.advectScalar(this.T, this.TTmp, this.TAux, dt, useMacCormack, -50, 200);
    this.advectScalar(this.s, this.sTmp, this.sAux, dt, useMacCormack, 0, 2.0);

    // Swap buffers
    [this.u, this.uTmp] = [this.uTmp, this.u];
    [this.v, this.vTmp] = [this.vTmp, this.v];
    [this.T, this.TTmp] = [this.TTmp, this.T];
    [this.s, this.sTmp] = [this.sTmp, this.s];
  }

  advectU(dt, macCormack) {
    const { nx, ny, u, uTmp, uAux } = this;
    // Forward Semi-Lagrangian step -> uTmp
    for (let j = 0; j < ny; j++) {
      for (let i = 1; i < nx; i++) {
        const cL = this.idx(i - 1, j);
        const cR = this.idx(i, j);
        if (this.cellType[cL] === CELL_SOLID || this.cellType[cR] === CELL_SOLID) {
          uTmp[this.idxU(i, j)] = 0;
          continue;
        }
        const [px, py] = this.traceBack(i, j + 0.5, dt);
        uTmp[this.idxU(i, j)] = this.sampleU(px, py);
      }
    }

    if (!macCormack) return;

    // MacCormack backward step (advect uTmp forward by -dt) -> uAux
    const invH = 1.0 / this.dx;
    for (let j = 0; j < ny; j++) {
      for (let i = 1; i < nx; i++) {
        const idx = this.idxU(i, j);
        const cL = this.idx(i - 1, j);
        const cR = this.idx(i, j);
        if (this.cellType[cL] === CELL_SOLID || this.cellType[cR] === CELL_SOLID) continue;

        const gx = i + dt * u[idx] * invH;
        const gy = j + 0.5 + dt * this.sampleV(i, j + 0.5) * invH;
        // Sample uTmp at (gx, gy)
        const x = Math.max(0, Math.min(nx - 1e-4, gx));
        const y = Math.max(0, Math.min(ny - 1 - 1e-4, gy - 0.5));
        const i0 = Math.floor(x), j0 = Math.floor(y);
        const tx = x - i0, ty = y - j0;
        const i1 = Math.min(nx, i0 + 1), j1 = Math.min(ny - 1, j0 + 1);
        const backVal =
          (1 - ty) * ((1 - tx) * uTmp[this.idxU(i0, j0)] + tx * uTmp[this.idxU(i1, j0)]) +
          ty * ((1 - tx) * uTmp[this.idxU(i0, j1)] + tx * uTmp[this.idxU(i1, j1)]);
        uAux[idx] = uTmp[idx] + 0.5 * (u[idx] - backVal);
      }
    }

    // Clamp to local extrema of original stencil (prevents overshoot/ringing)
    for (let j = 0; j < ny; j++) {
      for (let i = 1; i < nx; i++) {
        const idx = this.idxU(i, j);
        const cL = this.idx(i - 1, j);
        const cR = this.idx(i, j);
        if (this.cellType[cL] === CELL_SOLID || this.cellType[cR] === CELL_SOLID) continue;

        const [px, py] = this.traceBack(i, j + 0.5, dt);
        const x = Math.max(0, Math.min(nx - 1e-4, px));
        const y = Math.max(0, Math.min(ny - 1 - 1e-4, py - 0.5));
        const i0 = Math.floor(x), j0 = Math.floor(y);
        const i1 = Math.min(nx, i0 + 1), j1 = Math.min(ny - 1, j0 + 1);
        const a = u[this.idxU(i0, j0)], b = u[this.idxU(i1, j0)];
        const c = u[this.idxU(i0, j1)], d = u[this.idxU(i1, j1)];
        const lo = Math.min(a, b, c, d);
        const hi = Math.max(a, b, c, d);
        uTmp[idx] = Math.max(lo, Math.min(hi, uAux[idx]));
      }
    }
  }

  advectV(dt, macCormack) {
    const { nx, ny, v, vTmp, vAux } = this;
    for (let j = 1; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const cB = this.idx(i, j - 1);
        const cT = this.idx(i, j);
        if (this.cellType[cB] === CELL_SOLID || this.cellType[cT] === CELL_SOLID) {
          vTmp[this.idxV(i, j)] = 0;
          continue;
        }
        const [px, py] = this.traceBack(i + 0.5, j, dt);
        vTmp[this.idxV(i, j)] = this.sampleV(px, py);
      }
    }

    if (!macCormack) return;

    const invH = 1.0 / this.dx;
    for (let j = 1; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = this.idxV(i, j);
        const cB = this.idx(i, j - 1);
        const cT = this.idx(i, j);
        if (this.cellType[cB] === CELL_SOLID || this.cellType[cT] === CELL_SOLID) continue;

        const gx = i + 0.5 + dt * this.sampleU(i + 0.5, j) * invH;
        const gy = j + dt * v[idx] * invH;
        const x = Math.max(0, Math.min(nx - 1 - 1e-4, gx - 0.5));
        const y = Math.max(0, Math.min(ny - 1e-4, gy));
        const i0 = Math.floor(x), j0 = Math.floor(y);
        const tx = x - i0, ty = y - j0;
        const i1 = Math.min(nx - 1, i0 + 1), j1 = Math.min(ny, j0 + 1);
        const backVal =
          (1 - ty) * ((1 - tx) * vTmp[this.idxV(i0, j0)] + tx * vTmp[this.idxV(i1, j0)]) +
          ty * ((1 - tx) * vTmp[this.idxV(i0, j1)] + tx * vTmp[this.idxV(i1, j1)]);
        vAux[idx] = vTmp[idx] + 0.5 * (v[idx] - backVal);
      }
    }

    for (let j = 1; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = this.idxV(i, j);
        const cB = this.idx(i, j - 1);
        const cT = this.idx(i, j);
        if (this.cellType[cB] === CELL_SOLID || this.cellType[cT] === CELL_SOLID) continue;

        const [px, py] = this.traceBack(i + 0.5, j, dt);
        const x = Math.max(0, Math.min(nx - 1 - 1e-4, px - 0.5));
        const y = Math.max(0, Math.min(ny - 1e-4, py));
        const i0 = Math.floor(x), j0 = Math.floor(y);
        const i1 = Math.min(nx - 1, i0 + 1), j1 = Math.min(ny, j0 + 1);
        const a = v[this.idxV(i0, j0)], b = v[this.idxV(i1, j0)];
        const c = v[this.idxV(i0, j1)], d = v[this.idxV(i1, j1)];
        const lo = Math.min(a, b, c, d);
        const hi = Math.max(a, b, c, d);
        vTmp[idx] = Math.max(lo, Math.min(hi, vAux[idx]));
      }
    }
  }

  advectScalar(src, dst, aux, dt, macCormack, minClamp, maxClamp) {
    const { nx, ny } = this;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = this.idx(i, j);
        if (this.cellType[c] !== CELL_FLUID) {
          dst[c] = src[c];
          continue;
        }
        const [px, py] = this.traceBack(i + 0.5, j + 0.5, dt);
        dst[c] = this.sampleScalar(src, px, py);
      }
    }

    if (!macCormack) return;

    const invH = 1.0 / this.dx;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = this.idx(i, j);
        if (this.cellType[c] !== CELL_FLUID) continue;
        const gx = i + 0.5 + dt * this.sampleU(i + 0.5, j + 0.5) * invH;
        const gy = j + 0.5 + dt * this.sampleV(i + 0.5, j + 0.5) * invH;
        const backVal = this.sampleScalar(dst, gx, gy);
        aux[c] = dst[c] + 0.5 * (src[c] - backVal);
      }
    }

    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = this.idx(i, j);
        if (this.cellType[c] !== CELL_FLUID) continue;
        const [px, py] = this.traceBack(i + 0.5, j + 0.5, dt);
        const x = Math.max(0, Math.min(nx - 1 - 1e-4, px - 0.5));
        const y = Math.max(0, Math.min(ny - 1 - 1e-4, py - 0.5));
        const i0 = Math.floor(x), j0 = Math.floor(y);
        const i1 = Math.min(nx - 1, i0 + 1), j1 = Math.min(ny - 1, j0 + 1);
        const a = src[this.idx(i0, j0)], b = src[this.idx(i1, j0)];
        const d = src[this.idx(i0, j1)], e = src[this.idx(i1, j1)];
        const lo = Math.max(minClamp, Math.min(a, b, d, e));
        const hi = Math.min(maxClamp, Math.max(a, b, d, e));
        dst[c] = Math.max(lo, Math.min(hi, aux[c]));
      }
    }
  }

  /**
   * Stage 5a: Implicit Diffusion for Scalars (Backward Euler + Gauss-Seidel)
   * Solves (I - D*dt*∇²) φ^{n+1} = φ^n, unconditionally stable for any diffusivity D.
   */
  diffuseScalar(field, rhs, dt, diffCoeff, iters) {
    const { nx, ny, dx } = this;
    const a = (diffCoeff * dt) / (dx * dx);
    if (a < 1e-7) return;

    rhs.set(field);
    for (let k = 0; k < iters; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const c = this.idx(i, j);
          if (this.cellType[c] !== CELL_FLUID) continue;

          // Zero-flux (Neumann) at solid walls: only average fluid/open neighbors
          let sum = 0;
          let count = 0;
          const nL = this.idx(i - 1, j), nR = this.idx(i + 1, j);
          const nB = this.idx(i, j - 1), nT = this.idx(i, j + 1);
          if (this.cellType[nL] !== CELL_SOLID) { sum += field[nL]; count++; }
          if (this.cellType[nR] !== CELL_SOLID) { sum += field[nR]; count++; }
          if (this.cellType[nB] !== CELL_SOLID) { sum += field[nB]; count++; }
          if (this.cellType[nT] !== CELL_SOLID) { sum += field[nT]; count++; }

          field[c] = (rhs[c] + a * sum) / (1 + a * count);
        }
      }
    }
  }

  /**
   * Stage 5b: Implicit Viscous Diffusion for Velocities (Backward Euler + Gauss-Seidel)
   */
  diffuseVelocity(dt, nu, iters) {
    const { nx, ny, dx, u, v, uTmp, vTmp } = this;
    const a = (nu * dt) / (dx * dx);
    if (a < 1e-7) return;
    const invDiag = 1.0 / (1.0 + 4.0 * a);

    uTmp.set(u);
    vTmp.set(v);

    for (let k = 0; k < iters; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx; i++) {
          const cL = this.idx(i - 1, j), cR = this.idx(i, j);
          if (this.cellType[cL] === CELL_SOLID || this.cellType[cR] === CELL_SOLID) continue;
          const idx = this.idxU(i, j);
          u[idx] = (uTmp[idx] + a * (
            u[this.idxU(i - 1, j)] + u[this.idxU(i + 1, j)] +
            u[this.idxU(i, j - 1)] + u[this.idxU(i, j + 1)]
          )) * invDiag;
        }
      }
      for (let j = 1; j < ny; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const cB = this.idx(i, j - 1), cT = this.idx(i, j);
          if (this.cellType[cB] === CELL_SOLID || this.cellType[cT] === CELL_SOLID) continue;
          const idx = this.idxV(i, j);
          v[idx] = (vTmp[idx] + a * (
            v[this.idxV(i - 1, j)] + v[this.idxV(i + 1, j)] +
            v[this.idxV(i, j - 1)] + v[this.idxV(i, j + 1)]
          )) * invDiag;
        }
      }
    }
  }

  /**
   * Boundary Conditions for Velocities:
   *  - Solid walls: no-penetration (normal velocity = 0)
   *  - Open outdoor boundaries: wind inflow or zero-gradient outflow
   */
  enforceVelocityBC() {
    const { nx, ny, u, v, params } = this;

    // 1. Solid wall no-penetration on all interior faces
    for (let j = 0; j < ny; j++) {
      u[this.idxU(0, j)] = 0;
      for (let i = 1; i < nx; i++) {
        const cL = this.idx(i - 1, j);
        const cR = this.idx(i, j);
        if (this.cellType[cL] === CELL_SOLID || this.cellType[cR] === CELL_SOLID) {
          u[this.idxU(i, j)] = 0;
        }
      }
      // Rightmost outdoor boundary
      const cEdge = this.idx(nx - 1, j);
      if (this.cellType[cEdge] === CELL_OPEN_OUTDOOR) {
        if (params.windSpeedX < -1e-3) {
          // Wind blowing INTO domain from right
          u[this.idxU(nx, j)] = params.windSpeedX;
          u[this.idxU(nx - 1, j)] = params.windSpeedX;
        } else {
          // Outflow: zero normal gradient + optional outward wind
          u[this.idxU(nx, j)] = Math.max(u[this.idxU(nx - 1, j)], params.windSpeedX);
        }
      } else {
        u[this.idxU(nx, j)] = 0;
      }
    }

    for (let i = 0; i < nx; i++) {
      v[this.idxV(i, 0)] = 0;
      for (let j = 1; j < ny; j++) {
        const cB = this.idx(i, j - 1);
        const cT = this.idx(i, j);
        if (this.cellType[cB] === CELL_SOLID || this.cellType[cT] === CELL_SOLID) {
          v[this.idxV(i, j)] = 0;
        }
      }
      // Top boundary (open outdoors above outside region)
      const cTop = this.idx(i, ny - 1);
      if (this.cellType[cTop] === CELL_OPEN_OUTDOOR) {
        if (params.windSpeedY < -1e-3) {
          v[this.idxV(i, ny)] = params.windSpeedY;
        } else {
          v[this.idxV(i, ny)] = v[this.idxV(i, ny - 1)];
        }
      } else {
        v[this.idxV(i, ny)] = 0;
      }
    }

    // Gentle outdoor wind nudging in the far-right buffer strip (last 4 columns)
    if (Math.abs(params.windSpeedX) > 1e-3 || Math.abs(params.windSpeedY) > 1e-3) {
      const relax = 0.08;
      for (let j = this.geom.floorJ; j < ny - 1; j++) {
        for (let i = nx - 4; i < nx; i++) {
          if (this.cellType[this.idx(i, j)] !== CELL_SOLID) {
            u[this.idxU(i, j)] += (params.windSpeedX - u[this.idxU(i, j)]) * relax;
            v[this.idxV(i, j)] += (params.windSpeedY - v[this.idxV(i, j)]) * relax;
          }
        }
      }
    }
  }

  /**
   * Boundary Conditions for Temperature & Smoke at open boundaries.
   */
  enforceScalarBC() {
    const { nx, ny, params } = this;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = this.idx(i, j);
        const type = this.cellType[c];
        if (type === CELL_OPEN_OUTDOOR) {
          this.T[c] = params.tempOutdoor;
          this.s[c] = 0.0; // Smoke leaving the domain dissipates into open atmosphere
        } else if (type === CELL_OPEN_VENT) {
          this.T[c] = params.tempIndoor;
          this.s[c] *= 0.85; // Clean indoor makeup air entering through rear vent
        }
      }
    }
  }

  /**
   * Stage 6: Chorin Pressure Projection
   *
   * 1. Compute velocity divergence in each fluid cell:
   *      div(i, j) = (u(i+1,j) - u(i,j) + v(i,j+1) - v(i,j)) / h
   * 2. Solve discrete Poisson equation:
   *      ∇²p = (ρ / dt) * div   (here we solve for scaled potential φ = (dt / ρ) * p)
   *    using Red-Black Successive Over-Relaxation (SOR).
   *    - Solid walls: Neumann BC (∂φ/∂n = 0)
   *    - Open outdoor cells: Dirichlet BC (φ = 0)
   *    - Rear vent cells: Dirichlet BC (φ = (dt / ρ) * ΔP)
   * 3. Subtract ∇φ from velocities:
   *      u <- u - ∂φ/∂x,   v <- v - ∂φ/∂y
   */
  project(dt) {
    const { nx, ny, dx, u, v, p, div, params } = this;
    const invH = 1.0 / dx;
    const h2 = dx * dx;
    const rhoAir = 1.2; // kg/m^3
    const phiVent = (dt / rhoAir) * params.pressureDiff;

    // 1. Compute divergence & set Dirichlet pressure values
    let maxDivBefore = 0;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = this.idx(i, j);
        const type = this.cellType[c];
        if (type === CELL_OPEN_OUTDOOR) {
          p[c] = 0.0;
          div[c] = 0.0;
        } else if (type === CELL_OPEN_VENT) {
          p[c] = phiVent;
          div[c] = 0.0;
        } else if (type === CELL_FLUID) {
          const d = (
            u[this.idxU(i + 1, j)] - u[this.idxU(i, j)] +
            v[this.idxV(i, j + 1)] - v[this.idxV(i, j)]
          ) * invH;
          div[c] = d;
          const ad = Math.abs(d);
          if (ad > maxDivBefore) maxDivBefore = ad;
        } else {
          div[c] = 0.0;
        }
      }
    }
    this.stats.maxDivBefore = maxDivBefore;

    // 2. Red-Black SOR iteration for ∇²φ = div
    //    Discrete stencil at fluid cell (i,j):
    //      sum_{nb != solid} (φ_nb - φ_c) / h² = div_c
    //    => φ_c = (sum_{nb != solid} φ_nb - h² * div_c) / count_non_solid
    const omega = params.sorOmega;
    const iters = params.poissonIters;

    for (let iter = 0; iter < iters; iter++) {
      for (let color = 0; color < 2; color++) {
        for (let j = 1; j < ny - 1; j++) {
          const iStart = 1 + ((j + color) & 1);
          for (let i = iStart; i < nx - 1; i += 2) {
            const c = this.idx(i, j);
            if (this.cellType[c] !== CELL_FLUID) continue;

            let sum = 0;
            let count = 0;

            const cL = c - 1;
            const cR = c + 1;
            const cB = c - nx;
            const cT = c + nx;

            if (this.cellType[cL] !== CELL_SOLID) { sum += p[cL]; count++; }
            if (this.cellType[cR] !== CELL_SOLID) { sum += p[cR]; count++; }
            if (this.cellType[cB] !== CELL_SOLID) { sum += p[cB]; count++; }
            if (this.cellType[cT] !== CELL_SOLID) { sum += p[cT]; count++; }

            if (count > 0) {
              const target = (sum - h2 * div[c]) / count;
              p[c] = p[c] + omega * (target - p[c]);
            }
          }
        }
      }
    }

    // 3. Subtract pressure gradient from staggered faces
    for (let j = 0; j < ny; j++) {
      for (let i = 1; i < nx; i++) {
        const cL = this.idx(i - 1, j);
        const cR = this.idx(i, j);
        if (this.cellType[cL] !== CELL_SOLID && this.cellType[cR] !== CELL_SOLID) {
          u[this.idxU(i, j)] -= (p[cR] - p[cL]) * invH;
        }
      }
    }

    for (let j = 1; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const cB = this.idx(i, j - 1);
        const cT = this.idx(i, j);
        if (this.cellType[cB] !== CELL_SOLID && this.cellType[cT] !== CELL_SOLID) {
          v[this.idxV(i, j)] -= (p[cT] - p[cB]) * invH;
        }
      }
    }
  }

  /**
   * Compute vorticity and live numerical diagnostics (CFL, post-projection divergence, smoke budget).
   */
  computeVorticityAndStats(dt) {
    const { nx, ny, dx, u, v, s, geom } = this;
    const invH = 1.0 / dx;
    const inv2H = 0.5 / dx;

    let maxSpeed = 0;
    let maxDivAfter = 0;
    let smokeIn = 0;
    let smokeOut = 0;

    for (let j = 1; j < ny - 1; j++) {
      for (let i = 1; i < nx - 1; i++) {
        const c = this.idx(i, j);
        if (this.cellType[c] !== CELL_FLUID) {
          this.curl[c] = 0;
          continue;
        }

        const uC = 0.5 * (u[this.idxU(i, j)] + u[this.idxU(i + 1, j)]);
        const vC = 0.5 * (v[this.idxV(i, j)] + v[this.idxV(i, j + 1)]);
        const spd = Math.hypot(uC, vC);
        if (spd > maxSpeed) maxSpeed = spd;

        const d = Math.abs(
          (u[this.idxU(i + 1, j)] - u[this.idxU(i, j)] +
           v[this.idxV(i, j + 1)] - v[this.idxV(i, j)]) * invH
        );
        if (d > maxDivAfter) maxDivAfter = d;

        const vR = 0.5 * (v[this.idxV(i + 1, j)] + v[this.idxV(i + 1, j + 1)]);
        const vL = 0.5 * (v[this.idxV(i - 1, j)] + v[this.idxV(i - 1, j + 1)]);
        const uT = 0.5 * (u[this.idxU(i, j + 1)] + u[this.idxU(i + 1, j + 1)]);
        const uB = 0.5 * (u[this.idxU(i, j - 1)] + u[this.idxU(i + 1, j - 1)]);
        this.curl[c] = (vR - vL - (uT - uB)) * inv2H;

        if (i < geom.wallI) smokeIn += s[c];
        else smokeOut += s[c];
      }
    }

    const totalSmoke = smokeIn + smokeOut + 1e-6;
    this.stats.time += dt;
    this.stats.stepCount += 1;
    this.stats.maxSpeed = maxSpeed;
    this.stats.cfl = (maxSpeed * dt) / dx;
    this.stats.maxDivAfter = maxDivAfter;
    this.stats.smokeInside = (smokeIn / totalSmoke) * 100;
    this.stats.smokeOutside = (smokeOut / totalSmoke) * 100;
  }
}
