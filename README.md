# smoke-window-sim

3D incompressible Boussinesq Large Eddy Simulation (LES) modeling smoke transport and buoyancy-driven natural ventilation between a confined UK room and an open exterior atmosphere through an angled casement window.

The core CFD solver is written in multithreaded **C99** (Apple Silicon ARM64 NEON + GCD, strict FP64 physics with a mixed-precision FP32 Multigrid preconditioner), wrapped in **Python (`ctypes` + zero-copy NumPy views)**, with a zero-dependency local HTTP binary streamer and a minimal **WebGL2 volumetric + 2D cross-section** scientific frontend.

*(The earlier 2D JavaScript prototype is preserved in git history at commit `58b5b80`.)*

## Physics & Numerical Methods

- **Governing Equations (`csrc/advect_les.c`, `csrc/step.c`)**:
  - 3D Marker-and-Cell (MAC) staggered finite-volume grid in hydrostatic perturbation form ($\theta = T - T_{\text{out}}$, $\phi = (p - p_{\text{hyd,out}}(z))/\rho_0$, reference state $T_{\text{ref}} = T_{\text{out}}$), so quiescent ambient outdoor air sits in exact equilibrium ($\mathbf{b} \equiv 0$, $\phi = 0$).
  - 2nd-order flux-conservative TVD finite-volume advection (Van Leer limiter for momentum, Monotonized Central for temperature and smoke) with 2-stage hydrostatic-preserving RK2 / divergence-free SSP-RK2 and adaptive face-max 3D CFL timestepping.
  - **Smagorinsky–Lilly buoyancy-stratified LES** ($\text{Pr}_t = \text{Sc}_t = 0.7$) with Mason–Thomson wall damping, conservative variable-viscosity stress divergence $\nabla \cdot (2\nu_{\text{eff}}\mathbf{S})$, and Werner–Wengle wall shear.
- **Open Exterior & Geometry Boundary Conditions (`csrc/bc.c`, `csrc/grid_geom.c`)**:
  - **NIST FDS / Dong (2014) energy-stable Bernoulli OPEN boundary conditions** across the exterior far-field faces ($+X, \pm Y, +Z$) plus a thin 5-cell implicit Rayleigh sponge layer to prevent backflow blow-ups and boundary reflections.
  - Standard UK room (`3.6 × 3.0 × 2.5 m`), masonry reveal, optional rear doorway/vent hydrostatic Bernoulli reservoir ($\Delta P$), and outward-swinging UK casement sash (left-hung, right-hung, or top-hung awning, $0^\circ\text{–}90^\circ$) represented via **dual-segment face blocking** for strictly zero diagonal leakage.
- **Elliptic Pressure Solver (`csrc/poisson_mg.c`)**:
  - Strict FP64 Flexible Conjugate Gradient preconditioned by a symmetric **FP32 Face-Aware Galerkin Geometric Multigrid V-cycle** with bit-packed Level-0 stencils, ARM64 NEON SIMD kernels, fused Red-only residual/restriction, and BFS pure-Neumann nullspace handling.

## Quick Start

Requires Python 3.9+, `numpy`, and `clang` (macOS Apple Silicon / Clang).

```sh
# 1. Compile libcfd3d.dylib (also compiles automatically on first import)
python3 -m pycfd.build

# 2. Launch the live 3D simulation server
python3 -m pycfd.server --port 8080 --grid fast
```

Open **`http://127.0.0.1:8080/`** in your browser.
- **3D view**: Drag to orbit, Shift+Drag to pan, scroll to zoom.
- **2D cross-sections (`X–Z`, `X–Y`, `Y–Z`)**: Hover to probe `(x, y, z)` field/velocity values; click or drag directly on any slice to reposition the 3D smoke source.

## Python Scientific API & Verification Suite

Drive simulations directly from Python with zero-copy FP64 NumPy views:

```python
from pycfd import CFD3DSolver

with CFD3DSolver(grid="default", preset="cold_day", win_angle_deg=45.0) as sim:
    m = sim.step_n(100)
    smoke = sim.field_view("smoke")  # shape (nz, ny, nx), float64
    print(f"Q_out={m['q_out']*1000:.1f} L/s, z_NPP={m['z_npp']:.2f} m, div={m['max_div']:.2e}")
```

Run the 8-part physics, conservation, Linden exchange-flow, and long-horizon stability verification suite:

```sh
python3 -m pycfd.verify
```

## Project Structure

- `csrc/` — C99 3D CFD solver (`cfd3d.h`, `grid_geom.c`, `bc.c`, `advect_les.c`, `poisson_mg.c`, `step.c`, `slices.c`)
- `pycfd/` — Build helper (`build.py`), `ctypes`/NumPy wrapper (`solver.py`), stdlib binary HTTP server (`server.py`), V&V suite (`verify.py`)
- `web/` — Bare-bones WebGL2 volumetric raymarcher (`viewer3d.js`), 2D slice plots (`slices2d.js`), and UI (`index.html`, `style.css`, `app.js`)
