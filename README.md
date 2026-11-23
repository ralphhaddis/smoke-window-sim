# smoke-window-sim

Browser-based 2D fluid simulator modeling how smoke moves between an indoor room and the outdoors through an open window. Built with vanilla JavaScript and HTML5 Canvas (no external dependencies).

This is an initial first iteration focused on getting the core solver, boundary conditions, and interactive controls working cleanly.

## What's Implemented

- **Solver (`solver.js`)**: 2D incompressible Navier-Stokes with Boussinesq thermal buoyancy on a `160 x 96` Marker-and-Cell (MAC) staggered grid (3.0 m physical domain height).
  - Fractional-step Chorin projection: emitter injection -> Boussinesq buoyancy & smoke excess density + vorticity confinement -> advection (2nd-order clamped MacCormack or 1st-order Semi-Lagrangian with RK2 midpoint backtrace) -> implicit Gauss-Seidel diffusion (viscosity, temperature, smoke) -> Red-Black SOR pressure Poisson solve -> gradient subtraction.
  - Supports solid walls (no-penetration), open outdoor boundaries (Dirichlet pressure, wind inflow/outflow), and an optional rear vent/door with configurable indoor-outdoor $\Delta P$.
- **Renderer (`renderer.js`)**: Renders onto an offscreen pixel buffer scaled to the main canvas, with view modes for smoke, temperature, speed (`|u|`), vorticity, and pressure, plus an optional velocity vector overlay and draggable emitter handle.
- **UI & Presets (`index.html`, `main.js`, `style.css`)**: Live sliders for temperatures, smoke density, $\Delta P$, outdoor wind, jet speed/angle, room/window geometry (including an angled sash baffle), and numerical knobs (advection order, viscosity, vorticity confinement, SOR iterations). Includes five presets (cold day exchange, wind blowing in, cross-ventilation, cold dense smoke, hot summer day), custom wall drawing/erasing on the canvas, single-stepping, and live CFL / divergence / indoor-outdoor smoke diagnostics.

## Running Locally

Because `main.js` uses ES modules (`<script type="module">`), serve the folder over a local HTTP server rather than opening `index.html` directly from `file://`:

```sh
python3 -m http.server 8000
```

Then open `http://localhost:8000` in your browser.
