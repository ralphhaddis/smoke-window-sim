# Window smoke simulation

The simulated fluid dynamics of a moveable source releasing smoke near a window. You can change the window angle, temp parameters, add wind, or open a second vent and watch the exchange change.

## Run it locally

You need an Apple Silicon Mac, Python 3.9 or newer, and Apple Clang. The C solver uses Apple’s threading library and the build targets the M3 processor. If Clang is missing, run `xcode-select --install` and let the installation finish. Use a browser with WebGL2 for the 3D view.

From the repository folder, run the following commands in Terminal.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
python -m pycfd.server --grid fast
```

Open [the local viewer](http://127.0.0.1:8080/). The first launch compiles the C solver automatically. Press Ctrl+C in Terminal to stop the server. For later runs, activate the same environment and repeat the last command.


## What the solver computes

The model discretises the room, the window opening, and the air outside.Each cube holds a temperature, pressure, and smoke concentration. The solver stores air velocity on the faces between cubes. Multiplying the speed through a face by its area gives the volume of air crossing it per second. Air can enter or leave at the far edges of the outdoor region, where a damping layer reduces the influence of the artificial boundary.

The smoke field is a relative concentration. The source adds smoke and nudges the nearby air toward the source temperature. With the default settings, a warm plume rises because of that heat. The **Dense Smoke** preset also gives the smoke itself extra weight. The model prescribes a generic source and does not simulate combustion.

### Governing equations

**Airflow.** The solver treats air as incompressible. Each cell must pass on as much air as it receives. Temperature and smoke density affect the vertical force through the Boussinesq approximation.

$$
\nabla \cdot \mathbf{u} = 0
$$

$$
\frac{\partial \mathbf{u}}{\partial t}
+ (\mathbf{u}\cdot\nabla)\mathbf{u}
= -\nabla\phi + \nabla\cdot(2\nu_{\mathrm{eff}}\mathbf{S})
+ g(\beta\theta-\alpha_s c)\,\hat{\mathbf{z}} + \mathbf{f}_{\mathrm{jet}}
$$

Here $\mathbf{u}$ is air velocity. Reading the right-hand side from left to right gives four effects.

- **Pressure** pushes air from high to low pressure. The solver uses the pressure difference from still outdoor air at the same height, divided by a reference density ($\phi$).
- **Friction and mixing** transfer momentum between neighbouring regions. The effective viscosity $\nu_{\mathrm{eff}}$ includes turbulent mixing, and $\mathbf{S}$ is the local rate of strain.
- **Buoyancy** acts vertically. Air warmer than outdoors feels an upward force set by its excess temperature $\theta$ and thermal expansion coefficient $\beta$. Smoke concentration $c$ adds a downward force when its density coefficient $\alpha_s$ is positive. Here $g$ is gravitational acceleration.
- **The optional jet** pushes air near the smoke source in the chosen direction.

**Heat and smoke.** Both travel with the air and spread by mixing. For either field $q$, the solver uses a transport equation with an effective diffusivity $D_q$ and a local source $s_q$.

$$
\frac{\partial q}{\partial t} + \nabla\cdot(\mathbf{u}q)
= \nabla\cdot(D_q\nabla q) + s_q
$$

The emitter adds smoke fastest at its centre. If its local injection rate is $r$, the smoke source is $s_c=r$ and the temperature source is $s_\theta=r(\theta_{\mathrm{source}}-\theta)$. This lets the same emitter warm or cool the surrounding air.

### Numerical choices

- **Conservation.** A finite-volume method computes how much crosses each cell face. A two-stage Runge–Kutta method advances the fields in time. The step size adapts to air speed, diffusion, and buoyancy.
- **Turbulence.** The grid resolves large eddies. A Smagorinsky–Lilly model estimates mixing by smaller eddies and accounts for the effects of nearby walls and vertical density changes. A finer grid captures smaller features and costs more per step.
- **Pressure.** After predicting the velocity, the solver calculates a pressure correction to balance flow in every cell. It uses several grid resolutions to speed up that calculation. The main fields and pressure iteration use double precision, while the multigrid calculations use single precision.

## Use the solver from Python

The browser and Python use the same C solver in `csrc/`. The `pycfd/` package handles compilation, exposes the fields as NumPy arrays, and serves the viewer in `web/`. To get a flow measurement directly, run this in Python from the repository folder with the environment active.

```python
from pycfd import CFD3DSolver

with CFD3DSolver(grid="fast", preset="cold_day") as sim:
    metrics = sim.step_n(50)
    print(f"After {metrics['sim_time']:.2f} s of simulated time:")
    print(f"Outflow {1000 * metrics['q_out']:.1f} L/s")
    print(f"Inflow  {1000 * metrics['q_in']:.1f} L/s")
```

To check the implementation, run `python -m pycfd.verify`. Its eight checks include whether still air stays still, smoke is conserved in a sealed room, the window panel blocks flow, and the direction of exchange reverses between winter and summer.
