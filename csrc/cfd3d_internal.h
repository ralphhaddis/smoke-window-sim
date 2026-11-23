#ifndef CFD3D_INTERNAL_H
#define CFD3D_INTERNAL_H

#include "cfd3d.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dispatch/dispatch.h>

/* Physical constants */
#define CFD_G          9.80665
#define CFD_P_ATM      101325.0
#define CFD_R_AIR      287.05
#define CFD_NU0        1.5e-5
#define CFD_ALPHA0     2.1e-5
#define CFD_DIFF0      1.5e-5
#define CFD_KAPPA_VK   0.41
#define CFD_PR_T       0.7
#define CFD_SC_T       0.7
#define CFD_PI         3.14159265358979323846

/* Boundary condition types */
enum {
    BC_WALL   = 0,
    BC_INFLOW = 1,
    BC_OPEN   = 2,
    BC_VENT   = 3
};

/* Sponge bitmask bits */
enum {
    SPONGE_PX = 1 << 0,
    SPONGE_MY = 1 << 1,
    SPONGE_PY = 1 << 2,
    SPONGE_PZ = 1 << 3
};

/* Inline indexing helpers: C-contiguous (k, j, i) */
static inline size_t idx_c(int i, int j, int k, int nx, int ny) {
    return ((size_t)k * (size_t)ny + (size_t)j) * (size_t)nx + (size_t)i;
}

static inline size_t idx_u(int i, int j, int k, int nx, int ny) {
    return ((size_t)k * (size_t)ny + (size_t)j) * (size_t)(nx + 1) + (size_t)i;
}

static inline size_t idx_v(int i, int j, int k, int nx, int ny) {
    return ((size_t)k * (size_t)(ny + 1) + (size_t)j) * (size_t)nx + (size_t)i;
}

static inline size_t idx_w(int i, int j, int k, int nx, int ny) {
    return ((size_t)k * (size_t)ny + (size_t)j) * (size_t)nx + (size_t)i;
}

static inline double cfd_clamp(double v, double lo, double hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

static inline int cfd_iclamp(int v, int lo, int hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

#define CFD_MAX_CHUNKS 16
#define CFD_CACHE_PAD  16 /* 16 doubles = 128 bytes (Apple Silicon cache line) */

typedef struct MGLevel {
    int nx, ny, nz;
    size_t ncells;
    float *wx;       /* (nx+1)*ny*nz interior x-face weights (exact integers on L0) */
    float *wy;       /* nx*(ny+1)*nz interior y-face weights */
    float *wz;       /* nx*ny*(nz+1) interior z-face weights */
    float *wd;       /* nx*ny*nz Dirichlet boundary weights (freed after mg_build) */
    float *diag;     /* nx*ny*nz diagonal of A */
    float *inv_diag; /* nx*ny*nz 1/diag or 0 for solid */
    float *x;        /* nx*ny*nz work solution */
    float *b;        /* nx*ny*nz work rhs */

    /* Pure-Neumann components on this level (only needed on finest and coarsest) */
    int n_neumann;
    int *neumann_offsets;
    int *neumann_cells;
} MGLevel;

#define CFD_MAX_MG_LEVELS 10

struct CFD3D {
    CFDParams params;
    CFDGeomInfo geom;
    CFDMetrics metrics;

    int nx, ny, nz;
    double h, inv_h, inv_h2;
    size_t ncells, nu, nv, nw;

    /* Derived thermodynamic reference state */
    double t_out_k;
    double t_in_k;
    double theta_in; /* T_in - T_out [K] */
    double beta;     /* 1 / T_out [1/K] */
    double rho0;     /* Reference outdoor air density [kg/m^3] */

    /* Ambient wind vectors at z_ref */
    double wind_ux_ref;
    double wind_uy_ref;
    int bc_px, bc_my, bc_py, bc_pz;

    /* Geometry masks */
    uint8_t *cell_type; /* ncells: CFD_CELL_FLUID / CFD_CELL_SOLID */
    uint8_t *cell_zone; /* ncells: CFD_ZONE_* */
    uint8_t *kx;        /* nu: 0 blocked, 1 open */
    uint8_t *ky;        /* nv: 0 blocked, 1 open */
    uint8_t *kz;        /* nw: 0 blocked, 1 open */
    double  *wall_dist; /* ncells: 3D Euclidean/Chamfer distance to nearest solid [m] */

    /* Rayleigh sponge layer */
    double  *sponge_sigma; /* ncells: damping rate [1/s] */
    uint8_t *sponge_mask;  /* ncells: bitmask SPONGE_* */

    /* Primary FP64 fields */
    double *u;     /* nu */
    double *v;     /* nv */
    double *w;     /* nw */
    double *phi;   /* ncells */
    double *theta; /* ncells */
    double *smoke; /* ncells */

    /* LES viscosity & scratch */
    double *nu_sgs; /* ncells */

    /* RK2 scratch arrays */
    double *u0, *v0, *w0;
    double *ru0, *rv0, *rw0;
    double *ru1, *rv1, *rw1;
    double *theta0, *smoke0;
    double *rtheta, *rsmoke;

    /* Boundary Dirichlet pressure values on open/vent faces */
    double *phi_bc_mx; /* ny * nz (i = 0, BC_VENT) */
    double *phi_bc_px; /* ny * nz (i = nx, BC_OPEN) */
    double *phi_bc_my; /* nx * nz (j = 0, BC_OPEN) */
    double *phi_bc_py; /* nx * nz (j = ny, BC_OPEN) */
    double *phi_bc_pz; /* nx * ny (k = nz, BC_OPEN) */

    /* PCG scratch vectors (finest level) */
    double *pcg_b;
    double *pcg_r;
    double *pcg_z;
    double *pcg_p;
    double *pcg_ap;

    /* Precomputed Gaussian smoke emitter kernel */
    double *emit_kernel; /* ncells: normalized G(x) [1/m^3], sum(G)*h^3 = 1 */
    double  emit_wx, emit_wy, emit_wz; /* World coordinates of emitter */
    double  emit_sigma;
    double  emit_jx, emit_jy, emit_jz; /* Unit jet direction */
    int     emit_i0, emit_i1, emit_j0, emit_j1, emit_k0, emit_k1; /* Bounding box */

    /* Multigrid hierarchy */
    int n_mg_levels;
    MGLevel mg[CFD_MAX_MG_LEVELS];
    uint8_t *l0_mask; /* ncells: 6-bit open interior neighbor flags (-x,+x,-y,+y,-z,+z) */
    uint8_t *l0_diag; /* ncells: exact integer diagonal in [0, 12] */

    /* Time stepping state */
    double dt_prev;
    double sim_time;
    int    step_count;

    /* GCD queue & Apple Silicon chunked threading configuration */
    dispatch_queue_t queue;
    int n_threads;
    int mg_smooth_sweeps;
    int mg_coarse_sweeps;
};

/* Coarse-chunked GCD parallel-for over k in [0, k_count) */
typedef void (*cfd_k_chunk_fn)(void *ctx, int k0, int k1, int tid);

typedef struct CFDChunkDispatch {
    void *user_ctx;
    cfd_k_chunk_fn fn;
    int k_count;
    int n_chunks;
} CFDChunkDispatch;

static inline void cfd_chunk_trampoline(void *vctx, size_t tid) {
    const CFDChunkDispatch *d = (const CFDChunkDispatch *)vctx;
    int t = (int)tid;
    int k0 = (t * d->k_count) / d->n_chunks;
    int k1 = ((t + 1) * d->k_count) / d->n_chunks;
    if (k0 < k1) {
        d->fn(d->user_ctx, k0, k1, t);
    }
}

static inline int cfd_parallel_for_k(const CFD3D *sim, int k_count, int min_k_for_par,
                                     void *user_ctx, cfd_k_chunk_fn fn) {
    int nc = sim->n_threads;
    if (nc <= 1 || k_count < min_k_for_par) {
        fn(user_ctx, 0, k_count, 0);
        return 1;
    }
    if (nc > k_count) nc = k_count;
    if (nc > CFD_MAX_CHUNKS) nc = CFD_MAX_CHUNKS;
    CFDChunkDispatch d = { user_ctx, fn, k_count, nc };
    dispatch_apply_f((size_t)nc, sim->queue, &d, cfd_chunk_trampoline);
    return nc;
}

/* Inline wind profile helper */
static inline double cfd_wind_abl_factor(const CFD3D *sim, double z) {
    if (!sim->params.wind_use_abl) {
        return 1.0;
    }
    const double z0 = 0.05;
    double zref = (sim->params.wind_z_ref > 0.1) ? sim->params.wind_z_ref : 2.0;
    double z_eff = (z > 0.5 * sim->h) ? z : (0.5 * sim->h);
    return log(z_eff / z0 + 1.0) / log(zref / z0 + 1.0);
}

static inline void cfd_wind_at_z(const CFD3D *sim, double z, double *ux, double *uy) {
    double f = cfd_wind_abl_factor(sim, z);
    *ux = sim->wind_ux_ref * f;
    *uy = sim->wind_uy_ref * f;
}

/* grid_geom.c */
void cfd_update_thermo_and_wind(CFD3D *sim);
void cfd_build_geometry(CFD3D *sim);
void cfd_build_emitter(CFD3D *sim);

/* bc.c */
void cfd_build_sponge(CFD3D *sim);
void cfd_enforce_velocity_bcs(CFD3D *sim, double *u, double *v, double *w);
void cfd_enforce_inflow_bcs(CFD3D *sim, double *u, double *v, double *w);
void cfd_compute_open_vent_phi_bc(CFD3D *sim, const double *u, const double *v, const double *w);
void cfd_apply_sponge_velocity(CFD3D *sim, double *u, double *v, double *w, double dt);
void cfd_apply_sponge_theta(CFD3D *sim, double *theta, double dt);

/* advect_les.c */
void cfd_compute_les_viscosity(CFD3D *sim, const double *u, const double *v, const double *w,
                               const double *theta, const double *smoke, double *nu_sgs,
                               double *out_nu_max, double *out_n2_max, double *out_u_l1_max);
void cfd_compute_momentum_rhs(CFD3D *sim, const double *u, const double *v, const double *w,
                              const double *theta, const double *smoke, const double *nu_sgs,
                              double *ru, double *rv, double *rw);
void cfd_compute_scalar_rhs(CFD3D *sim, const double *u, const double *v, const double *w,
                            const double *theta, const double *smoke, const double *nu_sgs,
                            double *rtheta, double *rsmoke);
void cfd_scalar_stage1_step(CFD3D *sim, double dt, const double *u, const double *v, const double *w,
                            const double *theta0, const double *smoke0, const double *nu_sgs,
                            double *theta1, double *smoke1);

/* poisson_mg.c */
void cfd_mg_free(CFD3D *sim);
int  cfd_mg_build(CFD3D *sim);
int  cfd_poisson_solve(CFD3D *sim, double dt, const double *u_star, const double *v_star,
                       const double *w_star, double *phi, int *out_iters, double *out_rel_res);
void cfd_project_velocity(CFD3D *sim, double dt, const double *phi,
                          double *u, double *v, double *w, double *out_max_div);

#endif /* CFD3D_INTERNAL_H */
