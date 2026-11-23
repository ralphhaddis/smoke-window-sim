#include "cfd3d_internal.h"
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

void cfd_default_params(CFDParams *p) {
    if (!p) return;
    memset(p, 0, sizeof(*p));

    /* Default 96 x 64 x 64 at h = 0.06875 m -> 6.60 m x 4.40 m x 4.40 m */
    p->nx = 96;
    p->ny = 64;
    p->nz = 64;
    p->h  = 0.06875;

    /* Standard UK room */
    p->room_lx    = 3.60;
    p->room_ly    = 3.00;
    p->room_lz    = 2.50;
    p->wall_thick = 0.25;

    /* Standard UK casement window */
    p->win_width      = 0.90;
    p->win_height     = 1.20;
    p->win_sill       = 0.90;
    p->win_y_offset   = 0.0;
    p->win_hinge_mode = CFD_HINGE_LEFT;
    p->win_angle_deg  = 45.0;

    /* Rear door / vent (closed by default) */
    p->vent_enabled = 0;
    p->vent_y0      = 1.05;
    p->vent_y1      = 1.95;
    p->vent_z0      = 0.0;
    p->vent_z1      = 2.00;
    p->delta_p_pa   = 0.0;
    p->vent_k_loss  = 0.5;

    /* Typical UK cool day */
    p->temp_in_c    = 20.0;
    p->temp_out_c   = 5.0;
    p->wind_speed   = 0.0;
    p->wind_dir_deg = 0.0;
    p->wind_use_abl = 0;
    p->wind_z_ref   = 2.0;

    /* Smoke emitter */
    p->smoke_enabled      = 1;
    p->smoke_x            = 1.80;
    p->smoke_y            = 1.50;
    p->smoke_z            = 1.10;
    p->smoke_radius       = 0.18;
    p->smoke_temp_c       = 30.0;
    p->smoke_rate         = 0.08;
    p->smoke_alpha_s      = 0.0;
    p->smoke_jet_speed    = 0.0;
    p->smoke_jet_azim_deg = 0.0;
    p->smoke_jet_elev_deg = 0.0;

    /* Numerics & LES */
    p->cfl_max       = 0.45;
    p->dt_min        = 1e-4;
    p->dt_max        = 0.05;
    p->cs_smag       = 0.17;
    p->sponge_sigma  = 1.5;
    p->pcg_max_iters = 50;
    p->pcg_rel_tol   = 1e-6;
    p->pcg_div_tol   = 1e-5;
}

typedef struct DiagChunkOut {
    double sm_in;
    double sm_out;
    double sm_max_in;
    double th_sum_in;
    double ke_sum;
    double max_v2;
    double max_cfl_vel;
    int    n_in;
    double _pad[8]; /* Pad to >= 128 bytes (Apple Silicon cache line) */
} DiagChunkOut;

typedef struct DiagCtx {
    const CFD3D *sim;
    double h3;
    DiagChunkOut out[CFD_MAX_CHUNKS];
} DiagCtx;

static void diag_chunk_fn(void *vctx, int k0, int k1, int tid) {
    DiagCtx *ctx = (DiagCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny;
    const double h3 = ctx->h3;
    const uint8_t *restrict cz = sim->cell_zone;
    const double *restrict u = sim->u;
    const double *restrict v = sim->v;
    const double *restrict w = sim->w;
    const double *restrict sm_arr = sim->smoke;
    const double *restrict th_arr = sim->theta;

    double sm_in = 0.0, sm_out = 0.0, sm_max_in = 0.0;
    double th_sum_in = 0.0;
    double ke_sum = 0.0;
    double max_v2 = 0.0;
    double max_cfl_vel = 0.0;
    int n_in = 0;

    for (int k = k0; k < k1; ++k) {
        for (int j = 0; j < ny; ++j) {
            const size_t row_c  = idx_c(0, j, k, nx, ny);
            const size_t row_u  = idx_u(0, j, k, nx, ny);
            const size_t row_v0 = idx_v(0, j,     k, nx, ny);
            const size_t row_v1 = idx_v(0, j + 1, k, nx, ny);
            const size_t row_w0 = idx_w(0, j, k,     nx, ny);
            const size_t row_w1 = idx_w(0, j, k + 1, nx, ny);

            const uint8_t *restrict czr = cz + row_c;
            const double *restrict ur   = u + row_u;
            const double *restrict v0r  = v + row_v0;
            const double *restrict v1r  = v + row_v1;
            const double *restrict w0r  = w + row_w0;
            const double *restrict w1r  = w + row_w1;
            const double *restrict smr  = sm_arr + row_c;
            const double *restrict thr  = th_arr + row_c;

            for (int i = 0; i < nx; ++i) {
                uint8_t zone = czr[i];
                if (zone == CFD_ZONE_SOLID) continue;

                double uc = 0.5 * (ur[i]  + ur[i + 1]);
                double vc = 0.5 * (v0r[i] + v1r[i]);
                double wc = 0.5 * (w0r[i] + w1r[i]);
                double v2 = uc * uc + vc * vc + wc * wc;
                ke_sum += v2;
                if (v2 > max_v2) max_v2 = v2;
                double l1 = fabs(uc) + fabs(vc) + fabs(wc);
                if (l1 > max_cfl_vel) max_cfl_vel = l1;

                double sm = smr[i];
                if (zone == CFD_ZONE_INDOOR || zone == CFD_ZONE_REVEAL) {
                    sm_in += sm * h3;
                    if (zone == CFD_ZONE_INDOOR) {
                        if (sm > sm_max_in) sm_max_in = sm;
                        th_sum_in += thr[i];
                        n_in++;
                    }
                } else {
                    sm_out += sm * h3;
                }
            }
        }
    }

    ctx->out[tid].sm_in       = sm_in;
    ctx->out[tid].sm_out      = sm_out;
    ctx->out[tid].sm_max_in   = sm_max_in;
    ctx->out[tid].th_sum_in   = th_sum_in;
    ctx->out[tid].ke_sum      = ke_sum;
    ctx->out[tid].max_v2      = max_v2;
    ctx->out[tid].max_cfl_vel = max_cfl_vel;
    ctx->out[tid].n_in        = n_in;
}

static void compute_diagnostics(CFD3D *sim, double dt, double max_div,
                                int pcg_iters, double pcg_rel_res, double step_ms) {
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double h = sim->h;
    const double h2 = h * h;
    const double h3 = h * h * h;
    const CFDGeomInfo *g = &sim->geom;

    /* Window volume exchange across indoor reveal plane i = ix_wall */
    double q_out = 0.0, q_in = 0.0;
    int nw_k = g->k_w1 - g->k_w0;
    int nw_j = g->j_w1 - g->j_w0;
    double u_prof[512];
    double *u_bar = (nw_k <= 512) ? u_prof : (double *)malloc((size_t)nw_k * sizeof(double));

    for (int k = g->k_w0; k < g->k_w1; ++k) {
        double sum_u = 0.0;
        for (int j = g->j_w0; j < g->j_w1; ++j) {
            double uf = sim->u[idx_u(g->ix_wall, j, k, nx, ny)];
            sum_u += uf;
            if (uf > 0.0) q_out += uf * h2;
            else          q_in  -= uf * h2;
        }
        u_bar[k - g->k_w0] = (nw_j > 0) ? (sum_u / (double)nw_j) : 0.0;
    }

    /* Neutral pressure plane height z_NPP via linear zero-crossing of width-averaged u(z) */
    double z_npp = -1.0;
    for (int kk = 0; kk < nw_k - 1; ++kk) {
        double u0 = u_bar[kk];
        double u1 = u_bar[kk + 1];
        if ((u0 <= 0.0 && u1 > 0.0) || (u0 >= 0.0 && u1 < 0.0)) {
            double denom = u1 - u0;
            if (fabs(denom) > 1e-12) {
                double frac = -u0 / denom;
                z_npp = ((double)(g->k_w0 + kk) + 0.5 + frac) * h;
                break;
            }
        }
    }
    if (nw_k > 512) free(u_bar);

    /* Rear vent net inflow */
    double q_vent = 0.0;
    if (sim->params.vent_enabled) {
        for (int k = g->k_v0; k < g->k_v1; ++k) {
            for (int j = g->j_v0; j < g->j_v1; ++j) {
                size_t fu = idx_u(0, j, k, nx, ny);
                if (sim->kx[fu]) q_vent += sim->u[fu] * h2;
            }
        }
    }

    /* Parallel domain integrals: smoke mass, indoor mean temp, KE, max speed, CFL */
    DiagCtx dctx;
    dctx.sim = sim;
    dctx.h3  = h3;
    int nt = cfd_parallel_for_k(sim, nz, 8, &dctx, diag_chunk_fn);

    double sm_in = 0.0, sm_out = 0.0, sm_max_in = 0.0;
    double th_sum_in = 0.0, ke_sum = 0.0, max_v2 = 0.0, max_cfl_vel = 0.0;
    int n_in = 0;
    for (int t = 0; t < nt; ++t) {
        sm_in     += dctx.out[t].sm_in;
        sm_out    += dctx.out[t].sm_out;
        th_sum_in += dctx.out[t].th_sum_in;
        ke_sum    += dctx.out[t].ke_sum;
        n_in      += dctx.out[t].n_in;
        if (dctx.out[t].sm_max_in   > sm_max_in)   sm_max_in   = dctx.out[t].sm_max_in;
        if (dctx.out[t].max_v2      > max_v2)      max_v2      = dctx.out[t].max_v2;
        if (dctx.out[t].max_cfl_vel > max_cfl_vel) max_cfl_vel = dctx.out[t].max_cfl_vel;
    }

    CFDMetrics *m = &sim->metrics;
    m->sim_time         = sim->sim_time;
    m->dt               = dt;
    m->step_count       = sim->step_count;
    m->q_out            = q_out;
    m->q_in             = q_in;
    m->q_net            = q_out - q_in;
    m->q_vent           = q_vent;
    m->ach              = (g->room_volume > 1e-9) ? (3600.0 * fmax(q_out, q_in) / g->room_volume) : 0.0;
    m->z_npp            = z_npp;
    m->smoke_indoor     = sm_in;
    m->smoke_outdoor    = sm_out;
    m->smoke_max_indoor = sm_max_in;
    m->temp_mean_in_c   = sim->params.temp_out_c + ((n_in > 0) ? (th_sum_in / (double)n_in) : 0.0);
    m->kinetic_energy   = 0.5 * sim->rho0 * ke_sum * h3;
    m->max_speed        = sqrt(max_v2);
    m->cfl              = (dt > 0.0) ? (dt * max_cfl_vel * sim->inv_h) : 0.0;
    m->max_div          = max_div;
    m->pcg_iters        = pcg_iters;
    m->pcg_rel_res      = pcg_rel_res;
    m->step_ms          = step_ms;
}

CFD3D *cfd_create(const CFDParams *p) {
    if (!p || p->nx < 16 || p->ny < 16 || p->nz < 16 || p->h <= 0.0) {
        return NULL;
    }
    CFD3D *sim = (CFD3D *)calloc(1, sizeof(CFD3D));
    if (!sim) return NULL;

    sim->params = *p;
    sim->nx = p->nx; sim->ny = p->ny; sim->nz = p->nz;
    sim->h = p->h;
    sim->inv_h = 1.0 / p->h;
    sim->inv_h2 = sim->inv_h * sim->inv_h;

    sim->ncells = (size_t)sim->nx * (size_t)sim->ny * (size_t)sim->nz;
    sim->nu     = (size_t)(sim->nx + 1) * (size_t)sim->ny * (size_t)sim->nz;
    sim->nv     = (size_t)sim->nx * (size_t)(sim->ny + 1) * (size_t)sim->nz;
    sim->nw     = (size_t)sim->nx * (size_t)sim->ny * (size_t)(sim->nz + 1);

    sim->queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    sim->n_threads        = 4;  /* Matches Apple M3 4 Performance cores */
    sim->mg_smooth_sweeps = 2;  /* 2 R->B pre + 2 B->R post symmetric V-cycle */
    sim->mg_coarse_sweeps = 12; /* 12 R->B + 12 B->R coarsest sweeps */

    const char *env_t = getenv("CFD_THREADS");
    if (env_t && atoi(env_t) > 0) {
        sim->n_threads = cfd_iclamp(atoi(env_t), 1, CFD_MAX_CHUNKS);
    }
    const char *env_s = getenv("CFD_MG_SMOOTH");
    if (env_s && atoi(env_s) > 0) {
        sim->mg_smooth_sweeps = cfd_iclamp(atoi(env_s), 1, 4);
    }
    const char *env_c = getenv("CFD_MG_COARSE");
    if (env_c && atoi(env_c) > 0) {
        sim->mg_coarse_sweeps = cfd_iclamp(atoi(env_c), 2, 60);
    }

    /* Allocate masks & fields */
    sim->cell_type    = (uint8_t *)calloc(sim->ncells, sizeof(uint8_t));
    sim->cell_zone    = (uint8_t *)calloc(sim->ncells, sizeof(uint8_t));
    sim->kx           = (uint8_t *)calloc(sim->nu,     sizeof(uint8_t));
    sim->ky           = (uint8_t *)calloc(sim->nv,     sizeof(uint8_t));
    sim->kz           = (uint8_t *)calloc(sim->nw,     sizeof(uint8_t));
    sim->wall_dist    = (double  *)calloc(sim->ncells, sizeof(double));
    sim->sponge_sigma = (double  *)calloc(sim->ncells, sizeof(double));
    sim->sponge_mask  = (uint8_t *)calloc(sim->ncells, sizeof(uint8_t));

    sim->u     = (double *)calloc(sim->nu,     sizeof(double));
    sim->v     = (double *)calloc(sim->nv,     sizeof(double));
    sim->w     = (double *)calloc(sim->nw,     sizeof(double));
    sim->phi   = (double *)calloc(sim->ncells, sizeof(double));
    sim->theta = (double *)calloc(sim->ncells, sizeof(double));
    sim->smoke = (double *)calloc(sim->ncells, sizeof(double));
    sim->nu_sgs = (double *)calloc(sim->ncells, sizeof(double));

    sim->u0     = (double *)calloc(sim->nu,     sizeof(double));
    sim->v0     = (double *)calloc(sim->nv,     sizeof(double));
    sim->w0     = (double *)calloc(sim->nw,     sizeof(double));
    sim->ru0    = (double *)calloc(sim->nu,     sizeof(double));
    sim->rv0    = (double *)calloc(sim->nv,     sizeof(double));
    sim->rw0    = (double *)calloc(sim->nw,     sizeof(double));
    sim->ru1    = (double *)calloc(sim->nu,     sizeof(double));
    sim->rv1    = (double *)calloc(sim->nv,     sizeof(double));
    sim->rw1    = (double *)calloc(sim->nw,     sizeof(double));
    sim->theta0 = (double *)calloc(sim->ncells, sizeof(double));
    sim->smoke0 = (double *)calloc(sim->ncells, sizeof(double));
    sim->rtheta = (double *)calloc(sim->ncells, sizeof(double));
    sim->rsmoke = (double *)calloc(sim->ncells, sizeof(double));

    sim->phi_bc_mx = (double *)calloc((size_t)sim->ny * (size_t)sim->nz, sizeof(double));
    sim->phi_bc_px = (double *)calloc((size_t)sim->ny * (size_t)sim->nz, sizeof(double));
    sim->phi_bc_my = (double *)calloc((size_t)sim->nx * (size_t)sim->nz, sizeof(double));
    sim->phi_bc_py = (double *)calloc((size_t)sim->nx * (size_t)sim->nz, sizeof(double));
    sim->phi_bc_pz = (double *)calloc((size_t)sim->nx * (size_t)sim->ny, sizeof(double));

    sim->pcg_b  = (double *)calloc(sim->ncells, sizeof(double));
    sim->pcg_r  = (double *)calloc(sim->ncells, sizeof(double));
    sim->pcg_z  = (double *)calloc(sim->ncells, sizeof(double));
    sim->pcg_p  = (double *)calloc(sim->ncells, sizeof(double));
    sim->pcg_ap = (double *)calloc(sim->ncells, sizeof(double));

    sim->emit_kernel = (double *)calloc(sim->ncells, sizeof(double));

    if (!sim->cell_type || !sim->cell_zone || !sim->kx || !sim->ky || !sim->kz ||
        !sim->wall_dist || !sim->sponge_sigma || !sim->sponge_mask ||
        !sim->u || !sim->v || !sim->w || !sim->phi || !sim->theta || !sim->smoke ||
        !sim->nu_sgs || !sim->u0 || !sim->v0 || !sim->w0 ||
        !sim->ru0 || !sim->rv0 || !sim->rw0 || !sim->ru1 || !sim->rv1 || !sim->rw1 ||
        !sim->theta0 || !sim->smoke0 || !sim->rtheta || !sim->rsmoke ||
        !sim->phi_bc_mx || !sim->phi_bc_px || !sim->phi_bc_my || !sim->phi_bc_py || !sim->phi_bc_pz ||
        !sim->pcg_b || !sim->pcg_r || !sim->pcg_z || !sim->pcg_p || !sim->pcg_ap ||
        !sim->emit_kernel) {
        cfd_destroy(sim);
        return NULL;
    }

    cfd_update_thermo_and_wind(sim);
    cfd_build_geometry(sim);
    cfd_build_sponge(sim);
    cfd_build_emitter(sim);
    if (cfd_mg_build(sim) != 0) {
        cfd_destroy(sim);
        return NULL;
    }

    cfd_reset(sim);
    return sim;
}

void cfd_destroy(CFD3D *sim) {
    if (!sim) return;
    cfd_mg_free(sim);
    free(sim->cell_type);
    free(sim->cell_zone);
    free(sim->kx);
    free(sim->ky);
    free(sim->kz);
    free(sim->wall_dist);
    free(sim->sponge_sigma);
    free(sim->sponge_mask);
    free(sim->u);
    free(sim->v);
    free(sim->w);
    free(sim->phi);
    free(sim->theta);
    free(sim->smoke);
    free(sim->nu_sgs);
    free(sim->u0);
    free(sim->v0);
    free(sim->w0);
    free(sim->ru0);
    free(sim->rv0);
    free(sim->rw0);
    free(sim->ru1);
    free(sim->rv1);
    free(sim->rw1);
    free(sim->theta0);
    free(sim->smoke0);
    free(sim->rtheta);
    free(sim->rsmoke);
    free(sim->phi_bc_mx);
    free(sim->phi_bc_px);
    free(sim->phi_bc_my);
    free(sim->phi_bc_py);
    free(sim->phi_bc_pz);
    free(sim->pcg_b);
    free(sim->pcg_r);
    free(sim->pcg_z);
    free(sim->pcg_p);
    free(sim->pcg_ap);
    free(sim->emit_kernel);
    free(sim);
}

void cfd_reset(CFD3D *sim) {
    if (!sim) return;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double h = sim->h;

    memset(sim->u,      0, sim->nu     * sizeof(double));
    memset(sim->v,      0, sim->nv     * sizeof(double));
    memset(sim->w,      0, sim->nw     * sizeof(double));
    memset(sim->phi,    0, sim->ncells * sizeof(double));
    memset(sim->theta,  0, sim->ncells * sizeof(double));
    memset(sim->smoke,  0, sim->ncells * sizeof(double));
    memset(sim->nu_sgs, 0, sim->ncells * sizeof(double));

    memset(sim->phi_bc_mx, 0, (size_t)ny * (size_t)nz * sizeof(double));
    memset(sim->phi_bc_px, 0, (size_t)ny * (size_t)nz * sizeof(double));
    memset(sim->phi_bc_my, 0, (size_t)nx * (size_t)nz * sizeof(double));
    memset(sim->phi_bc_py, 0, (size_t)nx * (size_t)nz * sizeof(double));
    memset(sim->phi_bc_pz, 0, (size_t)nx * (size_t)ny * sizeof(double));

    /* Set initial indoor temperature perturbation theta = T_in - T_out
     * and exact initial indoor hydrostatic perturbation pressure phi^0(z) */
    double z_ref_hyd = sim->params.vent_enabled
        ? 0.5 * (sim->geom.vent_z0 + sim->geom.vent_z1)
        : 0.5 * (sim->geom.win_z0 + sim->geom.win_z1);
    double dp_over_rho = sim->params.vent_enabled ? (sim->params.delta_p_pa / sim->rho0) : 0.0;
    double g_beta_dT = CFD_G * sim->beta * sim->theta_in;

    for (int k = 0; k < nz; ++k) {
        double zc = (k + 0.5) * h;
        double phi_hyd = dp_over_rho + g_beta_dT * (zc - z_ref_hyd);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t c = idx_c(i, j, k, nx, ny);
                uint8_t zone = sim->cell_zone[c];
                if (zone == CFD_ZONE_INDOOR || zone == CFD_ZONE_REVEAL) {
                    sim->theta[c] = sim->theta_in;
                    sim->phi[c]   = phi_hyd;
                } else {
                    sim->theta[c] = 0.0;
                    sim->phi[c]   = 0.0;
                }
            }
        }
    }

    /* Seed initial open/vent boundary pressures to match phi^0 */
    cfd_compute_open_vent_phi_bc(sim, sim->u, sim->v, sim->w);

    /* If ambient wind is active, seed exterior domain with U_inf(z) */
    if (sim->params.wind_speed > 1e-3) {
        int ix_ext = sim->geom.ix_ext;
        for (int k = 0; k < nz; ++k) {
            double ux, uy;
            cfd_wind_at_z(sim, (k + 0.5) * h, &ux, &uy);
            for (int j = 0; j < ny; ++j) {
                for (int i = ix_ext + 1; i <= nx; ++i) {
                    size_t fu = idx_u(i, j, k, nx, ny);
                    if (sim->kx[fu]) sim->u[fu] = ux;
                }
            }
            for (int j = 0; j <= ny; ++j) {
                for (int i = ix_ext; i < nx; ++i) {
                    size_t fv = idx_v(i, j, k, nx, ny);
                    if (sim->ky[fv]) sim->v[fv] = uy;
                }
            }
        }
    }
    cfd_enforce_velocity_bcs(sim, sim->u, sim->v, sim->w);

    sim->sim_time = 0.0;
    sim->step_count = 0;
    sim->dt_prev = 0.0;

    compute_diagnostics(sim, 0.0, 0.0, 0, 0.0, 0.0);
}

int cfd_update_params(CFD3D *sim, const CFDParams *p) {
    if (!sim || !p) return -1;
    if (p->nx != sim->nx || p->ny != sim->ny || p->nz != sim->nz || fabs(p->h - sim->h) > 1e-12) {
        return -1; /* Grid dimensions require cfd_create */
    }

    const CFDParams old = sim->params;
    int geom_changed =
        (old.room_lx != p->room_lx) || (old.room_ly != p->room_ly) ||
        (old.room_lz != p->room_lz) || (old.wall_thick != p->wall_thick) ||
        (old.win_width != p->win_width) || (old.win_height != p->win_height) ||
        (old.win_sill != p->win_sill) || (old.win_y_offset != p->win_y_offset) ||
        (old.win_hinge_mode != p->win_hinge_mode) || (old.win_angle_deg != p->win_angle_deg) ||
        (old.vent_enabled != p->vent_enabled) ||
        (old.vent_y0 != p->vent_y0) || (old.vent_y1 != p->vent_y1) ||
        (old.vent_z0 != p->vent_z0) || (old.vent_z1 != p->vent_z1);

    int old_bc_px = sim->bc_px, old_bc_my = sim->bc_my, old_bc_py = sim->bc_py;
    int sponge_changed = (old.sponge_sigma != p->sponge_sigma);

    double old_t_out_k = sim->t_out_k;
    double old_t_in_c  = old.temp_in_c;
    sim->params = *p;
    cfd_update_thermo_and_wind(sim);
    double dT_out = sim->t_out_k - old_t_out_k;
    double dT_in  = p->temp_in_c - old_t_in_c;
    if (fabs(dT_out) > 1e-12 || fabs(dT_in) > 1e-12) {
        for (size_t c = 0; c < sim->ncells; ++c) {
            if (sim->cell_type[c] != CFD_CELL_FLUID) continue;
            uint8_t zc = sim->cell_zone[c];
            if (zc == CFD_ZONE_INDOOR) {
                sim->theta[c] += dT_in - dT_out;
            } else if (zc == CFD_ZONE_REVEAL) {
                sim->theta[c] -= dT_out;
            }
        }
    }
    int bc_changed = (old_bc_px != sim->bc_px) || (old_bc_my != sim->bc_my) || (old_bc_py != sim->bc_py);

    if (geom_changed) {
        cfd_build_geometry(sim);
        for (size_t c = 0; c < sim->ncells; ++c) {
            if (sim->cell_type[c] == CFD_CELL_SOLID) {
                sim->phi[c] = 0.0;
                sim->theta[c] = 0.0;
                sim->smoke[c] = 0.0;
                sim->nu_sgs[c] = 0.0;
            }
        }
    }
    if (geom_changed || bc_changed || sponge_changed) {
        cfd_build_sponge(sim);
    }
    if (geom_changed || bc_changed) {
        if (cfd_mg_build(sim) != 0) return -1;
    }

    cfd_build_emitter(sim);
    cfd_enforce_velocity_bcs(sim, sim->u, sim->v, sim->w);
    return 0;
}

/* ========================================================================= */
/* RK2 State Save, Stage 1 & Stage 2 Parallel Update Kernels                 */
/* ========================================================================= */

static void save_state_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    CFD3D *sim = (CFD3D *)vctx;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const size_t nk = (size_t)(k1 - k0);
    const size_t slab_c = (size_t)nx * (size_t)ny;
    const size_t slab_u = (size_t)(nx + 1) * (size_t)ny;
    const size_t slab_v = (size_t)nx * (size_t)(ny + 1);
    const size_t slab_w = slab_c;

    memcpy(sim->u0     + (size_t)k0 * slab_u, sim->u     + (size_t)k0 * slab_u, nk * slab_u * sizeof(double));
    memcpy(sim->v0     + (size_t)k0 * slab_v, sim->v     + (size_t)k0 * slab_v, nk * slab_v * sizeof(double));
    memcpy(sim->w0     + (size_t)k0 * slab_w, sim->w     + (size_t)k0 * slab_w, nk * slab_w * sizeof(double));
    memcpy(sim->theta0 + (size_t)k0 * slab_c, sim->theta + (size_t)k0 * slab_c, nk * slab_c * sizeof(double));
    memcpy(sim->smoke0 + (size_t)k0 * slab_c, sim->smoke + (size_t)k0 * slab_c, nk * slab_c * sizeof(double));
    if (k1 == nz) {
        memcpy(sim->w0 + (size_t)nz * slab_w, sim->w + (size_t)nz * slab_w, slab_w * sizeof(double));
    }
}

typedef struct Stage1Ctx {
    const CFD3D *sim;
    double dt;
    const double *u0, *v0, *w0;
    const double *ru0, *rv0, *rw0;
    const double *phi;
    double *u1, *v1, *w1;
} Stage1Ctx;

static void rk2_stage1_uvw_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    const Stage1Ctx *ctx = (const Stage1Ctx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double dt = ctx->dt;
    const double inv_h = sim->inv_h;
    const double inv_h2_b = 2.0 * inv_h;
    const double *restrict phi = ctx->phi;
    const uint8_t *restrict kx = sim->kx;
    const uint8_t *restrict ky = sim->ky;
    const uint8_t *restrict kz = sim->kz;
    const int open_px = (sim->bc_px == BC_OPEN);
    const int open_my = (sim->bc_my == BC_OPEN);
    const int open_py = (sim->bc_py == BC_OPEN);
    const int open_pz = (sim->bc_pz == BC_OPEN);

    for (int k = k0; k < k1; ++k) {
        /* 1. u-faces (unit stride in i) */
        for (int j = 0; j < ny; ++j) {
            const size_t row_c = idx_c(0, j, k, nx, ny);
            const size_t row_u = idx_u(0, j, k, nx, ny);
            const double *restrict pr  = phi + row_c;
            const double *restrict u0r = ctx->u0 + row_u;
            const double *restrict rur = ctx->ru0 + row_u;
            const uint8_t *restrict kxr = kx + row_u;
            double *restrict u1r = ctx->u1 + row_u;

            if (kxr[0]) {
                double dp = inv_h2_b * (pr[0] - sim->phi_bc_mx[(size_t)k * (size_t)ny + (size_t)j]);
                u1r[0] = u0r[0] + dt * (rur[0] - dp);
            } else {
                u1r[0] = 0.0;
            }

            for (int i = 1; i < nx; ++i) {
                double dp = inv_h * (pr[i] - pr[i - 1]);
                u1r[i] = kxr[i] ? (u0r[i] + dt * (rur[i] - dp)) : 0.0;
            }

            if (kxr[nx] && open_px) {
                double dp = inv_h2_b * (sim->phi_bc_px[(size_t)k * (size_t)ny + (size_t)j] - pr[nx - 1]);
                u1r[nx] = u0r[nx] + dt * (rur[nx] - dp);
            } else if (!kxr[nx]) {
                u1r[nx] = 0.0;
            }
        }

        /* 2. v-faces (100% unit-stride: j outer, i inner!) */
        {
            const size_t row_c0 = idx_c(0, 0, k, nx, ny);
            const size_t row_v0 = idx_v(0, 0, k, nx, ny);
            const double *restrict pr0   = phi + row_c0;
            const double *restrict bc_my = sim->phi_bc_my + (size_t)k * (size_t)nx;
            const double *restrict v0r0  = ctx->v0 + row_v0;
            const double *restrict rvr0  = ctx->rv0 + row_v0;
            const uint8_t *restrict kyr0 = ky + row_v0;
            double *restrict v1r0        = ctx->v1 + row_v0;
            if (open_my) {
                for (int i = 0; i < nx; ++i) {
                    double dp = inv_h2_b * (pr0[i] - bc_my[i]);
                    v1r0[i] = kyr0[i] ? (v0r0[i] + dt * (rvr0[i] - dp)) : 0.0;
                }
            } else {
                for (int i = 0; i < nx; ++i) {
                    if (!kyr0[i]) v1r0[i] = 0.0;
                }
            }

            for (int j = 1; j < ny; ++j) {
                const size_t row_c  = idx_c(0, j,     k, nx, ny);
                const size_t row_cm = idx_c(0, j - 1, k, nx, ny);
                const size_t row_v  = idx_v(0, j,     k, nx, ny);
                const double *restrict pr   = phi + row_c;
                const double *restrict prm  = phi + row_cm;
                const double *restrict v0r  = ctx->v0 + row_v;
                const double *restrict rvr  = ctx->rv0 + row_v;
                const uint8_t *restrict kyr = ky + row_v;
                double *restrict v1r        = ctx->v1 + row_v;
                for (int i = 0; i < nx; ++i) {
                    double dp = inv_h * (pr[i] - prm[i]);
                    v1r[i] = kyr[i] ? (v0r[i] + dt * (rvr[i] - dp)) : 0.0;
                }
            }

            const size_t row_cn = idx_c(0, ny - 1, k, nx, ny);
            const size_t row_vn = idx_v(0, ny,     k, nx, ny);
            const double *restrict prn   = phi + row_cn;
            const double *restrict bc_py = sim->phi_bc_py + (size_t)k * (size_t)nx;
            const double *restrict v0rn  = ctx->v0 + row_vn;
            const double *restrict rvrn  = ctx->rv0 + row_vn;
            const uint8_t *restrict kyrn = ky + row_vn;
            double *restrict v1rn        = ctx->v1 + row_vn;
            if (open_py) {
                for (int i = 0; i < nx; ++i) {
                    double dp = inv_h2_b * (bc_py[i] - prn[i]);
                    v1rn[i] = kyrn[i] ? (v0rn[i] + dt * (rvrn[i] - dp)) : 0.0;
                }
            } else {
                for (int i = 0; i < nx; ++i) {
                    if (!kyrn[i]) v1rn[i] = 0.0;
                }
            }
        }

        /* 3. w-faces for slab k */
        if (k == 0) {
            memset(&ctx->w1[idx_w(0, 0, 0, nx, ny)], 0, (size_t)nx * (size_t)ny * sizeof(double));
        } else {
            for (int j = 0; j < ny; ++j) {
                const size_t row_c  = idx_c(0, j, k,     nx, ny);
                const size_t row_cm = idx_c(0, j, k - 1, nx, ny);
                const size_t row_w  = idx_w(0, j, k,     nx, ny);
                const double *restrict pr   = phi + row_c;
                const double *restrict prm  = phi + row_cm;
                const double *restrict w0r  = ctx->w0 + row_w;
                const double *restrict rwr  = ctx->rw0 + row_w;
                const uint8_t *restrict kzr = kz + row_w;
                double *restrict w1r        = ctx->w1 + row_w;
                for (int i = 0; i < nx; ++i) {
                    double dp = inv_h * (pr[i] - prm[i]);
                    w1r[i] = kzr[i] ? (w0r[i] + dt * (rwr[i] - dp)) : 0.0;
                }
            }
        }
    }

    /* Top boundary w-face k = nz handled by last chunk */
    if (k1 == nz) {
        for (int j = 0; j < ny; ++j) {
            const size_t row_cn = idx_c(0, j, nz - 1, nx, ny);
            const size_t row_wn = idx_w(0, j, nz,     nx, ny);
            const double *restrict prn   = phi + row_cn;
            const double *restrict bc_pz = sim->phi_bc_pz + (size_t)j * (size_t)nx;
            const double *restrict w0rn  = ctx->w0 + row_wn;
            const double *restrict rwrn  = ctx->rw0 + row_wn;
            const uint8_t *restrict kzr  = kz + row_wn;
            double *restrict w1rn        = ctx->w1 + row_wn;
            if (open_pz) {
                for (int i = 0; i < nx; ++i) {
                    double dp = inv_h2_b * (bc_pz[i] - prn[i]);
                    w1rn[i] = kzr[i] ? (w0rn[i] + dt * (rwrn[i] - dp)) : 0.0;
                }
            } else {
                for (int i = 0; i < nx; ++i) {
                    w1rn[i] = 0.0;
                }
            }
        }
    }
}

typedef struct Stage2MomCtx {
    const CFD3D *sim;
    double half_dt;
} Stage2MomCtx;

static void rk2_stage2_mom_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    const Stage2MomCtx *ctx = (const Stage2MomCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double hdt = ctx->half_dt;
    const size_t nk = (size_t)(k1 - k0);
    const size_t slab_u = (size_t)(nx + 1) * (size_t)ny;
    const size_t slab_v = (size_t)nx * (size_t)(ny + 1);
    const size_t slab_w = (size_t)nx * (size_t)ny;

    /* u */
    {
        const size_t off = (size_t)k0 * slab_u;
        const size_t len = nk * slab_u;
        const uint8_t *restrict kx = sim->kx + off;
        const double *restrict u0  = sim->u0 + off;
        const double *restrict ru0 = sim->ru0 + off;
        const double *restrict ru1 = sim->ru1 + off;
        double *restrict u         = sim->u + off;
        for (size_t i = 0; i < len; ++i) {
            u[i] = kx[i] ? (u0[i] + hdt * (ru0[i] + ru1[i])) : 0.0;
        }
    }
    /* v */
    {
        const size_t off = (size_t)k0 * slab_v;
        const size_t len = nk * slab_v;
        const uint8_t *restrict ky = sim->ky + off;
        const double *restrict v0  = sim->v0 + off;
        const double *restrict rv0 = sim->rv0 + off;
        const double *restrict rv1 = sim->rv1 + off;
        double *restrict v         = sim->v + off;
        for (size_t i = 0; i < len; ++i) {
            v[i] = ky[i] ? (v0[i] + hdt * (rv0[i] + rv1[i])) : 0.0;
        }
    }
    /* w */
    {
        const size_t off = (size_t)k0 * slab_w;
        const size_t len = (nk + (size_t)(k1 == nz ? 1 : 0)) * slab_w;
        const uint8_t *restrict kz = sim->kz + off;
        const double *restrict w0  = sim->w0 + off;
        const double *restrict rw0 = sim->rw0 + off;
        const double *restrict rw1 = sim->rw1 + off;
        double *restrict w         = sim->w + off;
        for (size_t i = 0; i < len; ++i) {
            w[i] = kz[i] ? (w0[i] + hdt * (rw0[i] + rw1[i])) : 0.0;
        }
    }
}

typedef struct Stage2ScalCtx {
    const CFD3D *sim;
    double dt;
} Stage2ScalCtx;

static void rk2_stage2_scal_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    const Stage2ScalCtx *ctx = (const Stage2ScalCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny;
    const double dt = ctx->dt;
    const int do_sponge = (sim->params.sponge_sigma > 0.0);
    const size_t slab = (size_t)nx * (size_t)ny;
    const size_t off = (size_t)k0 * slab;
    const size_t len = (size_t)(k1 - k0) * slab;

    const uint8_t *restrict ct  = sim->cell_type + off;
    const double *restrict th0  = sim->theta0 + off;
    const double *restrict sm0  = sim->smoke0 + off;
    const double *restrict rth  = sim->rtheta + off;
    const double *restrict rsm  = sim->rsmoke + off;
    const double *restrict sig  = sim->sponge_sigma + off;
    double *restrict th         = sim->theta + off;
    double *restrict sm         = sim->smoke + off;

    if (do_sponge) {
        for (size_t i = 0; i < len; ++i) {
            if (ct[i] != CFD_CELL_FLUID) continue;
            double th2 = 0.5 * (th0[i] + th[i] + dt * rth[i]);
            double s = sig[i];
            if (s > 0.0) th2 /= (1.0 + dt * s);
            th[i] = th2;
            double sm2 = 0.5 * (sm0[i] + sm[i] + dt * rsm[i]);
            sm[i] = (sm2 > 0.0) ? sm2 : 0.0;
        }
    } else {
        for (size_t i = 0; i < len; ++i) {
            if (ct[i] != CFD_CELL_FLUID) continue;
            th[i] = 0.5 * (th0[i] + th[i] + dt * rth[i]);
            double sm2 = 0.5 * (sm0[i] + sm[i] + dt * rsm[i]);
            sm[i] = (sm2 > 0.0) ? sm2 : 0.0;
        }
    }
}

static double compute_adaptive_dt_from_bounds(const CFD3D *sim, double nu_max, double n2_max, double u_l1_max) {
    const double h = sim->h;
    const CFDParams *p = &sim->params;

    /* Include ambient wind speed & jet speed in initial velocity bound */
    double u_ref = fmax(u_l1_max, p->wind_speed);
    if (p->smoke_enabled && p->smoke_jet_speed > u_ref) u_ref = p->smoke_jet_speed;

    double dt_adv  = p->cfl_max * h / fmax(u_ref, 1e-6);
    double dt_diff = 0.075 * h * h / fmax(nu_max, 1e-12);
    double n_bv    = sqrt(fmax(n2_max, 0.0));
    double dt_buoy = 0.5 / fmax(n_bv, 1e-6);

    double dt = p->dt_max;
    if (dt_adv  < dt) dt = dt_adv;
    if (dt_diff < dt) dt = dt_diff;
    if (dt_buoy < dt) dt = dt_buoy;

    if (sim->dt_prev > 0.0) {
        double dt_grow = 1.15 * sim->dt_prev;
        if (dt > dt_grow) dt = dt_grow;
    }
    dt = cfd_clamp(dt, p->dt_min, p->dt_max);
    return dt;
}

static int cfd_step_internal(CFD3D *sim, int do_diagnostics, CFDMetrics *out_metrics) {
    if (!sim) return -1;
    double t_start = do_diagnostics ? now_ms() : 0.0;

    /* 1. Compute LES viscosity + velocity L1 bound at t^n, then adaptive dt */
    double nu_max = CFD_ALPHA0, n2_max = 0.0, u_l1_max = 0.0;
    cfd_compute_les_viscosity(sim, sim->u, sim->v, sim->w, sim->theta, sim->smoke,
                              sim->nu_sgs, &nu_max, &n2_max, &u_l1_max);
    double dt = compute_adaptive_dt_from_bounds(sim, nu_max, n2_max, u_l1_max);

    /* 2. Save state at t^n (parallelized across chunks) */
    cfd_parallel_for_k(sim, sim->nz, 8, sim, save_state_chunk_fn);

    /* 3. Momentum Stage 1: R^n and u^(1) = u^n + dt*(R^n - grad phi^n) */
    cfd_compute_momentum_rhs(sim, sim->u0, sim->v0, sim->w0,
                             sim->theta0, sim->smoke0, sim->nu_sgs,
                             sim->ru0, sim->rv0, sim->rw0);

    Stage1Ctx s1 = {
        sim, dt, sim->u0, sim->v0, sim->w0,
        sim->ru0, sim->rv0, sim->rw0, sim->phi,
        sim->u, sim->v, sim->w
    };
    cfd_parallel_for_k(sim, sim->nz, 8, &s1, rk2_stage1_uvw_chunk_fn);
    cfd_enforce_inflow_bcs(sim, sim->u, sim->v, sim->w);

    /* 4. Scalar Stage 1 (fused RHS + Euler update using divergence-free u^n) */
    cfd_scalar_stage1_step(sim, dt, sim->u0, sim->v0, sim->w0,
                           sim->theta0, sim->smoke0, sim->nu_sgs,
                           sim->theta, sim->smoke);

    /* 5. Momentum Stage 2: u* = u^n + 0.5*dt*(R^n + R^(1)), reusing nu_sgs from Stage 1 (FDS) */
    cfd_compute_momentum_rhs(sim, sim->u, sim->v, sim->w,
                             sim->theta, sim->smoke, sim->nu_sgs,
                             sim->ru1, sim->rv1, sim->rw1);

    Stage2MomCtx s2m = { sim, 0.5 * dt };
    cfd_parallel_for_k(sim, sim->nz, 8, &s2m, rk2_stage2_mom_chunk_fn);

    /* Implicit Rayleigh sponge on predicted velocity (before projection) */
    cfd_apply_sponge_velocity(sim, sim->u, sim->v, sim->w, dt);
    cfd_enforce_inflow_bcs(sim, sim->u, sim->v, sim->w);

    /* 6. Pressure projection: Dong/FDS OPEN BC + Vent BC -> FP32-MG / FP64-FCG -> u^{n+1} */
    cfd_compute_open_vent_phi_bc(sim, sim->u, sim->v, sim->w);

    int pcg_iters = 0;
    double pcg_rel_res = 0.0;
    cfd_poisson_solve(sim, dt, sim->u, sim->v, sim->w, sim->phi, &pcg_iters, &pcg_rel_res);

    double max_div = 0.0;
    cfd_project_velocity(sim, dt, sim->phi, sim->u, sim->v, sim->w,
                         do_diagnostics ? &max_div : NULL);
    cfd_enforce_inflow_bcs(sim, sim->u, sim->v, sim->w);

    /* 7. Scalar Stage 2: div-free SSP-RK2 using u^{n+1}, fused with sponge_theta */
    cfd_compute_scalar_rhs(sim, sim->u, sim->v, sim->w,
                           sim->theta, sim->smoke, sim->nu_sgs,
                           sim->rtheta, sim->rsmoke);
    Stage2ScalCtx s2s = { sim, dt };
    cfd_parallel_for_k(sim, sim->nz, 8, &s2s, rk2_stage2_scal_chunk_fn);

    /* 8. Advance clocks and compute diagnostics (only when requested) */
    sim->dt_prev = dt;
    sim->sim_time += dt;
    sim->step_count++;

    if (do_diagnostics) {
        double elapsed_ms = now_ms() - t_start;
        compute_diagnostics(sim, dt, max_div, pcg_iters, pcg_rel_res, elapsed_ms);
    }
    if (out_metrics) *out_metrics = sim->metrics;
    return 0;
}

int cfd_step(CFD3D *sim, CFDMetrics *out_metrics) {
    return cfd_step_internal(sim, 1, out_metrics);
}

int cfd_step_n(CFD3D *sim, int n_steps, CFDMetrics *out_metrics) {
    if (!sim || n_steps < 0) return -1;
    for (int s = 0; s < n_steps; ++s) {
        int do_diag = (s == n_steps - 1);
        if (cfd_step_internal(sim, do_diag, NULL) != 0) return -1;
    }
    if (out_metrics) *out_metrics = sim->metrics;
    return 0;
}

void cfd_get_metrics(const CFD3D *sim, CFDMetrics *out_metrics) {
    if (!sim || !out_metrics) return;
    *out_metrics = sim->metrics;
}

void cfd_get_geom_info(const CFD3D *sim, CFDGeomInfo *out_info) {
    if (!sim || !out_info) return;
    *out_info = sim->geom;
}

double *cfd_get_field_ptr(CFD3D *sim, int ptr_id, int *out_nz, int *out_ny, int *out_nx) {
    if (!sim) return NULL;
    int nz = sim->nz, ny = sim->ny, nx = sim->nx;
    double *ptr = NULL;
    switch (ptr_id) {
        case CFD_PTR_U:         ptr = sim->u;         nx = sim->nx + 1; break;
        case CFD_PTR_V:         ptr = sim->v;         ny = sim->ny + 1; break;
        case CFD_PTR_W:         ptr = sim->w;         nz = sim->nz + 1; break;
        case CFD_PTR_PHI:       ptr = sim->phi;       break;
        case CFD_PTR_THETA:     ptr = sim->theta;     break;
        case CFD_PTR_SMOKE:     ptr = sim->smoke;     break;
        case CFD_PTR_NU_SGS:    ptr = sim->nu_sgs;    break;
        case CFD_PTR_WALL_DIST: ptr = sim->wall_dist; break;
        default: return NULL;
    }
    if (out_nz) *out_nz = nz;
    if (out_ny) *out_ny = ny;
    if (out_nx) *out_nx = nx;
    return ptr;
}

uint8_t *cfd_get_mask_ptr(CFD3D *sim, int mask_id, int *out_nz, int *out_ny, int *out_nx) {
    if (!sim) return NULL;
    int nz = sim->nz, ny = sim->ny, nx = sim->nx;
    uint8_t *ptr = NULL;
    switch (mask_id) {
        case CFD_MASK_CELL_TYPE: ptr = sim->cell_type; break;
        case CFD_MASK_CELL_ZONE: ptr = sim->cell_zone; break;
        case CFD_MASK_KX:        ptr = sim->kx;        nx = sim->nx + 1; break;
        case CFD_MASK_KY:        ptr = sim->ky;        ny = sim->ny + 1; break;
        case CFD_MASK_KZ:        ptr = sim->kz;        nz = sim->nz + 1; break;
        default: return NULL;
    }
    if (out_nz) *out_nz = nz;
    if (out_ny) *out_ny = ny;
    if (out_nx) *out_nx = nx;
    return ptr;
}
