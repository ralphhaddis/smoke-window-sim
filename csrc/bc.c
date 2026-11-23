#include "cfd3d_internal.h"

static double dist_point_to_sash(const CFDGeomInfo *g, double px, double py, double pz) {
    /* Distance to 3D rectangle defined by corners[0], e1 = corners[1]-corners[0], e2 = corners[3]-corners[0] */
    double p0x = g->sash_corners[0][0];
    double p0y = g->sash_corners[0][1];
    double p0z = g->sash_corners[0][2];

    double e1x = g->sash_corners[1][0] - p0x;
    double e1y = g->sash_corners[1][1] - p0y;
    double e1z = g->sash_corners[1][2] - p0z;

    double e2x = g->sash_corners[3][0] - p0x;
    double e2y = g->sash_corners[3][1] - p0y;
    double e2z = g->sash_corners[3][2] - p0z;

    double e1_sq = e1x*e1x + e1y*e1y + e1z*e1z;
    double e2_sq = e2x*e2x + e2y*e2y + e2z*e2z;

    double rx = px - p0x, ry = py - p0y, rz = pz - p0z;
    double s = cfd_clamp((rx*e1x + ry*e1y + rz*e1z) / fmax(e1_sq, 1e-12), 0.0, 1.0);
    double t = cfd_clamp((rx*e2x + ry*e2y + rz*e2z) / fmax(e2_sq, 1e-12), 0.0, 1.0);

    double cx = p0x + s*e1x + t*e2x;
    double cy = p0y + s*e1y + t*e2y;
    double cz = p0z + s*e1z + t*e2z;

    double dx = px - cx, dy = py - cy, dz = pz - cz;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

void cfd_build_sponge(CFD3D *sim) {
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double h = sim->h;
    const int ix_ext = sim->geom.ix_ext;
    const int n_sponge = 5;
    const double smax = fmax(0.0, sim->params.sponge_sigma);

    memset(sim->sponge_sigma, 0, sim->ncells * sizeof(double));
    memset(sim->sponge_mask,  0, sim->ncells * sizeof(uint8_t));

    if (smax <= 0.0) return;

    for (int k = 0; k < nz; ++k) {
        double zc = (k + 0.5) * h;
        for (int j = 0; j < ny; ++j) {
            double yc = (j + 0.5) * h;
            for (int i = ix_ext; i < nx; ++i) {
                size_t c = idx_c(i, j, k, nx, ny);
                if (sim->cell_type[c] != CFD_CELL_FLUID) continue;

                double xc = (i + 0.5) * h;
                /* Strictly > 1.0 m away from the window sash */
                if (dist_point_to_sash(&sim->geom, xc, yc, zc) <= 1.0) continue;

                int dmin = n_sponge;
                uint8_t mask = 0;

                if (sim->bc_px == BC_OPEN) {
                    int d = (nx - 1) - i;
                    if (d < n_sponge) {
                        mask |= SPONGE_PX;
                        if (d < dmin) dmin = d;
                    }
                }
                if (sim->bc_my == BC_OPEN) {
                    int d = j;
                    if (d < n_sponge) {
                        mask |= SPONGE_MY;
                        if (d < dmin) dmin = d;
                    }
                }
                if (sim->bc_py == BC_OPEN) {
                    int d = (ny - 1) - j;
                    if (d < n_sponge) {
                        mask |= SPONGE_PY;
                        if (d < dmin) dmin = d;
                    }
                }
                if (sim->bc_pz == BC_OPEN) {
                    int d = (nz - 1) - k;
                    if (d < n_sponge) {
                        mask |= SPONGE_PZ;
                        if (d < dmin) dmin = d;
                    }
                }

                if (mask != 0 && dmin < n_sponge) {
                    double frac = (double)(n_sponge - dmin) / (double)n_sponge;
                    sim->sponge_sigma[c] = smax * frac * frac;
                    sim->sponge_mask[c]  = mask;
                }
            }
        }
    }
}

void cfd_enforce_inflow_bcs(CFD3D *sim, double *u, double *v, double *w) {
    (void)w;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double h = sim->h;

    /* Upwind Dirichlet wind inflow faces */
    if (sim->bc_px == BC_INFLOW) {
        for (int k = 0; k < nz; ++k) {
            double ux, uy;
            cfd_wind_at_z(sim, (k + 0.5) * h, &ux, &uy);
            (void)uy;
            for (int j = 0; j < ny; ++j) {
                size_t fu = idx_u(nx, j, k, nx, ny);
                if (sim->kx[fu]) u[fu] = ux;
            }
        }
    }
    if (sim->bc_my == BC_INFLOW) {
        for (int k = 0; k < nz; ++k) {
            double ux, uy;
            cfd_wind_at_z(sim, (k + 0.5) * h, &ux, &uy);
            (void)ux;
            for (int i = 0; i < nx; ++i) {
                size_t fv = idx_v(i, 0, k, nx, ny);
                if (sim->ky[fv]) v[fv] = uy;
            }
        }
    }
    if (sim->bc_py == BC_INFLOW) {
        for (int k = 0; k < nz; ++k) {
            double ux, uy;
            cfd_wind_at_z(sim, (k + 0.5) * h, &ux, &uy);
            (void)ux;
            for (int i = 0; i < nx; ++i) {
                size_t fv = idx_v(i, ny, k, nx, ny);
                if (sim->ky[fv]) v[fv] = uy;
            }
        }
    }
}

void cfd_enforce_velocity_bcs(CFD3D *sim, double *u, double *v, double *w) {
    /* Zero blocked faces */
    for (size_t idx = 0; idx < sim->nu; ++idx) {
        if (!sim->kx[idx]) u[idx] = 0.0;
    }
    for (size_t idx = 0; idx < sim->nv; ++idx) {
        if (!sim->ky[idx]) v[idx] = 0.0;
    }
    for (size_t idx = 0; idx < sim->nw; ++idx) {
        if (!sim->kz[idx]) w[idx] = 0.0;
    }

    cfd_enforce_inflow_bcs(sim, u, v, w);
}

static inline double dong_open_phi(double u_n_out, double u_face_sq, double u_inf_sq) {
    double u_inf = sqrt(u_inf_sq);
    double u0 = (u_inf > 0.2) ? u_inf : 0.2;
    double s0 = 0.5 * (1.0 - tanh(u_n_out / (0.05 * u0)));
    double bern = 0.5 * u_inf_sq - 0.5 * u_face_sq;
    return (bern < 0.0 ? bern : 0.0) * s0;
}

void cfd_compute_open_vent_phi_bc(CFD3D *sim, const double *u, const double *v, const double *w) {
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double h = sim->h;

    /* -X rear vent (BC_VENT) */
    memset(sim->phi_bc_mx, 0, (size_t)ny * (size_t)nz * sizeof(double));
    if (sim->params.vent_enabled) {
        double z_mid = 0.5 * (sim->geom.vent_z0 + sim->geom.vent_z1);
        double k_loss = fmax(0.0, sim->params.vent_k_loss);
        double dp_over_rho = sim->params.delta_p_pa / sim->rho0;
        double g_beta_dT = CFD_G * sim->beta * sim->theta_in;

        for (int k = 0; k < nz; ++k) {
            double zc = (k + 0.5) * h;
            double phi_hall = dp_over_rho + g_beta_dT * (zc - z_mid);
            for (int j = 0; j < ny; ++j) {
                size_t fu = idx_u(0, j, k, nx, ny);
                if (sim->kx[fu]) {
                    double ux_in = (u[fu] > 0.0) ? u[fu] : 0.0;
                    sim->phi_bc_mx[(size_t)k * (size_t)ny + (size_t)j] =
                        phi_hall - 0.5 * (1.0 + k_loss) * ux_in * ux_in;
                }
            }
        }
    }

    /* +X exterior atmospheric face */
    memset(sim->phi_bc_px, 0, (size_t)ny * (size_t)nz * sizeof(double));
    if (sim->bc_px == BC_OPEN) {
        for (int k = 0; k < nz; ++k) {
            double ux_inf, uy_inf;
            cfd_wind_at_z(sim, (k + 0.5) * h, &ux_inf, &uy_inf);
            double u_inf_sq = ux_inf * ux_inf + uy_inf * uy_inf;
            for (int j = 0; j < ny; ++j) {
                size_t fu = idx_u(nx, j, k, nx, ny);
                if (!sim->kx[fu]) continue;
                double un = u[fu];
                double vf = 0.5 * (v[idx_v(nx - 1, j, k, nx, ny)] + v[idx_v(nx - 1, j + 1, k, nx, ny)]);
                double wf = 0.5 * (w[idx_w(nx - 1, j, k, nx, ny)] + w[idx_w(nx - 1, j, k + 1, nx, ny)]);
                double usq = un * un + vf * vf + wf * wf;
                sim->phi_bc_px[(size_t)k * (size_t)ny + (size_t)j] = dong_open_phi(un, usq, u_inf_sq);
            }
        }
    }

    /* -Y exterior atmospheric face */
    memset(sim->phi_bc_my, 0, (size_t)nx * (size_t)nz * sizeof(double));
    if (sim->bc_my == BC_OPEN) {
        for (int k = 0; k < nz; ++k) {
            double ux_inf, uy_inf;
            cfd_wind_at_z(sim, (k + 0.5) * h, &ux_inf, &uy_inf);
            double u_inf_sq = ux_inf * ux_inf + uy_inf * uy_inf;
            for (int i = 0; i < nx; ++i) {
                size_t fv = idx_v(i, 0, k, nx, ny);
                if (!sim->ky[fv]) continue;
                double un = -v[fv]; /* outward normal is -y */
                double uf = 0.5 * (u[idx_u(i, 0, k, nx, ny)] + u[idx_u(i + 1, 0, k, nx, ny)]);
                double wf = 0.5 * (w[idx_w(i, 0, k, nx, ny)] + w[idx_w(i, 0, k + 1, nx, ny)]);
                double usq = un * un + uf * uf + wf * wf;
                sim->phi_bc_my[(size_t)k * (size_t)nx + (size_t)i] = dong_open_phi(un, usq, u_inf_sq);
            }
        }
    }

    /* +Y exterior atmospheric face */
    memset(sim->phi_bc_py, 0, (size_t)nx * (size_t)nz * sizeof(double));
    if (sim->bc_py == BC_OPEN) {
        for (int k = 0; k < nz; ++k) {
            double ux_inf, uy_inf;
            cfd_wind_at_z(sim, (k + 0.5) * h, &ux_inf, &uy_inf);
            double u_inf_sq = ux_inf * ux_inf + uy_inf * uy_inf;
            for (int i = 0; i < nx; ++i) {
                size_t fv = idx_v(i, ny, k, nx, ny);
                if (!sim->ky[fv]) continue;
                double un = v[fv]; /* outward normal is +y */
                double uf = 0.5 * (u[idx_u(i, ny - 1, k, nx, ny)] + u[idx_u(i + 1, ny - 1, k, nx, ny)]);
                double wf = 0.5 * (w[idx_w(i, ny - 1, k, nx, ny)] + w[idx_w(i, ny - 1, k + 1, nx, ny)]);
                double usq = un * un + uf * uf + wf * wf;
                sim->phi_bc_py[(size_t)k * (size_t)nx + (size_t)i] = dong_open_phi(un, usq, u_inf_sq);
            }
        }
    }

    /* +Z exterior sky face */
    memset(sim->phi_bc_pz, 0, (size_t)nx * (size_t)ny * sizeof(double));
    if (sim->bc_pz == BC_OPEN) {
        double ux_inf, uy_inf;
        cfd_wind_at_z(sim, nz * h, &ux_inf, &uy_inf);
        double u_inf_sq = ux_inf * ux_inf + uy_inf * uy_inf;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t fw = idx_w(i, j, nz, nx, ny);
                if (!sim->kz[fw]) continue;
                double un = w[fw]; /* outward normal is +z */
                double uf = 0.5 * (u[idx_u(i, j, nz - 1, nx, ny)] + u[idx_u(i + 1, j, nz - 1, nx, ny)]);
                double vf = 0.5 * (v[idx_v(i, j, nz - 1, nx, ny)] + v[idx_v(i, j + 1, nz - 1, nx, ny)]);
                double usq = un * un + uf * uf + vf * vf;
                sim->phi_bc_pz[(size_t)j * (size_t)nx + (size_t)i] = dong_open_phi(un, usq, u_inf_sq);
            }
        }
    }
}

typedef struct SpongeVelCtx {
    const CFD3D *sim;
    double *u, *v, *w;
    double dt;
} SpongeVelCtx;

static void sponge_vel_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    SpongeVelCtx *ctx = (SpongeVelCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const int ix_ext = sim->geom.ix_ext;
    const double h = sim->h;
    const double dt = ctx->dt;
    const double *restrict sig = sim->sponge_sigma;
    const uint8_t *restrict msk = sim->sponge_mask;
    const uint8_t *restrict kx = sim->kx;
    const uint8_t *restrict ky = sim->ky;
    const uint8_t *restrict kz = sim->kz;
    double *restrict u = ctx->u;
    double *restrict v = ctx->v;
    double *restrict w = ctx->w;

    for (int k = k0; k < k1; ++k) {
        double ux_inf, uy_inf;
        cfd_wind_at_z(sim, (k + 0.5) * h, &ux_inf, &uy_inf);

        /* u-faces in exterior domain */
        for (int j = 0; j < ny; ++j) {
            const size_t row_c = idx_c(0, j, k, nx, ny);
            const size_t row_u = idx_u(0, j, k, nx, ny);
            for (int i = ix_ext + 1; i < nx; ++i) {
                size_t fu = row_u + (size_t)i;
                if (!kx[fu]) continue;
                size_t cL = row_c + (size_t)(i - 1);
                size_t cR = row_c + (size_t)i;
                double s = 0.5 * (sig[cL] + sig[cR]);
                if (s > 0.0) {
                    uint8_t m = (uint8_t)(msk[cL] | msk[cR]);
                    if ((m & SPONGE_PX) && u[fu] > 0.0) continue;
                    u[fu] = (u[fu] + dt * s * ux_inf) / (1.0 + dt * s);
                }
            }
            if (sim->bc_px != BC_INFLOW) {
                size_t fu = row_u + (size_t)nx;
                if (kx[fu]) {
                    size_t c = row_c + (size_t)(nx - 1);
                    double s = sig[c];
                    if (s > 0.0) {
                        if (!((msk[c] & SPONGE_PX) && u[fu] > 0.0)) {
                            u[fu] = (u[fu] + dt * s * ux_inf) / (1.0 + dt * s);
                        }
                    }
                }
            }
        }

        /* v-faces in exterior domain */
        for (int j = 0; j <= ny; ++j) {
            if (j == 0  && sim->bc_my == BC_INFLOW) continue;
            if (j == ny && sim->bc_py == BC_INFLOW) continue;
            const size_t row_v = idx_v(0, j, k, nx, ny);
            if (j == 0) {
                const size_t row_c = idx_c(0, 0, k, nx, ny);
                for (int i = ix_ext; i < nx; ++i) {
                    size_t fv = row_v + (size_t)i;
                    if (!ky[fv]) continue;
                    double s = sig[row_c + (size_t)i];
                    if (s > 0.0) {
                        uint8_t m = msk[row_c + (size_t)i];
                        if ((m & SPONGE_MY) && v[fv] < 0.0) continue;
                        if ((m & SPONGE_PY) && v[fv] > 0.0) continue;
                        v[fv] = (v[fv] + dt * s * uy_inf) / (1.0 + dt * s);
                    }
                }
            } else if (j == ny) {
                const size_t row_c = idx_c(0, ny - 1, k, nx, ny);
                for (int i = ix_ext; i < nx; ++i) {
                    size_t fv = row_v + (size_t)i;
                    if (!ky[fv]) continue;
                    double s = sig[row_c + (size_t)i];
                    if (s > 0.0) {
                        uint8_t m = msk[row_c + (size_t)i];
                        if ((m & SPONGE_MY) && v[fv] < 0.0) continue;
                        if ((m & SPONGE_PY) && v[fv] > 0.0) continue;
                        v[fv] = (v[fv] + dt * s * uy_inf) / (1.0 + dt * s);
                    }
                }
            } else {
                const size_t row_cL = idx_c(0, j - 1, k, nx, ny);
                const size_t row_cR = idx_c(0, j,     k, nx, ny);
                for (int i = ix_ext; i < nx; ++i) {
                    size_t fv = row_v + (size_t)i;
                    if (!ky[fv]) continue;
                    double s = 0.5 * (sig[row_cL + (size_t)i] + sig[row_cR + (size_t)i]);
                    if (s > 0.0) {
                        uint8_t m = (uint8_t)(msk[row_cL + (size_t)i] | msk[row_cR + (size_t)i]);
                        if ((m & SPONGE_MY) && v[fv] < 0.0) continue;
                        if ((m & SPONGE_PY) && v[fv] > 0.0) continue;
                        v[fv] = (v[fv] + dt * s * uy_inf) / (1.0 + dt * s);
                    }
                }
            }
        }

        /* w-faces for interior k >= 1 */
        if (k >= 1) {
            for (int j = 0; j < ny; ++j) {
                const size_t row_w  = idx_w(0, j, k,     nx, ny);
                const size_t row_cB = idx_c(0, j, k - 1, nx, ny);
                const size_t row_cT = idx_c(0, j, k,     nx, ny);
                for (int i = ix_ext; i < nx; ++i) {
                    size_t fw = row_w + (size_t)i;
                    if (!kz[fw]) continue;
                    double s = 0.5 * (sig[row_cB + (size_t)i] + sig[row_cT + (size_t)i]);
                    if (s > 0.0) {
                        uint8_t m = (uint8_t)(msk[row_cB + (size_t)i] | msk[row_cT + (size_t)i]);
                        if ((m & SPONGE_PZ) && w[fw] > 0.0) continue;
                        w[fw] = w[fw] / (1.0 + dt * s);
                    }
                }
            }
        }
    }

    /* Top boundary w-face k = nz handled by last chunk */
    if (k1 == nz) {
        for (int j = 0; j < ny; ++j) {
            const size_t row_w = idx_w(0, j, nz,     nx, ny);
            const size_t row_c = idx_c(0, j, nz - 1, nx, ny);
            for (int i = ix_ext; i < nx; ++i) {
                size_t fw = row_w + (size_t)i;
                if (!kz[fw]) continue;
                double s = sig[row_c + (size_t)i];
                if (s > 0.0) {
                    if ((msk[row_c + (size_t)i] & SPONGE_PZ) && w[fw] > 0.0) continue;
                    w[fw] = w[fw] / (1.0 + dt * s);
                }
            }
        }
    }
}

void cfd_apply_sponge_velocity(CFD3D *sim, double *u, double *v, double *w, double dt) {
    if (sim->params.sponge_sigma <= 0.0) return;
    SpongeVelCtx ctx = { sim, u, v, w, dt };
    cfd_parallel_for_k(sim, sim->nz, 8, &ctx, sponge_vel_chunk_fn);
}

typedef struct SpongeThetaCtx {
    const CFD3D *sim;
    double *theta;
    double dt;
} SpongeThetaCtx;

static void sponge_theta_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    SpongeThetaCtx *ctx = (SpongeThetaCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny;
    const int ix_ext = sim->geom.ix_ext;
    const double dt = ctx->dt;
    const double *restrict sig = sim->sponge_sigma;
    double *restrict theta = ctx->theta;

    for (int k = k0; k < k1; ++k) {
        for (int j = 0; j < ny; ++j) {
            const size_t row = idx_c(0, j, k, nx, ny);
            const double *restrict sr = sig + row;
            double *restrict tr = theta + row;
            for (int i = ix_ext; i < nx; ++i) {
                double s = sr[i];
                if (s > 0.0) {
                    tr[i] /= (1.0 + dt * s);
                }
            }
        }
    }
}

void cfd_apply_sponge_theta(CFD3D *sim, double *theta, double dt) {
    if (sim->params.sponge_sigma <= 0.0) return;
    SpongeThetaCtx ctx = { sim, theta, dt };
    cfd_parallel_for_k(sim, sim->nz, 8, &ctx, sponge_theta_chunk_fn);
}
