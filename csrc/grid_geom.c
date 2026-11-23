#include "cfd3d_internal.h"

void cfd_update_thermo_and_wind(CFD3D *sim) {
    const CFDParams *p = &sim->params;
    sim->t_out_k  = p->temp_out_c + 273.15;
    sim->t_in_k   = p->temp_in_c  + 273.15;
    if (sim->t_out_k < 150.0) sim->t_out_k = 150.0;
    if (sim->t_in_k  < 150.0) sim->t_in_k  = 150.0;
    sim->theta_in = sim->t_in_k - sim->t_out_k;
    sim->beta     = 1.0 / sim->t_out_k;
    sim->rho0     = CFD_P_ATM / (CFD_R_AIR * sim->t_out_k);

    double rad = p->wind_dir_deg * (CFD_PI / 180.0);
    double spd = (p->wind_speed > 0.0) ? p->wind_speed : 0.0;
    /* dir=0 -> blowing (-1, 0, 0) onto window; dir=90 -> (0, +1, 0) */
    sim->wind_ux_ref = -spd * cos(rad);
    sim->wind_uy_ref =  spd * sin(rad);

    /* Upwind wind-inflow faces where U_inf . n_out < -1e-3 */
    sim->bc_px = (sim->wind_ux_ref < -1e-3) ? BC_INFLOW : BC_OPEN;
    sim->bc_my = (-sim->wind_uy_ref < -1e-3) ? BC_INFLOW : BC_OPEN;
    sim->bc_py = (sim->wind_uy_ref < -1e-3) ? BC_INFLOW : BC_OPEN;
    sim->bc_pz = BC_OPEN;
}

typedef struct SashRect {
    double p0[3];
    double e1[3];
    double e2[3];
    double inv_e1_sq;
    double inv_e2_sq;
    double pad1;
    double pad2;
    double n[3];
} SashRect;

static int dual_segment_hits_sash(const SashRect *s,
                                  double ax, double ay, double az,
                                  double bx, double by, double bz) {
    double da = (ax - s->p0[0]) * s->n[0]
              + (ay - s->p0[1]) * s->n[1]
              + (az - s->p0[2]) * s->n[2];
    double db = (bx - s->p0[0]) * s->n[0]
              + (by - s->p0[1]) * s->n[1]
              + (bz - s->p0[2]) * s->n[2];

    /* Half-open sign test guarantees watertight separation with no isolated cells */
    if ((da < 0.0) == (db < 0.0)) {
        return 0;
    }

    double denom = da - db;
    if (fabs(denom) < 1e-15) {
        return 0;
    }
    double tau = da / denom;
    double ix = ax + tau * (bx - ax) - s->p0[0];
    double iy = ay + tau * (by - ay) - s->p0[1];
    double iz = az + tau * (bz - az) - s->p0[2];

    double u1 = (ix * s->e1[0] + iy * s->e1[1] + iz * s->e1[2]) * s->inv_e1_sq;
    if (u1 < -s->pad1 || u1 > 1.0 + s->pad1) {
        return 0;
    }
    double u2 = (ix * s->e2[0] + iy * s->e2[1] + iz * s->e2[2]) * s->inv_e2_sq;
    if (u2 < -s->pad2 || u2 > 1.0 + s->pad2) {
        return 0;
    }
    return 1;
}

static void compute_wall_distance(CFD3D *sim) {
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double h = sim->h;
    const double d1 = h;
    const double d2 = h * 1.4142135623730951;
    const double d3 = h * 1.7320508075688772;
    const double big = 1e6;
    double *wd = sim->wall_dist;

    /* Seed solid cells with 0 and fluid cells touching any blocked face with 0.5*h */
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t c = idx_c(i, j, k, nx, ny);
                if (sim->cell_type[c] == CFD_CELL_SOLID) {
                    wd[c] = 0.0;
                    continue;
                }
                int blocked =
                    (sim->kx[idx_u(i,     j, k, nx, ny)] == 0) ||
                    (sim->kx[idx_u(i + 1, j, k, nx, ny)] == 0) ||
                    (sim->ky[idx_v(i, j,     k, nx, ny)] == 0) ||
                    (sim->ky[idx_v(i, j + 1, k, nx, ny)] == 0) ||
                    (sim->kz[idx_w(i, j, k,     nx, ny)] == 0) ||
                    (sim->kz[idx_w(i, j, k + 1, nx, ny)] == 0);
                wd[c] = blocked ? (0.5 * h) : big;
            }
        }
    }

    /* Two forward-backward Chamfer 3D passes yield near-exact Euclidean distance */
    for (int pass = 0; pass < 2; ++pass) {
        /* Forward pass */
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t c = idx_c(i, j, k, nx, ny);
                    if (sim->cell_type[c] == CFD_CELL_SOLID) continue;
                    double best = wd[c];
                    for (int dk = -1; dk <= 0; ++dk) {
                        int kk = k + dk;
                        if (kk < 0) continue;
                        for (int dj = -1; dj <= 1; ++dj) {
                            if (dk == 0 && dj > 0) continue;
                            int jj = j + dj;
                            if (jj < 0 || jj >= ny) continue;
                            for (int di = -1; di <= 1; ++di) {
                                if (dk == 0 && dj == 0 && di >= 0) continue;
                                int ii = i + di;
                                if (ii < 0 || ii >= nx) continue;
                                int manh = abs(di) + abs(dj) + abs(dk);
                                double step = (manh == 1) ? d1 : ((manh == 2) ? d2 : d3);
                                double cand = wd[idx_c(ii, jj, kk, nx, ny)] + step;
                                if (cand < best) best = cand;
                            }
                        }
                    }
                    wd[c] = best;
                }
            }
        }
        /* Backward pass */
        for (int k = nz - 1; k >= 0; --k) {
            for (int j = ny - 1; j >= 0; --j) {
                for (int i = nx - 1; i >= 0; --i) {
                    size_t c = idx_c(i, j, k, nx, ny);
                    if (sim->cell_type[c] == CFD_CELL_SOLID) continue;
                    double best = wd[c];
                    for (int dk = 0; dk <= 1; ++dk) {
                        int kk = k + dk;
                        if (kk >= nz) continue;
                        for (int dj = -1; dj <= 1; ++dj) {
                            if (dk == 0 && dj < 0) continue;
                            int jj = j + dj;
                            if (jj < 0 || jj >= ny) continue;
                            for (int di = -1; di <= 1; ++di) {
                                if (dk == 0 && dj == 0 && di <= 0) continue;
                                int ii = i + di;
                                if (ii < 0 || ii >= nx) continue;
                                int manh = abs(di) + abs(dj) + abs(dk);
                                double step = (manh == 1) ? d1 : ((manh == 2) ? d2 : d3);
                                double cand = wd[idx_c(ii, jj, kk, nx, ny)] + step;
                                if (cand < best) best = cand;
                            }
                        }
                    }
                    wd[c] = best;
                }
            }
        }
    }
}

void cfd_build_geometry(CFD3D *sim) {
    const CFDParams *p = &sim->params;
    CFDGeomInfo *g = &sim->geom;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double h = sim->h;

    g->nx = nx; g->ny = ny; g->nz = nz;
    g->h  = h;
    g->lx = nx * h;
    g->ly = ny * h;
    g->lz = nz * h;

    /* Snap room & masonry facade to cell faces */
    int ix_wall = (int)llround(p->room_lx / h);
    int wall_cells = (int)llround(p->wall_thick / h);
    if (wall_cells < 1) wall_cells = 1;
    ix_wall = cfd_iclamp(ix_wall, 4, nx - wall_cells - 6);
    int ix_ext = ix_wall + wall_cells;

    int n_room_y = (int)llround(p->room_ly / h);
    n_room_y = cfd_iclamp(n_room_y, 6, ny - 2);
    int iy_min = (ny - n_room_y) / 2;
    int iy_max = iy_min + n_room_y;

    int iz_ceil = (int)llround(p->room_lz / h);
    iz_ceil = cfd_iclamp(iz_ceil, 6, nz - 4);

    g->ix_wall = ix_wall;
    g->ix_ext  = ix_ext;
    g->iy_min  = iy_min;
    g->iy_max  = iy_max;
    g->iz_ceil = iz_ceil;

    g->room_x0 = 0.0;
    g->room_x1 = ix_wall * h;
    g->wall_x1 = ix_ext  * h;
    g->room_y0 = iy_min  * h;
    g->room_y1 = iy_max  * h;
    g->room_z0 = 0.0;
    g->room_z1 = iz_ceil * h;

    /* Snap UK casement window opening */
    double room_yc = 0.5 * (g->room_y0 + g->room_y1);
    double wy0_req = room_yc + p->win_y_offset - 0.5 * p->win_width;
    double wy1_req = wy0_req + p->win_width;
    double wz0_req = p->win_sill;
    double wz1_req = p->win_sill + p->win_height;

    int j_w0 = (int)llround(wy0_req / h);
    int j_w1 = (int)llround(wy1_req / h);
    j_w0 = cfd_iclamp(j_w0, iy_min + 1, iy_max - 2);
    j_w1 = cfd_iclamp(j_w1, j_w0 + 1,   iy_max - 1);

    int k_w0 = (int)llround(wz0_req / h);
    int k_w1 = (int)llround(wz1_req / h);
    k_w0 = cfd_iclamp(k_w0, 1,        iz_ceil - 2);
    k_w1 = cfd_iclamp(k_w1, k_w0 + 1, iz_ceil - 1);

    g->j_w0 = j_w0; g->j_w1 = j_w1;
    g->k_w0 = k_w0; g->k_w1 = k_w1;
    g->win_y0 = j_w0 * h; g->win_y1 = j_w1 * h;
    g->win_z0 = k_w0 * h; g->win_z1 = k_w1 * h;

    /* Snap optional rear door / vent on x = 0 indoor wall */
    int j_v0 = (int)llround((g->room_y0 + p->vent_y0) / h);
    int j_v1 = (int)llround((g->room_y0 + p->vent_y1) / h);
    j_v0 = cfd_iclamp(j_v0, iy_min, iy_max - 1);
    j_v1 = cfd_iclamp(j_v1, j_v0 + 1, iy_max);

    int k_v0 = (int)llround(p->vent_z0 / h);
    int k_v1 = (int)llround(p->vent_z1 / h);
    k_v0 = cfd_iclamp(k_v0, 0, iz_ceil - 1);
    k_v1 = cfd_iclamp(k_v1, k_v0 + 1, iz_ceil);

    g->j_v0 = j_v0; g->j_v1 = j_v1;
    g->k_v0 = k_v0; g->k_v1 = k_v1;
    g->vent_y0 = j_v0 * h; g->vent_y1 = j_v1 * h;
    g->vent_z0 = k_v0 * h; g->vent_z1 = k_v1 * h;

    /* Classify cells */
    int n_fluid = 0;
    int n_indoor = 0;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t c = idx_c(i, j, k, nx, ny);
                if (i >= ix_ext) {
                    sim->cell_type[c] = CFD_CELL_FLUID;
                    sim->cell_zone[c] = CFD_ZONE_OUTDOOR;
                    n_fluid++;
                } else if (j < iy_min || j >= iy_max || k >= iz_ceil) {
                    sim->cell_type[c] = CFD_CELL_SOLID;
                    sim->cell_zone[c] = CFD_ZONE_SOLID;
                } else if (i < ix_wall) {
                    sim->cell_type[c] = CFD_CELL_FLUID;
                    sim->cell_zone[c] = CFD_ZONE_INDOOR;
                    n_fluid++;
                    n_indoor++;
                } else {
                    /* Brick wall reveal [ix_wall, ix_ext) */
                    if (j >= j_w0 && j < j_w1 && k >= k_w0 && k < k_w1) {
                        sim->cell_type[c] = CFD_CELL_FLUID;
                        sim->cell_zone[c] = CFD_ZONE_REVEAL;
                        n_fluid++;
                    } else {
                        sim->cell_type[c] = CFD_CELL_SOLID;
                        sim->cell_zone[c] = CFD_ZONE_SOLID;
                    }
                }
            }
        }
    }
    g->n_fluid_cells  = n_fluid;
    g->n_indoor_cells = n_indoor;
    g->room_volume    = (double)n_indoor * h * h * h;

    /* Construct exact 3D angled casement sash rectangle */
    double alpha_deg = cfd_clamp(p->win_angle_deg, 0.0, 90.0);
    double alpha = alpha_deg * (CFD_PI / 180.0);
    double sa = sin(alpha);
    double ca = cos(alpha);
    double W = g->win_y1 - g->win_y0;
    double H = g->win_z1 - g->win_z0;
    double xh = g->wall_x1;

    SashRect sash;
    if (p->win_hinge_mode == CFD_HINGE_RIGHT) {
        sash.p0[0] = xh;         sash.p0[1] = g->win_y1;  sash.p0[2] = g->win_z0;
        sash.e1[0] = W * sa;     sash.e1[1] = -W * ca;    sash.e1[2] = 0.0;
        sash.e2[0] = 0.0;        sash.e2[1] = 0.0;        sash.e2[2] = H;
    } else if (p->win_hinge_mode == CFD_HINGE_TOP) {
        sash.p0[0] = xh;         sash.p0[1] = g->win_y0;  sash.p0[2] = g->win_z1;
        sash.e1[0] = 0.0;        sash.e1[1] = W;          sash.e1[2] = 0.0;
        sash.e2[0] = H * sa;     sash.e2[1] = 0.0;        sash.e2[2] = -H * ca;
    } else {
        /* CFD_HINGE_LEFT (default) */
        sash.p0[0] = xh;         sash.p0[1] = g->win_y0;  sash.p0[2] = g->win_z0;
        sash.e1[0] = W * sa;     sash.e1[1] = W * ca;     sash.e1[2] = 0.0;
        sash.e2[0] = 0.0;        sash.e2[1] = 0.0;        sash.e2[2] = H;
    }

    double e1_sq = sash.e1[0]*sash.e1[0] + sash.e1[1]*sash.e1[1] + sash.e1[2]*sash.e1[2];
    double e2_sq = sash.e2[0]*sash.e2[0] + sash.e2[1]*sash.e2[1] + sash.e2[2]*sash.e2[2];
    sash.inv_e1_sq = 1.0 / fmax(e1_sq, 1e-12);
    sash.inv_e2_sq = 1.0 / fmax(e2_sq, 1e-12);
    sash.pad1 = 0.01 * h / fmax(sqrt(e1_sq), 1e-6);
    sash.pad2 = 0.01 * h / fmax(sqrt(e2_sq), 1e-6);

    /* Unit normal = normalize(e1 x e2) */
    double nx_v = sash.e1[1]*sash.e2[2] - sash.e1[2]*sash.e2[1];
    double ny_v = sash.e1[2]*sash.e2[0] - sash.e1[0]*sash.e2[2];
    double nz_v = sash.e1[0]*sash.e2[1] - sash.e1[1]*sash.e2[0];
    double nlen = sqrt(nx_v*nx_v + ny_v*ny_v + nz_v*nz_v);
    sash.n[0] = nx_v / nlen;
    sash.n[1] = ny_v / nlen;
    sash.n[2] = nz_v / nlen;

    /* Store 4 sash corners and normal in geom info for visualization */
    for (int d = 0; d < 3; ++d) {
        g->sash_corners[0][d] = sash.p0[d];
        g->sash_corners[1][d] = sash.p0[d] + sash.e1[d];
        g->sash_corners[2][d] = sash.p0[d] + sash.e1[d] + sash.e2[d];
        g->sash_corners[3][d] = sash.p0[d] + sash.e2[d];
        g->sash_normal[d]     = sash.n[d];
    }

    /* Build face permeability masks kx, ky, kz */
    int n_sash_blocked = 0;

    /* kx faces: (nx+1) * ny * nz */
    for (int k = 0; k < nz; ++k) {
        double zc = (k + 0.5) * h;
        for (int j = 0; j < ny; ++j) {
            double yc = (j + 0.5) * h;
            for (int i = 0; i <= nx; ++i) {
                size_t fu = idx_u(i, j, k, nx, ny);
                if (i == 0) {
                    int is_vent = p->vent_enabled &&
                                  (sim->cell_type[idx_c(0, j, k, nx, ny)] == CFD_CELL_FLUID) &&
                                  (j >= j_v0 && j < j_v1 && k >= k_v0 && k < k_v1);
                    sim->kx[fu] = is_vent ? 1 : 0;
                } else if (i == nx) {
                    sim->kx[fu] = (sim->cell_type[idx_c(nx - 1, j, k, nx, ny)] == CFD_CELL_FLUID) ? 1 : 0;
                } else {
                    int fl = (sim->cell_type[idx_c(i - 1, j, k, nx, ny)] == CFD_CELL_FLUID) &&
                             (sim->cell_type[idx_c(i,     j, k, nx, ny)] == CFD_CELL_FLUID);
                    if (!fl) {
                        sim->kx[fu] = 0;
                    } else if (dual_segment_hits_sash(&sash,
                                                      (i - 0.5) * h, yc, zc,
                                                      (i + 0.5) * h, yc, zc)) {
                        sim->kx[fu] = 0;
                        n_sash_blocked++;
                    } else {
                        sim->kx[fu] = 1;
                    }
                }
            }
        }
    }

    /* ky faces: nx * (ny+1) * nz */
    for (int k = 0; k < nz; ++k) {
        double zc = (k + 0.5) * h;
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double xc = (i + 0.5) * h;
                size_t fv = idx_v(i, j, k, nx, ny);
                if (j == 0) {
                    sim->ky[fv] = (sim->cell_type[idx_c(i, 0, k, nx, ny)] == CFD_CELL_FLUID) ? 1 : 0;
                } else if (j == ny) {
                    sim->ky[fv] = (sim->cell_type[idx_c(i, ny - 1, k, nx, ny)] == CFD_CELL_FLUID) ? 1 : 0;
                } else {
                    int fl = (sim->cell_type[idx_c(i, j - 1, k, nx, ny)] == CFD_CELL_FLUID) &&
                             (sim->cell_type[idx_c(i, j,     k, nx, ny)] == CFD_CELL_FLUID);
                    if (!fl) {
                        sim->ky[fv] = 0;
                    } else if (dual_segment_hits_sash(&sash,
                                                      xc, (j - 0.5) * h, zc,
                                                      xc, (j + 0.5) * h, zc)) {
                        sim->ky[fv] = 0;
                        n_sash_blocked++;
                    } else {
                        sim->ky[fv] = 1;
                    }
                }
            }
        }
    }

    /* kz faces: nx * ny * (nz+1) */
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            double yc = (j + 0.5) * h;
            for (int i = 0; i < nx; ++i) {
                double xc = (i + 0.5) * h;
                size_t fw = idx_w(i, j, k, nx, ny);
                if (k == 0) {
                    /* Solid floor & exterior ground */
                    sim->kz[fw] = 0;
                } else if (k == nz) {
                    sim->kz[fw] = (sim->cell_type[idx_c(i, j, nz - 1, nx, ny)] == CFD_CELL_FLUID) ? 1 : 0;
                } else {
                    int fl = (sim->cell_type[idx_c(i, j, k - 1, nx, ny)] == CFD_CELL_FLUID) &&
                             (sim->cell_type[idx_c(i, j, k,     nx, ny)] == CFD_CELL_FLUID);
                    if (!fl) {
                        sim->kz[fw] = 0;
                    } else if (dual_segment_hits_sash(&sash,
                                                      xc, yc, (k - 0.5) * h,
                                                      xc, yc, (k + 0.5) * h)) {
                        sim->kz[fw] = 0;
                        n_sash_blocked++;
                    } else {
                        sim->kz[fw] = 1;
                    }
                }
            }
        }
    }

    g->n_sash_blocked_faces = n_sash_blocked;

    /* Precompute 3D wall distance for Mason-Thomson LES damping */
    compute_wall_distance(sim);
}

void cfd_build_emitter(CFD3D *sim) {
    const CFDParams *p = &sim->params;
    const CFDGeomInfo *g = &sim->geom;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double h = sim->h;

    memset(sim->emit_kernel, 0, sim->ncells * sizeof(double));

    double wx = cfd_clamp(p->smoke_x, 0.5 * h, g->lx - 0.5 * h);
    double wy = cfd_clamp(g->room_y0 + p->smoke_y, 0.5 * h, g->ly - 0.5 * h);
    double wz = cfd_clamp(p->smoke_z, 0.5 * h, g->lz - 0.5 * h);
    double sigma = fmax(p->smoke_radius, 1.2 * h);

    sim->emit_wx = wx;
    sim->emit_wy = wy;
    sim->emit_wz = wz;
    sim->emit_sigma = sigma;

    double az = p->smoke_jet_azim_deg * (CFD_PI / 180.0);
    double el = p->smoke_jet_elev_deg * (CFD_PI / 180.0);
    sim->emit_jx = cos(el) * cos(az);
    sim->emit_jy = cos(el) * sin(az);
    sim->emit_jz = sin(el);

    double rcut = 3.5 * sigma;
    int i0 = cfd_iclamp((int)floor((wx - rcut) / h), 0, nx - 1);
    int i1 = cfd_iclamp((int)ceil ((wx + rcut) / h), 0, nx - 1);
    int j0 = cfd_iclamp((int)floor((wy - rcut) / h), 0, ny - 1);
    int j1 = cfd_iclamp((int)ceil ((wy + rcut) / h), 0, ny - 1);
    int k0 = cfd_iclamp((int)floor((wz - rcut) / h), 0, nz - 1);
    int k1 = cfd_iclamp((int)ceil ((wz + rcut) / h), 0, nz - 1);

    sim->emit_i0 = i0; sim->emit_i1 = i1;
    sim->emit_j0 = j0; sim->emit_j1 = j1;
    sim->emit_k0 = k0; sim->emit_k1 = k1;

    int emit_in_room = (wx < g->room_x1);
    int emit_outside = (wx >= g->wall_x1);
    int win_open = (p->win_angle_deg > 1e-3);

    double inv_2sig2 = 0.5 / (sigma * sigma);
    double sum_w = 0.0;
    for (int k = k0; k <= k1; ++k) {
        double dz = (k + 0.5) * h - wz;
        for (int j = j0; j <= j1; ++j) {
            double dy = (j + 0.5) * h - wy;
            for (int i = i0; i <= i1; ++i) {
                size_t c = idx_c(i, j, k, nx, ny);
                if (sim->cell_type[c] != CFD_CELL_FLUID) continue;
                uint8_t zc = sim->cell_zone[c];
                if (emit_in_room && zc == CFD_ZONE_OUTDOOR) continue;
                if (emit_outside && zc == CFD_ZONE_INDOOR) continue;
                if (!win_open && !emit_outside && zc == CFD_ZONE_OUTDOOR) continue;
                if (!win_open && emit_outside && zc != CFD_ZONE_OUTDOOR) continue;
                double dx = (i + 0.5) * h - wx;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 > rcut * rcut) continue;
                double w = exp(-r2 * inv_2sig2);
                sim->emit_kernel[c] = w;
                sum_w += w;
            }
        }
    }

    double cell_vol = h * h * h;
    if (sum_w > 1e-15) {
        double inv_norm = 1.0 / (sum_w * cell_vol);
        for (int k = k0; k <= k1; ++k) {
            for (int j = j0; j <= j1; ++j) {
                for (int i = i0; i <= i1; ++i) {
                    size_t c = idx_c(i, j, k, nx, ny);
                    sim->emit_kernel[c] *= inv_norm;
                }
            }
        }
    }
}
