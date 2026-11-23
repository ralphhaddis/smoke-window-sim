#include "cfd3d_internal.h"

static inline double van_leer_face(double q_uu, double q_u, double q_d, int has_uu) {
    if (!has_uu) return q_u;
    double dm = q_u - q_uu;
    double dp = q_d - q_u;
    double prod = dm * dp;
    if (prod <= 0.0) return q_u;
    return q_u + (prod / (dm + dp));
}

static inline double mc_face(double q_uu, double q_u, double q_d, int has_uu) {
    if (!has_uu) return q_u;
    double dm = q_u - q_uu;
    double dp = q_d - q_u;
    if (dm * dp <= 0.0) return q_u;
    double adm = fabs(dm);
    double adp = fabs(dp);
    double m = fmin(2.0 * fmin(adm, adp), 0.5 * (adm + adp));
    return q_u + copysign(0.5 * m, dp);
}

/* Werner-Wengle (1993) capped wall shear stress magnitude with sign of u_tan */
static inline double wall_shear_stress_fast(double u_tan, double nu_eff_wall,
                                            double inv_yw, double yw_over_nu0,
                                            double inv_ww_denom) {
    double ut = fabs(u_tan);
    if (ut < 1e-15) return 0.0;
    double tau_eff = nu_eff_wall * (ut * inv_yw);
    double tau_lam = CFD_NU0 * (ut * inv_yw);
    double re_w = ut * yw_over_nu0;
    double tau_ww;
    if (re_w <= 139.4761) {
        tau_ww = tau_lam;
    } else {
        /* pow(base, 1.75) = base * sqrt(base) * sqrt(sqrt(base)) */
        double base = ut * inv_ww_denom;
        double sb = sqrt(base);
        tau_ww = base * sb * sqrt(sb);
    }
    double tau_cap = fmax(tau_lam, tau_ww);
    double tau = fmin(tau_eff, tau_cap);
    return copysign(tau, u_tan);
}

/* ========================================================================= */
/* 1. Smagorinsky-Lilly LES Eddy Viscosity with Mason-Thomson Wall Damping   */
/* ========================================================================= */

typedef struct LESCtx {
    const CFD3D  *sim;
    const double *u, *v, *w, *theta, *smoke;
    double       *nu_sgs;
    double        nu_max_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
    double        n2_max_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
    double        u_l1_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
} LESCtx;

static void les_chunk_fn(void *vctx, int k0, int k1, int tid) {
    LESCtx *ctx = (LESCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double h = sim->h;
    const double inv_h = sim->inv_h;
    const double inv_2h = 0.5 * inv_h;
    const double cs_h = sim->params.cs_smag * h;
    const double inv_cs_h2 = 1.0 / fmax(cs_h * cs_h, 1e-12);
    const double beta = sim->beta;
    const double alpha_s = sim->params.smoke_alpha_s;
    const double inv_prt = 1.0 / CFD_PR_T;
    const double half_h = 0.5 * h;
    const double nu_cap_min = fmin(0.05, 0.25 * h * 1.5);
    const double *restrict u = ctx->u;
    const double *restrict v = ctx->v;
    const double *restrict w = ctx->w;
    const double *restrict theta = ctx->theta;
    const double *restrict smoke = ctx->smoke;
    const uint8_t *restrict kx = sim->kx;
    const uint8_t *restrict ky = sim->ky;
    const uint8_t *restrict kz = sim->kz;
    const uint8_t *restrict ct = sim->cell_type;
    const double *restrict wd = sim->wall_dist;
    double *restrict nu_sgs = ctx->nu_sgs;

    double nu_max_loc = CFD_NU0;
    double n2_max_loc = 0.0;
    double u_l1_loc   = 0.0;

    for (int k = k0; k < k1; ++k) {
        double ux_inf, uy_inf;
        cfd_wind_at_z(sim, (k + 0.5) * h, &ux_inf, &uy_inf);

        for (int j = 0; j < ny; ++j) {
            const size_t row_c  = idx_c(0, j, k, nx, ny);
            const size_t row_u  = idx_u(0, j, k, nx, ny);
            const size_t row_v0 = idx_v(0, j,     k, nx, ny);
            const size_t row_v1 = idx_v(0, j + 1, k, nx, ny);
            const size_t row_w0 = idx_w(0, j, k,     nx, ny);
            const size_t row_w1 = idx_w(0, j, k + 1, nx, ny);

            const double *restrict ur  = u + row_u;
            const double *restrict v0r = v + row_v0;
            const double *restrict v1r = v + row_v1;
            const double *restrict w0r = w + row_w0;
            const double *restrict w1r = w + row_w1;
            const uint8_t *restrict kxr  = kx + row_u;
            const uint8_t *restrict ky0r = ky + row_v0;
            const uint8_t *restrict ky1r = ky + row_v1;
            const uint8_t *restrict kz0r = kz + row_w0;
            const uint8_t *restrict kz1r = kz + row_w1;
            const uint8_t *restrict ctr  = ct + row_c;
            const double *restrict wdr   = wd + row_c;
            const double *restrict thr   = theta + row_c;
            const double *restrict smr   = smoke + row_c;
            double *restrict nur         = nu_sgs + row_c;

            const double *restrict u_ym = (j > 0)      ? (u + idx_u(0, j - 1, k, nx, ny)) : NULL;
            const double *restrict u_yp = (j < ny - 1) ? (u + idx_u(0, j + 1, k, nx, ny)) : NULL;
            const double *restrict w0_ym = (j > 0)      ? (w + idx_w(0, j - 1, k,     nx, ny)) : NULL;
            const double *restrict w1_ym = (j > 0)      ? (w + idx_w(0, j - 1, k + 1, nx, ny)) : NULL;
            const double *restrict w0_yp = (j < ny - 1) ? (w + idx_w(0, j + 1, k,     nx, ny)) : NULL;
            const double *restrict w1_yp = (j < ny - 1) ? (w + idx_w(0, j + 1, k + 1, nx, ny)) : NULL;

            const double *restrict u_zm  = (k > 0)      ? (u + idx_u(0, j, k - 1, nx, ny)) : NULL;
            const double *restrict u_zp  = (k < nz - 1) ? (u + idx_u(0, j, k + 1, nx, ny)) : NULL;
            const double *restrict v0_zm = (k > 0)      ? (v + idx_v(0, j,     k - 1, nx, ny)) : NULL;
            const double *restrict v1_zm = (k > 0)      ? (v + idx_v(0, j + 1, k - 1, nx, ny)) : NULL;
            const double *restrict v0_zp = (k < nz - 1) ? (v + idx_v(0, j,     k + 1, nx, ny)) : NULL;
            const double *restrict v1_zp = (k < nz - 1) ? (v + idx_v(0, j + 1, k + 1, nx, ny)) : NULL;
            const double *restrict th_zm = (k > 0)      ? (theta + idx_c(0, j, k - 1, nx, ny)) : NULL;
            const double *restrict th_zp = (k < nz - 1) ? (theta + idx_c(0, j, k + 1, nx, ny)) : NULL;
            const double *restrict sm_zm = (k > 0)      ? (smoke + idx_c(0, j, k - 1, nx, ny)) : NULL;
            const double *restrict sm_zp = (k < nz - 1) ? (smoke + idx_c(0, j, k + 1, nx, ny)) : NULL;

            for (int i = 0; i < nx; ++i) {
                if (ctr[i] != CFD_CELL_FLUID) {
                    nur[i] = 0.0;
                    continue;
                }

                double uL = ur[i];
                double uR = ur[i + 1];
                double vL = v0r[i];
                double vR = v1r[i];
                double wB = w0r[i];
                double wT = w1r[i];

                double l1 = fmax(fabs(uL), fabs(uR))
                          + fmax(fabs(vL), fabs(vR))
                          + fmax(fabs(wB), fabs(wT));
                if (l1 > u_l1_loc) u_l1_loc = l1;

                double uc = 0.5 * (uL + uR);
                double vc = 0.5 * (vL + vR);
                double wc = 0.5 * (wB + wT);

                double sxx = (uR - uL) * inv_h;
                double syy = (vR - vL) * inv_h;
                double szz = (wT - wB) * inv_h;

                /* X neighbors of v, w */
                double vm, vp, wm, wp;
                if (i > 0 && kxr[i]) {
                    vm = 0.5 * (v0r[i - 1] + v1r[i - 1]);
                    wm = 0.5 * (w0r[i - 1] + w1r[i - 1]);
                } else if (i == 0 && kxr[0]) {
                    vm = (uL <= 0.0) ? vc : -vc;
                    wm = (uL <= 0.0) ? wc : -wc;
                } else {
                    vm = -vc;
                    wm = -wc;
                }
                if (i < nx - 1 && kxr[i + 1]) {
                    vp = 0.5 * (v0r[i + 1] + v1r[i + 1]);
                    wp = 0.5 * (w0r[i + 1] + w1r[i + 1]);
                } else if (i == nx - 1 && kxr[nx]) {
                    if (sim->bc_px == BC_INFLOW || uR < 0.0) {
                        vp = 2.0 * uy_inf - vc;
                        wp = -wc;
                    } else {
                        vp = vc;
                        wp = wc;
                    }
                } else {
                    vp = -vc;
                    wp = -wc;
                }

                /* Y neighbors of u, w */
                double um, up, wym, wyp;
                if (j > 0 && ky0r[i]) {
                    um  = 0.5 * (u_ym[i] + u_ym[i + 1]);
                    wym = 0.5 * (w0_ym[i] + w1_ym[i]);
                } else if (j == 0 && ky0r[i]) {
                    if (sim->bc_my == BC_INFLOW || vL > 0.0) {
                        um  = 2.0 * ux_inf - uc;
                        wym = -wc;
                    } else {
                        um  = uc;
                        wym = wc;
                    }
                } else {
                    um  = -uc;
                    wym = -wc;
                }
                if (j < ny - 1 && ky1r[i]) {
                    up  = 0.5 * (u_yp[i] + u_yp[i + 1]);
                    wyp = 0.5 * (w0_yp[i] + w1_yp[i]);
                } else if (j == ny - 1 && ky1r[i]) {
                    if (sim->bc_py == BC_INFLOW || vR < 0.0) {
                        up  = 2.0 * ux_inf - uc;
                        wyp = -wc;
                    } else {
                        up  = uc;
                        wyp = wc;
                    }
                } else {
                    up  = -uc;
                    wyp = -wc;
                }

                /* Z neighbors of u, v */
                double uzm, uzp, vzm, vzp;
                if (k > 0 && kz0r[i]) {
                    uzm = 0.5 * (u_zm[i] + u_zm[i + 1]);
                    vzm = 0.5 * (v0_zm[i] + v1_zm[i]);
                } else {
                    uzm = -uc;
                    vzm = -vc;
                }
                if (k < nz - 1 && kz1r[i]) {
                    uzp = 0.5 * (u_zp[i] + u_zp[i + 1]);
                    vzp = 0.5 * (v0_zp[i] + v1_zp[i]);
                } else if (k == nz - 1 && kz1r[i]) {
                    if (wT < 0.0) {
                        uzp = 2.0 * ux_inf - uc;
                        vzp = 2.0 * uy_inf - vc;
                    } else {
                        uzp = uc;
                        vzp = vc;
                    }
                } else {
                    uzp = -uc;
                    vzp = -vc;
                }

                double sxy = 0.5 * ((up - um) + (vp - vm)) * inv_2h;
                double sxz = 0.5 * ((uzp - uzm) + (wp - wm)) * inv_2h;
                double syz = 0.5 * ((vzp - vzm) + (wyp - wym)) * inv_2h;

                double s2 = 2.0 * (sxx * sxx + syy * syy + szz * szz)
                          + 4.0 * (sxy * sxy + sxz * sxz + syz * syz);

                /* Vertical buoyancy stratification N^2_BV */
                int open_bot = (k > 0)      && (kz0r[i] != 0);
                int open_top = (k < nz - 1) && (kz1r[i] != 0);
                double dth_dz = 0.0, dsm_dz = 0.0;
                if (open_bot && open_top) {
                    dth_dz = (th_zp[i] - th_zm[i]) * inv_2h;
                    dsm_dz = (sm_zp[i] - sm_zm[i]) * inv_2h;
                } else if (open_top) {
                    dth_dz = (th_zp[i] - thr[i]) * inv_h;
                    dsm_dz = (sm_zp[i] - smr[i]) * inv_h;
                } else if (open_bot) {
                    dth_dz = (thr[i] - th_zm[i]) * inv_h;
                    dsm_dz = (smr[i] - sm_zm[i]) * inv_h;
                }

                double n2_bv = CFD_G * (beta * dth_dz - alpha_s * dsm_dz);
                double n2_abs = CFD_G * (fabs(beta * dth_dz) + fabs(alpha_s * dsm_dz));
                if (n2_abs > n2_max_loc) n2_max_loc = n2_abs;

                /* Mason-Thomson wall-damped mixing length (single division) */
                double kw = CFD_KAPPA_VK * fmax(wdr[i], half_h);
                double kw2 = kw * kw;
                double l_sq = kw2 / (inv_cs_h2 * kw2 + 1.0);

                /* Lilly buoyancy-modified Smagorinsky */
                double arg = s2 - n2_bv * inv_prt;
                double nu_t = (arg > 0.0) ? (l_sq * sqrt(arg)) : 0.0;

                /* Fast-path physical cap (skips sqrt for ~98% of cells) */
                if (nu_t > nu_cap_min) {
                    double spd = sqrt(uc * uc + vc * vc + wc * wc);
                    double nu_cap = fmin(0.05, 0.25 * h * fmax(1.5, spd));
                    if (nu_t > nu_cap) nu_t = nu_cap;
                }

                nur[i] = nu_t;

                double diff_eff = fmax(CFD_NU0 + nu_t, CFD_ALPHA0 + nu_t * inv_prt);
                if (diff_eff > nu_max_loc) nu_max_loc = diff_eff;
            }
        }
    }

    ctx->nu_max_t[tid * CFD_CACHE_PAD] = nu_max_loc;
    ctx->n2_max_t[tid * CFD_CACHE_PAD] = n2_max_loc;
    ctx->u_l1_t[tid * CFD_CACHE_PAD]   = u_l1_loc;
}

void cfd_compute_les_viscosity(CFD3D *sim, const double *u, const double *v, const double *w,
                               const double *theta, const double *smoke, double *nu_sgs,
                               double *out_nu_max, double *out_n2_max, double *out_u_l1_max) {
    LESCtx ctx;
    ctx.sim = sim;
    ctx.u = u; ctx.v = v; ctx.w = w;
    ctx.theta = theta; ctx.smoke = smoke;
    ctx.nu_sgs = nu_sgs;

    int nt = cfd_parallel_for_k(sim, sim->nz, 8, &ctx, les_chunk_fn);

    double nu_max = CFD_ALPHA0;
    double n2_max = 0.0;
    double u_l1   = 0.0;
    for (int t = 0; t < nt; ++t) {
        if (ctx.nu_max_t[t * CFD_CACHE_PAD] > nu_max) nu_max = ctx.nu_max_t[t * CFD_CACHE_PAD];
        if (ctx.n2_max_t[t * CFD_CACHE_PAD] > n2_max) n2_max = ctx.n2_max_t[t * CFD_CACHE_PAD];
        if (ctx.u_l1_t[t * CFD_CACHE_PAD]   > u_l1)   u_l1   = ctx.u_l1_t[t * CFD_CACHE_PAD];
    }
    if (out_nu_max)   *out_nu_max   = nu_max;
    if (out_n2_max)   *out_n2_max   = n2_max;
    if (out_u_l1_max) *out_u_l1_max = u_l1;
}

/* ========================================================================= */
/* 2. Momentum RHS: Van Leer TVD Advection + Variable-Viscosity Stress + B   */
/* ========================================================================= */

typedef struct MomCtx {
    const CFD3D  *sim;
    const double *u, *v, *w, *theta, *smoke, *nu_sgs;
    double       *ru, *rv, *rw;
    double        inv_yw, yw_over_nu0, inv_ww_denom;
} MomCtx;

#define wall_shear_stress(u_, nu_, h_) \
    wall_shear_stress_fast((u_), (nu_), inv_yw, yw_over_nu0, inv_ww_denom)

static void mom_uv_slab(const MomCtx *ctx, int k) {
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double h = sim->h;
    const double inv_h = sim->inv_h;
    const double inv_yw = ctx->inv_yw;
    const double yw_over_nu0 = ctx->yw_over_nu0;
    const double inv_ww_denom = ctx->inv_ww_denom;
    const double *restrict u = ctx->u;
    const double *restrict v = ctx->v;
    const double *restrict w = ctx->w;
    const double *restrict nu_sgs = ctx->nu_sgs;
    const uint8_t *restrict kx = sim->kx;
    const uint8_t *restrict ky = sim->ky;
    const uint8_t *restrict kz = sim->kz;
    double *restrict ru = ctx->ru;
    double *restrict rv = ctx->rv;

    double ux_inf, uy_inf;
    cfd_wind_at_z(sim, (k + 0.5) * h, &ux_inf, &uy_inf);

    const int jet_on = sim->params.smoke_enabled && (sim->params.smoke_jet_speed > 1e-6);
    const double jet_spd = sim->params.smoke_jet_speed;
    const double jet_ux = jet_spd * sim->emit_jx;
    const double jet_uy = jet_spd * sim->emit_jy;
    const double inv_2sig2 = 0.5 / (sim->emit_sigma * sim->emit_sigma);
    const double rcut2 = (3.5 * sim->emit_sigma) * (3.5 * sim->emit_sigma);
    const double jet_inv_tau = 1.0 / fmax(0.1, 2.0 * sim->emit_sigma / fmax(jet_spd, 0.5));
    const double zc = (k + 0.5) * h;

    /* -------------------- U-MOMENTUM -------------------- */
    for (int j = 0; j < ny; ++j) {
        double yc = (j + 0.5) * h;
        for (int i = 0; i <= nx; ++i) {
            size_t fu = idx_u(i, j, k, nx, ny);
            if (!kx[fu]) {
                ru[fu] = 0.0;
                continue;
            }
            double uc = u[fu];

            /* Boundary faces i = 0 (BC_VENT) and i = nx (BC_OPEN / BC_INFLOW) */
            if (i == 0) {
                /* BC_VENT: outflow convects upwind, inflow driven by phi_vent */
                double adv_x = (uc < 0.0) ? (-uc * (u[idx_u(1, j, k, nx, ny)] - uc) * inv_h) : 0.0;
                ru[fu] = adv_x;
                continue;
            }
            if (i == nx) {
                if (sim->bc_px == BC_INFLOW) {
                    ru[fu] = 0.0;
                    continue;
                }
                /* BC_OPEN on +X */
                double adv = 0.0;
                if (uc > 0.0) {
                    adv -= uc * (uc - u[idx_u(nx - 1, j, k, nx, ny)]) * inv_h;
                }
                double vc_b = 0.5 * (v[idx_v(nx - 1, j, k, nx, ny)] + v[idx_v(nx - 1, j + 1, k, nx, ny)]);
                double wc_b = 0.5 * (w[idx_w(nx - 1, j, k, nx, ny)] + w[idx_w(nx - 1, j, k + 1, nx, ny)]);
                if (vc_b > 0.0) {
                    double um_y = (j > 0 && kx[idx_u(nx, j - 1, k, nx, ny)]) ? u[idx_u(nx, j - 1, k, nx, ny)] : ux_inf;
                    adv -= vc_b * (uc - um_y) * inv_h;
                } else if (vc_b < 0.0) {
                    double up_y = (j < ny - 1 && kx[idx_u(nx, j + 1, k, nx, ny)]) ? u[idx_u(nx, j + 1, k, nx, ny)] : ux_inf;
                    adv -= vc_b * (up_y - uc) * inv_h;
                }
                if (wc_b > 0.0) {
                    double um_z = (k > 0 && kx[idx_u(nx, j, k - 1, nx, ny)]) ? u[idx_u(nx, j, k - 1, nx, ny)] : 0.0;
                    adv -= wc_b * (uc - um_z) * inv_h;
                } else if (wc_b < 0.0) {
                    double up_z = (k < nz - 1 && kx[idx_u(nx, j, k + 1, nx, ny)]) ? u[idx_u(nx, j, k + 1, nx, ny)] : ux_inf;
                    adv -= wc_b * (up_z - uc) * inv_h;
                }
                ru[fu] = adv;
                continue;
            }

            /* Interior open u-face: 1 <= i <= nx-1 */
            size_t cL = idx_c(i - 1, j, k, nx, ny);
            size_t cR = idx_c(i,     j, k, nx, ny);
            double nuL = CFD_NU0 + nu_sgs[cL];
            double nuR = CFD_NU0 + nu_sgs[cR];

            /* X-advection fluxes at cell centers cR (i) and cL (i-1) */
            double uR = u[idx_u(i + 1, j, k, nx, ny)];
            double uL = u[idx_u(i - 1, j, k, nx, ny)];

            double vel_xp = 0.5 * (uc + uR);
            double u_fxp;
            if (vel_xp >= 0.0) {
                int has_uu = (i >= 1) && (kx[idx_u(i - 1, j, k, nx, ny)] != 0);
                u_fxp = van_leer_face(uL, uc, uR, has_uu);
            } else {
                int has_uu = (i + 2 <= nx) && (kx[idx_u(i + 1, j, k, nx, ny)] != 0)
                                           && (kx[idx_u(i + 2, j, k, nx, ny)] != 0);
                double uRR = has_uu ? u[idx_u(i + 2, j, k, nx, ny)] : uR;
                u_fxp = van_leer_face(uRR, uR, uc, has_uu);
            }
            double fx_p = vel_xp * u_fxp;

            double vel_xm = 0.5 * (uL + uc);
            double u_fxm;
            if (vel_xm >= 0.0) {
                int has_uu = (i >= 2) && (kx[idx_u(i - 1, j, k, nx, ny)] != 0)
                                      && (kx[idx_u(i - 2, j, k, nx, ny)] != 0);
                double uLL = has_uu ? u[idx_u(i - 2, j, k, nx, ny)] : uL;
                u_fxm = van_leer_face(uLL, uL, uc, has_uu);
            } else {
                int has_uu = (i + 1 <= nx) && (kx[idx_u(i + 1, j, k, nx, ny)] != 0);
                u_fxm = van_leer_face(uR, uc, uL, has_uu);
            }
            double fx_m = vel_xm * u_fxm;

            /* Y-advection and viscous shear at j+1/2 and j-1/2 */
            double fy_p = 0.0, fy_m = 0.0;
            double tau_xy_p = 0.0, tau_xy_m = 0.0;

            /* +y edge (j+1) */
            int kyL_p = ky[idx_v(i - 1, j + 1, k, nx, ny)];
            int kyR_p = ky[idx_v(i,     j + 1, k, nx, ny)];
            double v_ep = 0.5 * (v[idx_v(i - 1, j + 1, k, nx, ny)] + v[idx_v(i, j + 1, k, nx, ny)]);
            if (j < ny - 1) {
                if (kyL_p && kyR_p && kx[idx_u(i, j + 1, k, nx, ny)]) {
                    double u_yp = u[idx_u(i, j + 1, k, nx, ny)];
                    double u_fyp;
                    if (v_ep >= 0.0) {
                        int has_uu = (j > 0) && ky[idx_v(i - 1, j, k, nx, ny)]
                                             && ky[idx_v(i, j, k, nx, ny)]
                                             && kx[idx_u(i, j - 1, k, nx, ny)];
                        double u_ym = has_uu ? u[idx_u(i, j - 1, k, nx, ny)] : uc;
                        u_fyp = van_leer_face(u_ym, uc, u_yp, has_uu);
                    } else {
                        int has_uu = (j + 2 < ny) && ky[idx_v(i - 1, j + 2, k, nx, ny)]
                                                  && ky[idx_v(i, j + 2, k, nx, ny)]
                                                  && kx[idx_u(i, j + 2, k, nx, ny)];
                        double u_ypp = has_uu ? u[idx_u(i, j + 2, k, nx, ny)] : u_yp;
                        u_fyp = van_leer_face(u_ypp, u_yp, uc, has_uu);
                    }
                    fy_p = v_ep * u_fyp;
                    double nu_e = CFD_NU0 + 0.25 * (nu_sgs[cL] + nu_sgs[cR]
                                + nu_sgs[idx_c(i - 1, j + 1, k, nx, ny)]
                                + nu_sgs[idx_c(i,     j + 1, k, nx, ny)]);
                    double dv_dx = (v[idx_v(i, j + 1, k, nx, ny)] - v[idx_v(i - 1, j + 1, k, nx, ny)]) * inv_h;
                    tau_xy_p = nu_e * ((u_yp - uc) * inv_h + dv_dx);
                } else {
                    /* Wall or sash edge */
                    if (kyL_p || kyR_p) {
                        fy_p = (v_ep >= 0.0) ? (v_ep * uc) : 0.0;
                    }
                    tau_xy_p = -wall_shear_stress(uc, 0.5 * (nuL + nuR), h);
                }
            } else {
                /* Domain +Y boundary */
                if (kyL_p || kyR_p) {
                    int is_inflow = (sim->bc_py == BC_INFLOW) || (v_ep < 0.0);
                    fy_p = v_ep * (is_inflow ? ux_inf : uc);
                    double nu_e = 0.5 * (nuL + nuR);
                    double dv_dx = (v[idx_v(i, ny, k, nx, ny)] - v[idx_v(i - 1, ny, k, nx, ny)]) * inv_h;
                    double du_dy = is_inflow ? (2.0 * (ux_inf - uc) * inv_h) : 0.0;
                    tau_xy_p = nu_e * (du_dy + dv_dx);
                } else {
                    tau_xy_p = -wall_shear_stress(uc, 0.5 * (nuL + nuR), h);
                }
            }

            /* -y edge (j) */
            int kyL_m = ky[idx_v(i - 1, j, k, nx, ny)];
            int kyR_m = ky[idx_v(i,     j, k, nx, ny)];
            double v_em = 0.5 * (v[idx_v(i - 1, j, k, nx, ny)] + v[idx_v(i, j, k, nx, ny)]);
            if (j > 0) {
                if (kyL_m && kyR_m && kx[idx_u(i, j - 1, k, nx, ny)]) {
                    double u_ym = u[idx_u(i, j - 1, k, nx, ny)];
                    double u_fym;
                    if (v_em >= 0.0) {
                        int has_uu = (j >= 2) && ky[idx_v(i - 1, j - 1, k, nx, ny)]
                                              && ky[idx_v(i, j - 1, k, nx, ny)]
                                              && kx[idx_u(i, j - 2, k, nx, ny)];
                        double u_ymm = has_uu ? u[idx_u(i, j - 2, k, nx, ny)] : u_ym;
                        u_fym = van_leer_face(u_ymm, u_ym, uc, has_uu);
                    } else {
                        int has_uu = (j + 1 < ny) && ky[idx_v(i - 1, j + 1, k, nx, ny)]
                                                  && ky[idx_v(i, j + 1, k, nx, ny)]
                                                  && kx[idx_u(i, j + 1, k, nx, ny)];
                        double u_yp = has_uu ? u[idx_u(i, j + 1, k, nx, ny)] : uc;
                        u_fym = van_leer_face(u_yp, uc, u_ym, has_uu);
                    }
                    fy_m = v_em * u_fym;
                    double nu_e = CFD_NU0 + 0.25 * (nu_sgs[cL] + nu_sgs[cR]
                                + nu_sgs[idx_c(i - 1, j - 1, k, nx, ny)]
                                + nu_sgs[idx_c(i,     j - 1, k, nx, ny)]);
                    double dv_dx = (v[idx_v(i, j, k, nx, ny)] - v[idx_v(i - 1, j, k, nx, ny)]) * inv_h;
                    tau_xy_m = nu_e * ((uc - u_ym) * inv_h + dv_dx);
                } else {
                    if (kyL_m || kyR_m) {
                        fy_m = (v_em < 0.0) ? (v_em * uc) : 0.0;
                    }
                    tau_xy_m = wall_shear_stress(uc, 0.5 * (nuL + nuR), h);
                }
            } else {
                /* Domain -Y boundary */
                if (kyL_m || kyR_m) {
                    int is_inflow = (sim->bc_my == BC_INFLOW) || (v_em > 0.0);
                    fy_m = v_em * (is_inflow ? ux_inf : uc);
                    double nu_e = 0.5 * (nuL + nuR);
                    double dv_dx = (v[idx_v(i, 0, k, nx, ny)] - v[idx_v(i - 1, 0, k, nx, ny)]) * inv_h;
                    double du_dy = is_inflow ? (2.0 * (uc - ux_inf) * inv_h) : 0.0;
                    tau_xy_m = nu_e * (du_dy + dv_dx);
                } else {
                    tau_xy_m = wall_shear_stress(uc, 0.5 * (nuL + nuR), h);
                }
            }

            /* Z-advection and viscous shear at k+1/2 and k-1/2 */
            double fz_p = 0.0, fz_m = 0.0;
            double tau_xz_p = 0.0, tau_xz_m = 0.0;

            /* +z edge (k+1) */
            int kzL_p = kz[idx_w(i - 1, j, k + 1, nx, ny)];
            int kzR_p = kz[idx_w(i,     j, k + 1, nx, ny)];
            double w_ep = 0.5 * (w[idx_w(i - 1, j, k + 1, nx, ny)] + w[idx_w(i, j, k + 1, nx, ny)]);
            if (k < nz - 1) {
                if (kzL_p && kzR_p && kx[idx_u(i, j, k + 1, nx, ny)]) {
                    double u_zp = u[idx_u(i, j, k + 1, nx, ny)];
                    double u_fzp;
                    if (w_ep >= 0.0) {
                        int has_uu = (k > 0) && kz[idx_w(i - 1, j, k, nx, ny)]
                                             && kz[idx_w(i, j, k, nx, ny)]
                                             && kx[idx_u(i, j, k - 1, nx, ny)];
                        double u_zm = has_uu ? u[idx_u(i, j, k - 1, nx, ny)] : uc;
                        u_fzp = van_leer_face(u_zm, uc, u_zp, has_uu);
                    } else {
                        int has_uu = (k + 2 < nz) && kz[idx_w(i - 1, j, k + 2, nx, ny)]
                                                  && kz[idx_w(i, j, k + 2, nx, ny)]
                                                  && kx[idx_u(i, j, k + 2, nx, ny)];
                        double u_zpp = has_uu ? u[idx_u(i, j, k + 2, nx, ny)] : u_zp;
                        u_fzp = van_leer_face(u_zpp, u_zp, uc, has_uu);
                    }
                    fz_p = w_ep * u_fzp;
                    double nu_e = CFD_NU0 + 0.25 * (nu_sgs[cL] + nu_sgs[cR]
                                + nu_sgs[idx_c(i - 1, j, k + 1, nx, ny)]
                                + nu_sgs[idx_c(i,     j, k + 1, nx, ny)]);
                    double dw_dx = (w[idx_w(i, j, k + 1, nx, ny)] - w[idx_w(i - 1, j, k + 1, nx, ny)]) * inv_h;
                    tau_xz_p = nu_e * ((u_zp - uc) * inv_h + dw_dx);
                } else {
                    if (kzL_p || kzR_p) {
                        fz_p = (w_ep >= 0.0) ? (w_ep * uc) : 0.0;
                    }
                    tau_xz_p = -wall_shear_stress(uc, 0.5 * (nuL + nuR), h);
                }
            } else {
                /* Domain +Z open sky */
                if (kzL_p || kzR_p) {
                    int is_inflow = (w_ep < 0.0);
                    fz_p = w_ep * (is_inflow ? ux_inf : uc);
                    double nu_e = 0.5 * (nuL + nuR);
                    double dw_dx = (w[idx_w(i, j, nz, nx, ny)] - w[idx_w(i - 1, j, nz, nx, ny)]) * inv_h;
                    double du_dz = is_inflow ? (2.0 * (ux_inf - uc) * inv_h) : 0.0;
                    tau_xz_p = nu_e * (du_dz + dw_dx);
                } else {
                    tau_xz_p = -wall_shear_stress(uc, 0.5 * (nuL + nuR), h);
                }
            }

            /* -z edge (k) */
            int kzL_m = kz[idx_w(i - 1, j, k, nx, ny)];
            int kzR_m = kz[idx_w(i,     j, k, nx, ny)];
            double w_em = 0.5 * (w[idx_w(i - 1, j, k, nx, ny)] + w[idx_w(i, j, k, nx, ny)]);
            if (k > 0) {
                if (kzL_m && kzR_m && kx[idx_u(i, j, k - 1, nx, ny)]) {
                    double u_zm = u[idx_u(i, j, k - 1, nx, ny)];
                    double u_fzm;
                    if (w_em >= 0.0) {
                        int has_uu = (k >= 2) && kz[idx_w(i - 1, j, k - 1, nx, ny)]
                                              && kz[idx_w(i, j, k - 1, nx, ny)]
                                              && kx[idx_u(i, j, k - 2, nx, ny)];
                        double u_zmm = has_uu ? u[idx_u(i, j, k - 2, nx, ny)] : u_zm;
                        u_fzm = van_leer_face(u_zmm, u_zm, uc, has_uu);
                    } else {
                        int has_uu = (k + 1 < nz) && kz[idx_w(i - 1, j, k + 1, nx, ny)]
                                                  && kz[idx_w(i, j, k + 1, nx, ny)]
                                                  && kx[idx_u(i, j, k + 1, nx, ny)];
                        double u_zp = has_uu ? u[idx_u(i, j, k + 1, nx, ny)] : uc;
                        u_fzm = van_leer_face(u_zp, uc, u_zm, has_uu);
                    }
                    fz_m = w_em * u_fzm;
                    double nu_e = CFD_NU0 + 0.25 * (nu_sgs[cL] + nu_sgs[cR]
                                + nu_sgs[idx_c(i - 1, j, k - 1, nx, ny)]
                                + nu_sgs[idx_c(i,     j, k - 1, nx, ny)]);
                    double dw_dx = (w[idx_w(i, j, k, nx, ny)] - w[idx_w(i - 1, j, k, nx, ny)]) * inv_h;
                    tau_xz_m = nu_e * ((uc - u_zm) * inv_h + dw_dx);
                } else {
                    if (kzL_m || kzR_m) {
                        fz_m = (w_em < 0.0) ? (w_em * uc) : 0.0;
                    }
                    tau_xz_m = wall_shear_stress(uc, 0.5 * (nuL + nuR), h);
                }
            } else {
                /* Solid floor/ground at k = 0 */
                tau_xz_m = wall_shear_stress(uc, 0.5 * (nuL + nuR), h);
            }

            /* Normal stresses tau_xx */
            double tau_xx_p = 2.0 * nuR * (uR - uc) * inv_h;
            double tau_xx_m = 2.0 * nuL * (uc - uL) * inv_h;

            double rhs = (-(fx_p - fx_m) - (fy_p - fy_m) - (fz_p - fz_m)
                          + (tau_xx_p - tau_xx_m) + (tau_xy_p - tau_xy_m) + (tau_xz_p - tau_xz_m)) * inv_h;

            /* Optional smooth Gaussian jet body force */
            if (jet_on && k >= sim->emit_k0 && k <= sim->emit_k1 &&
                j >= sim->emit_j0 && j <= sim->emit_j1 &&
                i >= sim->emit_i0 && i <= sim->emit_i1 + 1) {
                double dx = i * h - sim->emit_wx;
                double dy = yc - sim->emit_wy;
                double dz = zc - sim->emit_wz;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < rcut2) {
                    rhs += jet_inv_tau * exp(-r2 * inv_2sig2) * (jet_ux - uc);
                }
            }

            ru[fu] = rhs;
        }
    }

    /* -------------------- V-MOMENTUM -------------------- */
    for (int j = 0; j <= ny; ++j) {
        double yf = j * h;
        for (int i = 0; i < nx; ++i) {
            size_t fv = idx_v(i, j, k, nx, ny);
            if (!ky[fv]) {
                rv[fv] = 0.0;
                continue;
            }
            double vc = v[fv];

            /* Boundary faces j = 0 and j = ny */
            if (j == 0) {
                if (sim->bc_my == BC_INFLOW) {
                    rv[fv] = 0.0;
                    continue;
                }
                double adv = 0.0;
                if (vc < 0.0) { /* outflow in -y */
                    adv -= vc * (v[idx_v(i, 1, k, nx, ny)] - vc) * inv_h;
                }
                double uc_b = 0.5 * (u[idx_u(i, 0, k, nx, ny)] + u[idx_u(i + 1, 0, k, nx, ny)]);
                double wc_b = 0.5 * (w[idx_w(i, 0, k, nx, ny)] + w[idx_w(i, 0, k + 1, nx, ny)]);
                if (uc_b > 0.0) {
                    double vm_x = (i > 0) ? (ky[idx_v(i - 1, 0, k, nx, ny)] ? v[idx_v(i - 1, 0, k, nx, ny)] : 0.0) : uy_inf;
                    adv -= uc_b * (vc - vm_x) * inv_h;
                } else if (uc_b < 0.0) {
                    double vp_x = (i < nx - 1) ? (ky[idx_v(i + 1, 0, k, nx, ny)] ? v[idx_v(i + 1, 0, k, nx, ny)] : 0.0) : uy_inf;
                    adv -= uc_b * (vp_x - vc) * inv_h;
                }
                if (wc_b > 0.0) {
                    double vm_z = (k > 0 && ky[idx_v(i, 0, k - 1, nx, ny)]) ? v[idx_v(i, 0, k - 1, nx, ny)] : 0.0;
                    adv -= wc_b * (vc - vm_z) * inv_h;
                } else if (wc_b < 0.0) {
                    double vp_z = (k < nz - 1 && ky[idx_v(i, 0, k + 1, nx, ny)]) ? v[idx_v(i, 0, k + 1, nx, ny)] : uy_inf;
                    adv -= wc_b * (vp_z - vc) * inv_h;
                }
                rv[fv] = adv;
                continue;
            }
            if (j == ny) {
                if (sim->bc_py == BC_INFLOW) {
                    rv[fv] = 0.0;
                    continue;
                }
                double adv = 0.0;
                if (vc > 0.0) { /* outflow in +y */
                    adv -= vc * (vc - v[idx_v(i, ny - 1, k, nx, ny)]) * inv_h;
                }
                double uc_b = 0.5 * (u[idx_u(i, ny - 1, k, nx, ny)] + u[idx_u(i + 1, ny - 1, k, nx, ny)]);
                double wc_b = 0.5 * (w[idx_w(i, ny - 1, k, nx, ny)] + w[idx_w(i, ny - 1, k + 1, nx, ny)]);
                if (uc_b > 0.0) {
                    double vm_x = (i > 0) ? (ky[idx_v(i - 1, ny, k, nx, ny)] ? v[idx_v(i - 1, ny, k, nx, ny)] : 0.0) : uy_inf;
                    adv -= uc_b * (vc - vm_x) * inv_h;
                } else if (uc_b < 0.0) {
                    double vp_x = (i < nx - 1) ? (ky[idx_v(i + 1, ny, k, nx, ny)] ? v[idx_v(i + 1, ny, k, nx, ny)] : 0.0) : uy_inf;
                    adv -= uc_b * (vp_x - vc) * inv_h;
                }
                if (wc_b > 0.0) {
                    double vm_z = (k > 0 && ky[idx_v(i, ny, k - 1, nx, ny)]) ? v[idx_v(i, ny, k - 1, nx, ny)] : 0.0;
                    adv -= wc_b * (vc - vm_z) * inv_h;
                } else if (wc_b < 0.0) {
                    double vp_z = (k < nz - 1 && ky[idx_v(i, ny, k + 1, nx, ny)]) ? v[idx_v(i, ny, k + 1, nx, ny)] : uy_inf;
                    adv -= wc_b * (vp_z - vc) * inv_h;
                }
                rv[fv] = adv;
                continue;
            }

            /* Interior open v-face: 1 <= j <= ny-1 */
            size_t cL = idx_c(i, j - 1, k, nx, ny);
            size_t cR = idx_c(i, j,     k, nx, ny);
            double nuL = CFD_NU0 + nu_sgs[cL];
            double nuR = CFD_NU0 + nu_sgs[cR];

            /* Y-advection fluxes */
            double vR = v[idx_v(i, j + 1, k, nx, ny)];
            double vL = v[idx_v(i, j - 1, k, nx, ny)];

            double vel_yp = 0.5 * (vc + vR);
            double v_fyp;
            if (vel_yp >= 0.0) {
                int has_uu = (j >= 1) && (ky[idx_v(i, j - 1, k, nx, ny)] != 0);
                v_fyp = van_leer_face(vL, vc, vR, has_uu);
            } else {
                int has_uu = (j + 2 <= ny) && (ky[idx_v(i, j + 1, k, nx, ny)] != 0)
                                           && (ky[idx_v(i, j + 2, k, nx, ny)] != 0);
                double vRR = has_uu ? v[idx_v(i, j + 2, k, nx, ny)] : vR;
                v_fyp = van_leer_face(vRR, vR, vc, has_uu);
            }
            double fy_p = vel_yp * v_fyp;

            double vel_ym = 0.5 * (vL + vc);
            double v_fym;
            if (vel_ym >= 0.0) {
                int has_uu = (j >= 2) && (ky[idx_v(i, j - 1, k, nx, ny)] != 0)
                                      && (ky[idx_v(i, j - 2, k, nx, ny)] != 0);
                double vLL = has_uu ? v[idx_v(i, j - 2, k, nx, ny)] : vL;
                v_fym = van_leer_face(vLL, vL, vc, has_uu);
            } else {
                int has_uu = (j + 1 <= ny) && (ky[idx_v(i, j + 1, k, nx, ny)] != 0);
                v_fym = van_leer_face(vR, vc, vL, has_uu);
            }
            double fy_m = vel_ym * v_fym;

            /* X-advection and viscous shear at i+1/2 and i-1/2 */
            double fx_p = 0.0, fx_m = 0.0;
            double tau_yx_p = 0.0, tau_yx_m = 0.0;

            /* +x edge (i+1) */
            int kxL_p = kx[idx_u(i + 1, j - 1, k, nx, ny)];
            int kxR_p = kx[idx_u(i + 1, j,     k, nx, ny)];
            double u_ep = 0.5 * (u[idx_u(i + 1, j - 1, k, nx, ny)] + u[idx_u(i + 1, j, k, nx, ny)]);
            if (i < nx - 1) {
                if (kxL_p && kxR_p && ky[idx_v(i + 1, j, k, nx, ny)]) {
                    double v_xp = v[idx_v(i + 1, j, k, nx, ny)];
                    double v_fxp;
                    if (u_ep >= 0.0) {
                        int has_uu = (i > 0) && kx[idx_u(i, j - 1, k, nx, ny)]
                                             && kx[idx_u(i, j, k, nx, ny)]
                                             && ky[idx_v(i - 1, j, k, nx, ny)];
                        double v_xm = has_uu ? v[idx_v(i - 1, j, k, nx, ny)] : vc;
                        v_fxp = van_leer_face(v_xm, vc, v_xp, has_uu);
                    } else {
                        int has_uu = (i + 2 < nx) && kx[idx_u(i + 2, j - 1, k, nx, ny)]
                                                  && kx[idx_u(i + 2, j, k, nx, ny)]
                                                  && ky[idx_v(i + 2, j, k, nx, ny)];
                        double v_xpp = has_uu ? v[idx_v(i + 2, j, k, nx, ny)] : v_xp;
                        v_fxp = van_leer_face(v_xpp, v_xp, vc, has_uu);
                    }
                    fx_p = u_ep * v_fxp;
                    double nu_e = CFD_NU0 + 0.25 * (nu_sgs[cL] + nu_sgs[cR]
                                + nu_sgs[idx_c(i + 1, j - 1, k, nx, ny)]
                                + nu_sgs[idx_c(i + 1, j,     k, nx, ny)]);
                    double du_dy = (u[idx_u(i + 1, j, k, nx, ny)] - u[idx_u(i + 1, j - 1, k, nx, ny)]) * inv_h;
                    tau_yx_p = nu_e * ((v_xp - vc) * inv_h + du_dy);
                } else {
                    if (kxL_p || kxR_p) {
                        fx_p = (u_ep >= 0.0) ? (u_ep * vc) : 0.0;
                    }
                    tau_yx_p = -wall_shear_stress(vc, 0.5 * (nuL + nuR), h);
                }
            } else {
                /* Domain +X boundary */
                if (kxL_p || kxR_p) {
                    int is_inflow = (sim->bc_px == BC_INFLOW) || (u_ep < 0.0);
                    fx_p = u_ep * (is_inflow ? uy_inf : vc);
                    double nu_e = 0.5 * (nuL + nuR);
                    double du_dy = (u[idx_u(nx, j, k, nx, ny)] - u[idx_u(nx, j - 1, k, nx, ny)]) * inv_h;
                    double dv_dx = is_inflow ? (2.0 * (uy_inf - vc) * inv_h) : 0.0;
                    tau_yx_p = nu_e * (dv_dx + du_dy);
                } else {
                    tau_yx_p = -wall_shear_stress(vc, 0.5 * (nuL + nuR), h);
                }
            }

            /* -x edge (i) */
            int kxL_m = kx[idx_u(i, j - 1, k, nx, ny)];
            int kxR_m = kx[idx_u(i, j,     k, nx, ny)];
            double u_em = 0.5 * (u[idx_u(i, j - 1, k, nx, ny)] + u[idx_u(i, j, k, nx, ny)]);
            if (i > 0) {
                if (kxL_m && kxR_m && ky[idx_v(i - 1, j, k, nx, ny)]) {
                    double v_xm = v[idx_v(i - 1, j, k, nx, ny)];
                    double v_fxm;
                    if (u_em >= 0.0) {
                        int has_uu = (i >= 2) && kx[idx_u(i - 1, j - 1, k, nx, ny)]
                                              && kx[idx_u(i - 1, j, k, nx, ny)]
                                              && ky[idx_v(i - 2, j, k, nx, ny)];
                        double v_xmm = has_uu ? v[idx_v(i - 2, j, k, nx, ny)] : v_xm;
                        v_fxm = van_leer_face(v_xmm, v_xm, vc, has_uu);
                    } else {
                        int has_uu = (i + 1 < nx) && kx[idx_u(i + 1, j - 1, k, nx, ny)]
                                                  && kx[idx_u(i + 1, j, k, nx, ny)]
                                                  && ky[idx_v(i + 1, j, k, nx, ny)];
                        double v_xp = has_uu ? v[idx_v(i + 1, j, k, nx, ny)] : vc;
                        v_fxm = van_leer_face(v_xp, vc, v_xm, has_uu);
                    }
                    fx_m = u_em * v_fxm;
                    double nu_e = CFD_NU0 + 0.25 * (nu_sgs[cL] + nu_sgs[cR]
                                + nu_sgs[idx_c(i - 1, j - 1, k, nx, ny)]
                                + nu_sgs[idx_c(i - 1, j,     k, nx, ny)]);
                    double du_dy = (u[idx_u(i, j, k, nx, ny)] - u[idx_u(i, j - 1, k, nx, ny)]) * inv_h;
                    tau_yx_m = nu_e * ((vc - v_xm) * inv_h + du_dy);
                } else {
                    if (kxL_m || kxR_m) {
                        fx_m = (u_em < 0.0) ? (u_em * vc) : 0.0;
                    }
                    tau_yx_m = wall_shear_stress(vc, 0.5 * (nuL + nuR), h);
                }
            } else {
                /* Domain -X: solid wall or BC_VENT */
                if (kxL_m || kxR_m) {
                    fx_m = (u_em < 0.0) ? (u_em * vc) : 0.0;
                    double nu_e = 0.5 * (nuL + nuR);
                    double du_dy = (u[idx_u(0, j, k, nx, ny)] - u[idx_u(0, j - 1, k, nx, ny)]) * inv_h;
                    double dv_dx = (u_em > 0.0) ? (2.0 * vc * inv_h) : 0.0;
                    tau_yx_m = nu_e * (dv_dx + du_dy);
                } else {
                    tau_yx_m = wall_shear_stress(vc, 0.5 * (nuL + nuR), h);
                }
            }

            /* Z-advection and viscous shear at k+1/2 and k-1/2 */
            double fz_p = 0.0, fz_m = 0.0;
            double tau_yz_p = 0.0, tau_yz_m = 0.0;

            int kzL_p = kz[idx_w(i, j - 1, k + 1, nx, ny)];
            int kzR_p = kz[idx_w(i, j,     k + 1, nx, ny)];
            double w_ep = 0.5 * (w[idx_w(i, j - 1, k + 1, nx, ny)] + w[idx_w(i, j, k + 1, nx, ny)]);
            if (k < nz - 1) {
                if (kzL_p && kzR_p && ky[idx_v(i, j, k + 1, nx, ny)]) {
                    double v_zp = v[idx_v(i, j, k + 1, nx, ny)];
                    double v_fzp;
                    if (w_ep >= 0.0) {
                        int has_uu = (k > 0) && kz[idx_w(i, j - 1, k, nx, ny)]
                                             && kz[idx_w(i, j, k, nx, ny)]
                                             && ky[idx_v(i, j, k - 1, nx, ny)];
                        double v_zm = has_uu ? v[idx_v(i, j, k - 1, nx, ny)] : vc;
                        v_fzp = van_leer_face(v_zm, vc, v_zp, has_uu);
                    } else {
                        int has_uu = (k + 2 < nz) && kz[idx_w(i, j - 1, k + 2, nx, ny)]
                                                  && kz[idx_w(i, j, k + 2, nx, ny)]
                                                  && ky[idx_v(i, j, k + 2, nx, ny)];
                        double v_zpp = has_uu ? v[idx_v(i, j, k + 2, nx, ny)] : v_zp;
                        v_fzp = van_leer_face(v_zpp, v_zp, vc, has_uu);
                    }
                    fz_p = w_ep * v_fzp;
                    double nu_e = CFD_NU0 + 0.25 * (nu_sgs[cL] + nu_sgs[cR]
                                + nu_sgs[idx_c(i, j - 1, k + 1, nx, ny)]
                                + nu_sgs[idx_c(i, j,     k + 1, nx, ny)]);
                    double dw_dy = (w[idx_w(i, j, k + 1, nx, ny)] - w[idx_w(i, j - 1, k + 1, nx, ny)]) * inv_h;
                    tau_yz_p = nu_e * ((v_zp - vc) * inv_h + dw_dy);
                } else {
                    if (kzL_p || kzR_p) {
                        fz_p = (w_ep >= 0.0) ? (w_ep * vc) : 0.0;
                    }
                    tau_yz_p = -wall_shear_stress(vc, 0.5 * (nuL + nuR), h);
                }
            } else {
                if (kzL_p || kzR_p) {
                    int is_inflow = (w_ep < 0.0);
                    fz_p = w_ep * (is_inflow ? uy_inf : vc);
                    double nu_e = 0.5 * (nuL + nuR);
                    double dw_dy = (w[idx_w(i, j, nz, nx, ny)] - w[idx_w(i, j - 1, nz, nx, ny)]) * inv_h;
                    double dv_dz = is_inflow ? (2.0 * (uy_inf - vc) * inv_h) : 0.0;
                    tau_yz_p = nu_e * (dv_dz + dw_dy);
                } else {
                    tau_yz_p = -wall_shear_stress(vc, 0.5 * (nuL + nuR), h);
                }
            }

            int kzL_m = kz[idx_w(i, j - 1, k, nx, ny)];
            int kzR_m = kz[idx_w(i, j,     k, nx, ny)];
            double w_em = 0.5 * (w[idx_w(i, j - 1, k, nx, ny)] + w[idx_w(i, j, k, nx, ny)]);
            if (k > 0) {
                if (kzL_m && kzR_m && ky[idx_v(i, j, k - 1, nx, ny)]) {
                    double v_zm = v[idx_v(i, j, k - 1, nx, ny)];
                    double v_fzm;
                    if (w_em >= 0.0) {
                        int has_uu = (k >= 2) && kz[idx_w(i, j - 1, k - 1, nx, ny)]
                                              && kz[idx_w(i, j, k - 1, nx, ny)]
                                              && ky[idx_v(i, j, k - 2, nx, ny)];
                        double v_zmm = has_uu ? v[idx_v(i, j, k - 2, nx, ny)] : v_zm;
                        v_fzm = van_leer_face(v_zmm, v_zm, vc, has_uu);
                    } else {
                        int has_uu = (k + 1 < nz) && kz[idx_w(i, j - 1, k + 1, nx, ny)]
                                                  && kz[idx_w(i, j, k + 1, nx, ny)]
                                                  && ky[idx_v(i, j, k + 1, nx, ny)];
                        double v_zp = has_uu ? v[idx_v(i, j, k + 1, nx, ny)] : vc;
                        v_fzm = van_leer_face(v_zp, vc, v_zm, has_uu);
                    }
                    fz_m = w_em * v_fzm;
                    double nu_e = CFD_NU0 + 0.25 * (nu_sgs[cL] + nu_sgs[cR]
                                + nu_sgs[idx_c(i, j - 1, k - 1, nx, ny)]
                                + nu_sgs[idx_c(i, j,     k - 1, nx, ny)]);
                    double dw_dy = (w[idx_w(i, j, k, nx, ny)] - w[idx_w(i, j - 1, k, nx, ny)]) * inv_h;
                    tau_yz_m = nu_e * ((vc - v_zm) * inv_h + dw_dy);
                } else {
                    if (kzL_m || kzR_m) {
                        fz_m = (w_em < 0.0) ? (w_em * vc) : 0.0;
                    }
                    tau_yz_m = wall_shear_stress(vc, 0.5 * (nuL + nuR), h);
                }
            } else {
                tau_yz_m = wall_shear_stress(vc, 0.5 * (nuL + nuR), h);
            }

            double tau_yy_p = 2.0 * nuR * (vR - vc) * inv_h;
            double tau_yy_m = 2.0 * nuL * (vc - vL) * inv_h;

            double rhs = (-(fx_p - fx_m) - (fy_p - fy_m) - (fz_p - fz_m)
                          + (tau_yx_p - tau_yx_m) + (tau_yy_p - tau_yy_m) + (tau_yz_p - tau_yz_m)) * inv_h;

            if (jet_on && k >= sim->emit_k0 && k <= sim->emit_k1 &&
                j >= sim->emit_j0 && j <= sim->emit_j1 + 1 &&
                i >= sim->emit_i0 && i <= sim->emit_i1) {
                double dx = (i + 0.5) * h - sim->emit_wx;
                double dy = yf - sim->emit_wy;
                double dz = zc - sim->emit_wz;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < rcut2) {
                    rhs += jet_inv_tau * exp(-r2 * inv_2sig2) * (jet_uy - vc);
                }
            }

            rv[fv] = rhs;
        }
    }
}

static void mom_w_slab(const MomCtx *ctx, int k) {
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double h = sim->h;
    const double inv_h = sim->inv_h;
    const double inv_yw = ctx->inv_yw;
    const double yw_over_nu0 = ctx->yw_over_nu0;
    const double inv_ww_denom = ctx->inv_ww_denom;
    const double *restrict u = ctx->u;
    const double *restrict v = ctx->v;
    const double *restrict w = ctx->w;
    const double *restrict theta = ctx->theta;
    const double *restrict smoke = ctx->smoke;
    const double *restrict nu_sgs = ctx->nu_sgs;
    const uint8_t *restrict kx = sim->kx;
    const uint8_t *restrict ky = sim->ky;
    const uint8_t *restrict kz = sim->kz;
    double *restrict rw = ctx->rw;

    if (k == 0) {
        memset(&rw[idx_w(0, 0, 0, nx, ny)], 0, (size_t)nx * (size_t)ny * sizeof(double));
        return;
    }

    const double beta = sim->beta;
    const double alpha_s = sim->params.smoke_alpha_s;
    const int jet_on = sim->params.smoke_enabled && (sim->params.smoke_jet_speed > 1e-6);
    const double jet_spd = sim->params.smoke_jet_speed;
    const double jet_uz = jet_spd * sim->emit_jz;
    const double inv_2sig2 = 0.5 / (sim->emit_sigma * sim->emit_sigma);
    const double rcut2 = (3.5 * sim->emit_sigma) * (3.5 * sim->emit_sigma);
    const double jet_inv_tau = 1.0 / fmax(0.1, 2.0 * sim->emit_sigma / fmax(jet_spd, 0.5));
    const double zf = k * h;

    /* Domain +Z open sky boundary */
    if (k == nz) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t fw = idx_w(i, j, nz, nx, ny);
                if (!kz[fw]) {
                    rw[fw] = 0.0;
                    continue;
                }
                double wc = w[fw];
                double adv = 0.0;
                if (wc > 0.0) {
                    adv -= wc * (wc - w[idx_w(i, j, nz - 1, nx, ny)]) * inv_h;
                }
                double uc_b = 0.5 * (u[idx_u(i, j, nz - 1, nx, ny)] + u[idx_u(i + 1, j, nz - 1, nx, ny)]);
                double vc_b = 0.5 * (v[idx_v(i, j, nz - 1, nx, ny)] + v[idx_v(i, j + 1, nz - 1, nx, ny)]);
                if (uc_b > 0.0) {
                    double wm_x = (i > 0 && kz[idx_w(i - 1, j, nz, nx, ny)]) ? w[idx_w(i - 1, j, nz, nx, ny)] : 0.0;
                    adv -= uc_b * (wc - wm_x) * inv_h;
                } else if (uc_b < 0.0) {
                    double wp_x = (i < nx - 1 && kz[idx_w(i + 1, j, nz, nx, ny)]) ? w[idx_w(i + 1, j, nz, nx, ny)] : 0.0;
                    adv -= uc_b * (wp_x - wc) * inv_h;
                }
                if (vc_b > 0.0) {
                    double wm_y = (j > 0 && kz[idx_w(i, j - 1, nz, nx, ny)]) ? w[idx_w(i, j - 1, nz, nx, ny)] : 0.0;
                    adv -= vc_b * (wc - wm_y) * inv_h;
                } else if (vc_b < 0.0) {
                    double wp_y = (j < ny - 1 && kz[idx_w(i, j + 1, nz, nx, ny)]) ? w[idx_w(i, j + 1, nz, nx, ny)] : 0.0;
                    adv -= vc_b * (wp_y - wc) * inv_h;
                }
                /* Buoyancy on outflowing sky face */
                double buoy = 0.0;
                if (wc >= 0.0) {
                    size_t cB = idx_c(i, j, nz - 1, nx, ny);
                    buoy = CFD_G * (beta * theta[cB] - alpha_s * smoke[cB]);
                }
                rw[fw] = adv + buoy;
            }
        }
        return;
    }

    /* Interior w-faces: 1 <= k <= nz-1 */
    for (int j = 0; j < ny; ++j) {
        double yc = (j + 0.5) * h;
        for (int i = 0; i < nx; ++i) {
            size_t fw = idx_w(i, j, k, nx, ny);
            if (!kz[fw]) {
                rw[fw] = 0.0;
                continue;
            }
            double wc = w[fw];
            size_t cB = idx_c(i, j, k - 1, nx, ny);
            size_t cT = idx_c(i, j, k,     nx, ny);
            double nuB = CFD_NU0 + nu_sgs[cB];
            double nuT = CFD_NU0 + nu_sgs[cT];

            /* Z-advection fluxes */
            double wT = w[idx_w(i, j, k + 1, nx, ny)];
            double wB = w[idx_w(i, j, k - 1, nx, ny)];

            double vel_zp = 0.5 * (wc + wT);
            double w_fzp;
            if (vel_zp >= 0.0) {
                int has_uu = (k >= 1) && (kz[idx_w(i, j, k - 1, nx, ny)] != 0);
                w_fzp = van_leer_face(wB, wc, wT, has_uu);
            } else {
                int has_uu = (k + 2 <= nz) && (kz[idx_w(i, j, k + 1, nx, ny)] != 0)
                                           && (kz[idx_w(i, j, k + 2, nx, ny)] != 0);
                double wTT = has_uu ? w[idx_w(i, j, k + 2, nx, ny)] : wT;
                w_fzp = van_leer_face(wTT, wT, wc, has_uu);
            }
            double fz_p = vel_zp * w_fzp;

            double vel_zm = 0.5 * (wB + wc);
            double w_fzm;
            if (vel_zm >= 0.0) {
                int has_uu = (k >= 2) && (kz[idx_w(i, j, k - 1, nx, ny)] != 0)
                                      && (kz[idx_w(i, j, k - 2, nx, ny)] != 0);
                double wBB = has_uu ? w[idx_w(i, j, k - 2, nx, ny)] : wB;
                w_fzm = van_leer_face(wBB, wB, wc, has_uu);
            } else {
                int has_uu = (k + 1 <= nz) && (kz[idx_w(i, j, k + 1, nx, ny)] != 0);
                w_fzm = van_leer_face(wT, wc, wB, has_uu);
            }
            double fz_m = vel_zm * w_fzm;

            /* X-advection and viscous shear at i+1/2 and i-1/2 */
            double fx_p = 0.0, fx_m = 0.0;
            double tau_zx_p = 0.0, tau_zx_m = 0.0;

            int kxB_p = kx[idx_u(i + 1, j, k - 1, nx, ny)];
            int kxT_p = kx[idx_u(i + 1, j, k,     nx, ny)];
            double u_ep = 0.5 * (u[idx_u(i + 1, j, k - 1, nx, ny)] + u[idx_u(i + 1, j, k, nx, ny)]);
            if (i < nx - 1) {
                if (kxB_p && kxT_p && kz[idx_w(i + 1, j, k, nx, ny)]) {
                    double w_xp = w[idx_w(i + 1, j, k, nx, ny)];
                    double w_fxp;
                    if (u_ep >= 0.0) {
                        int has_uu = (i > 0) && kx[idx_u(i, j, k - 1, nx, ny)]
                                             && kx[idx_u(i, j, k, nx, ny)]
                                             && kz[idx_w(i - 1, j, k, nx, ny)];
                        double w_xm = has_uu ? w[idx_w(i - 1, j, k, nx, ny)] : wc;
                        w_fxp = van_leer_face(w_xm, wc, w_xp, has_uu);
                    } else {
                        int has_uu = (i + 2 < nx) && kx[idx_u(i + 2, j, k - 1, nx, ny)]
                                                  && kx[idx_u(i + 2, j, k, nx, ny)]
                                                  && kz[idx_w(i + 2, j, k, nx, ny)];
                        double w_xpp = has_uu ? w[idx_w(i + 2, j, k, nx, ny)] : w_xp;
                        w_fxp = van_leer_face(w_xpp, w_xp, wc, has_uu);
                    }
                    fx_p = u_ep * w_fxp;
                    double nu_e = CFD_NU0 + 0.25 * (nu_sgs[cB] + nu_sgs[cT]
                                + nu_sgs[idx_c(i + 1, j, k - 1, nx, ny)]
                                + nu_sgs[idx_c(i + 1, j, k,     nx, ny)]);
                    double du_dz = (u[idx_u(i + 1, j, k, nx, ny)] - u[idx_u(i + 1, j, k - 1, nx, ny)]) * inv_h;
                    tau_zx_p = nu_e * ((w_xp - wc) * inv_h + du_dz);
                } else {
                    if (kxB_p || kxT_p) {
                        fx_p = (u_ep >= 0.0) ? (u_ep * wc) : 0.0;
                    }
                    tau_zx_p = -wall_shear_stress(wc, 0.5 * (nuB + nuT), h);
                }
            } else {
                if (kxB_p || kxT_p) {
                    int is_inflow = (sim->bc_px == BC_INFLOW) || (u_ep < 0.0);
                    fx_p = is_inflow ? 0.0 : (u_ep * wc);
                    double nu_e = 0.5 * (nuB + nuT);
                    double du_dz = (u[idx_u(nx, j, k, nx, ny)] - u[idx_u(nx, j, k - 1, nx, ny)]) * inv_h;
                    double dw_dx = is_inflow ? (-2.0 * wc * inv_h) : 0.0;
                    tau_zx_p = nu_e * (dw_dx + du_dz);
                } else {
                    tau_zx_p = -wall_shear_stress(wc, 0.5 * (nuB + nuT), h);
                }
            }

            int kxB_m = kx[idx_u(i, j, k - 1, nx, ny)];
            int kxT_m = kx[idx_u(i, j, k,     nx, ny)];
            double u_em = 0.5 * (u[idx_u(i, j, k - 1, nx, ny)] + u[idx_u(i, j, k, nx, ny)]);
            if (i > 0) {
                if (kxB_m && kxT_m && kz[idx_w(i - 1, j, k, nx, ny)]) {
                    double w_xm = w[idx_w(i - 1, j, k, nx, ny)];
                    double w_fxm;
                    if (u_em >= 0.0) {
                        int has_uu = (i >= 2) && kx[idx_u(i - 1, j, k - 1, nx, ny)]
                                              && kx[idx_u(i - 1, j, k, nx, ny)]
                                              && kz[idx_w(i - 2, j, k, nx, ny)];
                        double w_xmm = has_uu ? w[idx_w(i - 2, j, k, nx, ny)] : w_xm;
                        w_fxm = van_leer_face(w_xmm, w_xm, wc, has_uu);
                    } else {
                        int has_uu = (i + 1 < nx) && kx[idx_u(i + 1, j, k - 1, nx, ny)]
                                                  && kx[idx_u(i + 1, j, k, nx, ny)]
                                                  && kz[idx_w(i + 1, j, k, nx, ny)];
                        double w_xp = has_uu ? w[idx_w(i + 1, j, k, nx, ny)] : wc;
                        w_fxm = van_leer_face(w_xp, wc, w_xm, has_uu);
                    }
                    fx_m = u_em * w_fxm;
                    double nu_e = CFD_NU0 + 0.25 * (nu_sgs[cB] + nu_sgs[cT]
                                + nu_sgs[idx_c(i - 1, j, k - 1, nx, ny)]
                                + nu_sgs[idx_c(i - 1, j, k,     nx, ny)]);
                    double du_dz = (u[idx_u(i, j, k, nx, ny)] - u[idx_u(i, j, k - 1, nx, ny)]) * inv_h;
                    tau_zx_m = nu_e * ((wc - w_xm) * inv_h + du_dz);
                } else {
                    if (kxB_m || kxT_m) {
                        fx_m = (u_em < 0.0) ? (u_em * wc) : 0.0;
                    }
                    tau_zx_m = wall_shear_stress(wc, 0.5 * (nuB + nuT), h);
                }
            } else {
                if (kxB_m || kxT_m) {
                    fx_m = (u_em < 0.0) ? (u_em * wc) : 0.0;
                    double nu_e = 0.5 * (nuB + nuT);
                    double du_dz = (u[idx_u(0, j, k, nx, ny)] - u[idx_u(0, j, k - 1, nx, ny)]) * inv_h;
                    double dw_dx = (u_em > 0.0) ? (2.0 * wc * inv_h) : 0.0;
                    tau_zx_m = nu_e * (dw_dx + du_dz);
                } else {
                    tau_zx_m = wall_shear_stress(wc, 0.5 * (nuB + nuT), h);
                }
            }

            /* Y-advection and viscous shear at j+1/2 and j-1/2 */
            double fy_p = 0.0, fy_m = 0.0;
            double tau_zy_p = 0.0, tau_zy_m = 0.0;

            int kyB_p = ky[idx_v(i, j + 1, k - 1, nx, ny)];
            int kyT_p = ky[idx_v(i, j + 1, k,     nx, ny)];
            double v_ep = 0.5 * (v[idx_v(i, j + 1, k - 1, nx, ny)] + v[idx_v(i, j + 1, k, nx, ny)]);
            if (j < ny - 1) {
                if (kyB_p && kyT_p && kz[idx_w(i, j + 1, k, nx, ny)]) {
                    double w_yp = w[idx_w(i, j + 1, k, nx, ny)];
                    double w_fyp;
                    if (v_ep >= 0.0) {
                        int has_uu = (j > 0) && ky[idx_v(i, j, k - 1, nx, ny)]
                                             && ky[idx_v(i, j, k, nx, ny)]
                                             && kz[idx_w(i, j - 1, k, nx, ny)];
                        double w_ym = has_uu ? w[idx_w(i, j - 1, k, nx, ny)] : wc;
                        w_fyp = van_leer_face(w_ym, wc, w_yp, has_uu);
                    } else {
                        int has_uu = (j + 2 < ny) && ky[idx_v(i, j + 2, k - 1, nx, ny)]
                                                  && ky[idx_v(i, j + 2, k, nx, ny)]
                                                  && kz[idx_w(i, j + 2, k, nx, ny)];
                        double w_ypp = has_uu ? w[idx_w(i, j + 2, k, nx, ny)] : w_yp;
                        w_fyp = van_leer_face(w_ypp, w_yp, wc, has_uu);
                    }
                    fy_p = v_ep * w_fyp;
                    double nu_e = CFD_NU0 + 0.25 * (nu_sgs[cB] + nu_sgs[cT]
                                + nu_sgs[idx_c(i, j + 1, k - 1, nx, ny)]
                                + nu_sgs[idx_c(i, j + 1, k,     nx, ny)]);
                    double dv_dz = (v[idx_v(i, j + 1, k, nx, ny)] - v[idx_v(i, j + 1, k - 1, nx, ny)]) * inv_h;
                    tau_zy_p = nu_e * ((w_yp - wc) * inv_h + dv_dz);
                } else {
                    if (kyB_p || kyT_p) {
                        fy_p = (v_ep >= 0.0) ? (v_ep * wc) : 0.0;
                    }
                    tau_zy_p = -wall_shear_stress(wc, 0.5 * (nuB + nuT), h);
                }
            } else {
                if (kyB_p || kyT_p) {
                    int is_inflow = (sim->bc_py == BC_INFLOW) || (v_ep < 0.0);
                    fy_p = is_inflow ? 0.0 : (v_ep * wc);
                    double nu_e = 0.5 * (nuB + nuT);
                    double dv_dz = (v[idx_v(i, ny, k, nx, ny)] - v[idx_v(i, ny, k - 1, nx, ny)]) * inv_h;
                    double dw_dy = is_inflow ? (-2.0 * wc * inv_h) : 0.0;
                    tau_zy_p = nu_e * (dw_dy + dv_dz);
                } else {
                    tau_zy_p = -wall_shear_stress(wc, 0.5 * (nuB + nuT), h);
                }
            }

            int kyB_m = ky[idx_v(i, j, k - 1, nx, ny)];
            int kyT_m = ky[idx_v(i, j, k,     nx, ny)];
            double v_em = 0.5 * (v[idx_v(i, j, k - 1, nx, ny)] + v[idx_v(i, j, k, nx, ny)]);
            if (j > 0) {
                if (kyB_m && kyT_m && kz[idx_w(i, j - 1, k, nx, ny)]) {
                    double w_ym = w[idx_w(i, j - 1, k, nx, ny)];
                    double w_fym;
                    if (v_em >= 0.0) {
                        int has_uu = (j >= 2) && ky[idx_v(i, j - 1, k - 1, nx, ny)]
                                              && ky[idx_v(i, j - 1, k, nx, ny)]
                                              && kz[idx_w(i, j - 2, k, nx, ny)];
                        double w_ymm = has_uu ? w[idx_w(i, j - 2, k, nx, ny)] : w_ym;
                        w_fym = van_leer_face(w_ymm, w_ym, wc, has_uu);
                    } else {
                        int has_uu = (j + 1 < ny) && ky[idx_v(i, j + 1, k - 1, nx, ny)]
                                                  && ky[idx_v(i, j + 1, k, nx, ny)]
                                                  && kz[idx_w(i, j + 1, k, nx, ny)];
                        double w_yp = has_uu ? w[idx_w(i, j + 1, k, nx, ny)] : wc;
                        w_fym = van_leer_face(w_yp, wc, w_ym, has_uu);
                    }
                    fy_m = v_em * w_fym;
                    double nu_e = CFD_NU0 + 0.25 * (nu_sgs[cB] + nu_sgs[cT]
                                + nu_sgs[idx_c(i, j - 1, k - 1, nx, ny)]
                                + nu_sgs[idx_c(i, j - 1, k,     nx, ny)]);
                    double dv_dz = (v[idx_v(i, j, k, nx, ny)] - v[idx_v(i, j, k - 1, nx, ny)]) * inv_h;
                    tau_zy_m = nu_e * ((wc - w_ym) * inv_h + dv_dz);
                } else {
                    if (kyB_m || kyT_m) {
                        fy_m = (v_em < 0.0) ? (v_em * wc) : 0.0;
                    }
                    tau_zy_m = wall_shear_stress(wc, 0.5 * (nuB + nuT), h);
                }
            } else {
                if (kyB_m || kyT_m) {
                    int is_inflow = (sim->bc_my == BC_INFLOW) || (v_em > 0.0);
                    fy_m = is_inflow ? 0.0 : (v_em * wc);
                    double nu_e = 0.5 * (nuB + nuT);
                    double dv_dz = (v[idx_v(i, 0, k, nx, ny)] - v[idx_v(i, 0, k - 1, nx, ny)]) * inv_h;
                    double dw_dy = is_inflow ? (2.0 * wc * inv_h) : 0.0;
                    tau_zy_m = nu_e * (dw_dy + dv_dz);
                } else {
                    tau_zy_m = wall_shear_stress(wc, 0.5 * (nuB + nuT), h);
                }
            }

            double tau_zz_p = 2.0 * nuT * (wT - wc) * inv_h;
            double tau_zz_m = 2.0 * nuB * (wc - wB) * inv_h;

            /* Exact hydrostatic-perturbation Boussinesq buoyancy */
            double theta_f = 0.5 * (theta[cB] + theta[cT]);
            double smoke_f = 0.5 * (smoke[cB] + smoke[cT]);
            double buoy = CFD_G * (beta * theta_f - alpha_s * smoke_f);

            double rhs = (-(fx_p - fx_m) - (fy_p - fy_m) - (fz_p - fz_m)
                          + (tau_zx_p - tau_zx_m) + (tau_zy_p - tau_zy_m) + (tau_zz_p - tau_zz_m)) * inv_h
                          + buoy;

            if (jet_on && k >= sim->emit_k0 && k <= sim->emit_k1 + 1 &&
                j >= sim->emit_j0 && j <= sim->emit_j1 &&
                i >= sim->emit_i0 && i <= sim->emit_i1) {
                double dx = (i + 0.5) * h - sim->emit_wx;
                double dy = yc - sim->emit_wy;
                double dz = zf - sim->emit_wz;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < rcut2) {
                    rhs += jet_inv_tau * exp(-r2 * inv_2sig2) * (jet_uz - wc);
                }
            }

            rw[fw] = rhs;
        }
    }
}

#undef wall_shear_stress

static void mom_uvw_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    const MomCtx *ctx = (const MomCtx *)vctx;
    const int nz = ctx->sim->nz;
    for (int k = k0; k < k1; ++k) {
        mom_uv_slab(ctx, k);
        mom_w_slab(ctx, k);
    }
    if (k1 == nz) {
        mom_w_slab(ctx, nz);
    }
}

void cfd_compute_momentum_rhs(CFD3D *sim, const double *u, const double *v, const double *w,
                              const double *theta, const double *smoke, const double *nu_sgs,
                              double *ru, double *rv, double *rw) {
    const double yw = 0.5 * sim->h;
    const double yw_over_nu0 = yw / CFD_NU0;
    const double inv_ww_denom = 1.0 / (8.3 * pow(yw_over_nu0, 1.0 / 7.0));
    MomCtx ctx = {
        sim, u, v, w, theta, smoke, nu_sgs, ru, rv, rw,
        2.0 * sim->inv_h, yw_over_nu0, inv_ww_denom
    };
    cfd_parallel_for_k(sim, sim->nz, 8, &ctx, mom_uvw_chunk_fn);
}

/* ========================================================================= */
/* 3. Scalar RHS: Flux-Conservative MC-TVD Advection + Variable Diffusivity  */
/* ========================================================================= */

typedef struct ScalCtx {
    const CFD3D  *sim;
    const double *u, *v, *w, *theta, *smoke, *nu_sgs;
    double       *out_theta, *out_smoke;
    double        dt;
    int           fused_stage1;
} ScalCtx;

static void scalar_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    const ScalCtx *ctx = (const ScalCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double inv_h = sim->inv_h;
    const double inv_prt = 1.0 / CFD_PR_T;
    const double inv_sct = 1.0 / CFD_SC_T;
    const double half_inv_prt = 0.5 * inv_prt;
    const double half_inv_sct = 0.5 * inv_sct;
    const double *restrict u = ctx->u;
    const double *restrict v = ctx->v;
    const double *restrict w = ctx->w;
    const double *restrict th = ctx->theta;
    const double *restrict sm = ctx->smoke;
    const double *restrict nu_sgs = ctx->nu_sgs;
    const uint8_t *restrict kx = sim->kx;
    const uint8_t *restrict ky = sim->ky;
    const uint8_t *restrict kz = sim->kz;
    const uint8_t *restrict ct = sim->cell_type;
    const double *restrict emit_k = sim->emit_kernel;
    double *restrict out_th = ctx->out_theta;
    double *restrict out_sm = ctx->out_smoke;
    const double dt = ctx->dt;
    const int fused_s1 = ctx->fused_stage1;
    const double theta_in = sim->theta_in;
    const int emit_on = sim->params.smoke_enabled && (sim->params.smoke_rate > 0.0);
    const double smoke_rate = sim->params.smoke_rate;
    const double theta_smoke = (sim->params.smoke_temp_c + 273.15) - sim->t_out_k;
    const size_t sy = (size_t)nx;
    const size_t sz = (size_t)nx * (size_t)ny;

    /* Carry +Y face fluxes from row j to row j+1 (eliminates all -Y recomputation!) */
    double prev_y_th_stack[512];
    double prev_y_sm_stack[512];
    double *prev_y_th = (nx <= 512) ? prev_y_th_stack : (double *)malloc((size_t)nx * sizeof(double));
    double *prev_y_sm = (nx <= 512) ? prev_y_sm_stack : (double *)malloc((size_t)nx * sizeof(double));

    for (int k = k0; k < k1; ++k) {
        const int has_km1 = (k > 0);
        const int has_km2 = (k >= 2);
        const int has_kp1 = (k + 1 < nz);
        const int has_kp2 = (k + 2 < nz);

        /* Precompute j = 0 (-Y domain boundary) fluxes into prev_y_th / prev_y_sm */
        {
            const size_t row_c0 = idx_c(0, 0, k, nx, ny);
            const size_t row_v0 = idx_v(0, 0, k, nx, ny);
            const uint8_t *restrict ctr0  = ct + row_c0;
            const double  *restrict thr0  = th + row_c0;
            const double  *restrict smr0  = sm + row_c0;
            const double  *restrict nur0  = nu_sgs + row_c0;
            const double  *restrict v0r   = v + row_v0;
            const uint8_t *restrict ky0r  = ky + row_v0;
            for (int i = 0; i < nx; ++i) {
                if (ctr0[i] == CFD_CELL_FLUID && ky0r[i]) {
                    double vf = v0r[i];
                    double th_c = thr0[i], sm_c = smr0[i];
                    int is_inflow = (sim->bc_my == BC_INFLOW) || (vf > 0.0);
                    double fth = vf * (is_inflow ? 0.0 : th_c);
                    double fsm = vf * (is_inflow ? 0.0 : sm_c);
                    if (is_inflow) {
                        double nu_c = nur0[i];
                        fth -= (CFD_ALPHA0 + nu_c * inv_prt) * th_c * (2.0 * inv_h);
                        fsm -= (CFD_DIFF0  + nu_c * inv_sct) * sm_c * (2.0 * inv_h);
                    }
                    prev_y_th[i] = fth;
                    prev_y_sm[i] = fsm;
                } else {
                    prev_y_th[i] = 0.0;
                    prev_y_sm[i] = 0.0;
                }
            }
        }

        for (int j = 0; j < ny; ++j) {
            const int has_jm1 = (j > 0);
            const int has_jp1 = (j + 1 < ny);
            const int has_jp2 = (j + 2 < ny);

            const size_t row_c  = idx_c(0, j, k, nx, ny);
            const size_t row_u  = idx_u(0, j, k, nx, ny);
            const size_t row_v0 = idx_v(0, j,     k, nx, ny);
            const size_t row_v1 = idx_v(0, j + 1, k, nx, ny);
            const size_t row_w0 = idx_w(0, j, k,     nx, ny);
            const size_t row_w1 = idx_w(0, j, k + 1, nx, ny);

            const uint8_t *restrict ctr  = ct + row_c;
            const double  *restrict thr  = th + row_c;
            const double  *restrict smr  = sm + row_c;
            const double  *restrict nur  = nu_sgs + row_c;
            const double  *restrict emr  = emit_k + row_c;
            const double  *restrict ur   = u + row_u;
            const uint8_t *restrict kxr  = kx + row_u;
            const double  *restrict v1r  = v + row_v1;
            const uint8_t *restrict ky0r = ky + row_v0;
            const uint8_t *restrict ky1r = ky + row_v1;
            const uint8_t *restrict kyp2 = has_jp2 ? (ky + idx_v(0, j + 2, k, nx, ny)) : NULL;
            const double  *restrict w0r  = w + row_w0;
            const double  *restrict w1r  = w + row_w1;
            const uint8_t *restrict kz0r = kz + row_w0;
            const uint8_t *restrict kz1r = kz + row_w1;
            const uint8_t *restrict kzm1 = has_km1 ? (kz + idx_w(0, j, k - 1, nx, ny)) : NULL;
            const uint8_t *restrict kzp2 = has_kp2 ? (kz + idx_w(0, j, k + 2, nx, ny)) : NULL;

            double *restrict othr = out_th + row_c;
            double *restrict osmr = out_sm + row_c;

            /* Precompute i = 0 (-X domain boundary: BC_VENT) flux into prev_x_th / prev_x_sm */
            double prev_x_th = 0.0, prev_x_sm = 0.0;
            if (ctr[0] == CFD_CELL_FLUID && kxr[0]) {
                double uf = ur[0];
                double th_c = thr[0], sm_c = smr[0];
                int is_inflow = (uf > 0.0);
                prev_x_th = uf * (is_inflow ? theta_in : th_c);
                prev_x_sm = uf * (is_inflow ? 0.0      : sm_c);
                if (is_inflow) {
                    double nu_c = nur[0];
                    prev_x_th -= (CFD_ALPHA0 + nu_c * inv_prt) * (th_c - theta_in) * (2.0 * inv_h);
                    prev_x_sm -= (CFD_DIFF0  + nu_c * inv_sct) * sm_c              * (2.0 * inv_h);
                }
            }

            for (int i = 0; i < nx; ++i) {
                if (ctr[i] != CFD_CELL_FLUID) {
                    prev_x_th = 0.0;
                    prev_x_sm = 0.0;
                    prev_y_th[i] = 0.0;
                    prev_y_sm[i] = 0.0;
                    if (!fused_s1) {
                        othr[i] = 0.0;
                        osmr[i] = 0.0;
                    }
                    continue;
                }

                const double th_c = thr[i];
                const double sm_c = smr[i];
                const double nu_c = nur[i];

                /* Start with reused -X and -Y total fluxes (adv - dif entering from -X/-Y) */
                double net_th = prev_x_th + prev_y_th[i];
                double net_sm = prev_x_sm + prev_y_sm[i];

                /* ----- +X face (i+1): compute ONCE, subtract from cell i, carry to cell i+1 ----- */
                double fx_th = 0.0, fx_sm = 0.0;
                if (kxr[i + 1]) {
                    double uf = ur[i + 1];
                    if (i < nx - 1) {
                        double th_R = thr[i + 1], sm_R = smr[i + 1];
                        int th_diff = (th_R != th_c);
                        int sm_diff = (sm_R != sm_c);
                        if (th_diff || sm_diff) {
                            double nu_sum = nu_c + nur[i + 1];
                            if (uf >= 0.0) {
                                int has_uu = (i > 0) && (kxr[i] != 0);
                                if (th_diff) {
                                    double th_f = mc_face(has_uu ? thr[i - 1] : th_c, th_c, th_R, has_uu);
                                    double al_f = CFD_ALPHA0 + nu_sum * half_inv_prt;
                                    fx_th = uf * th_f - al_f * (th_R - th_c) * inv_h;
                                } else {
                                    fx_th = uf * th_c;
                                }
                                if (sm_diff) {
                                    double sm_f = mc_face(has_uu ? smr[i - 1] : sm_c, sm_c, sm_R, has_uu);
                                    double dc_f = CFD_DIFF0  + nu_sum * half_inv_sct;
                                    fx_sm = uf * sm_f - dc_f * (sm_R - sm_c) * inv_h;
                                } else {
                                    fx_sm = uf * sm_c;
                                }
                            } else {
                                int has_uu = (i + 2 < nx) && (kxr[i + 2] != 0);
                                if (th_diff) {
                                    double th_f = mc_face(has_uu ? thr[i + 2] : th_R, th_R, th_c, has_uu);
                                    double al_f = CFD_ALPHA0 + nu_sum * half_inv_prt;
                                    fx_th = uf * th_f - al_f * (th_R - th_c) * inv_h;
                                } else {
                                    fx_th = uf * th_c;
                                }
                                if (sm_diff) {
                                    double sm_f = mc_face(has_uu ? smr[i + 2] : sm_R, sm_R, sm_c, has_uu);
                                    double dc_f = CFD_DIFF0  + nu_sum * half_inv_sct;
                                    fx_sm = uf * sm_f - dc_f * (sm_R - sm_c) * inv_h;
                                } else {
                                    fx_sm = uf * sm_c;
                                }
                            }
                        } else {
                            fx_th = uf * th_c;
                            fx_sm = uf * sm_c;
                        }
                    } else {
                        int is_inflow = (sim->bc_px == BC_INFLOW) || (uf < 0.0);
                        fx_th = uf * (is_inflow ? 0.0 : th_c);
                        fx_sm = uf * (is_inflow ? 0.0 : sm_c);
                        if (is_inflow) {
                            fx_th -= (CFD_ALPHA0 + nu_c * inv_prt) * (0.0 - th_c) * (2.0 * inv_h);
                            fx_sm -= (CFD_DIFF0  + nu_c * inv_sct) * (0.0 - sm_c) * (2.0 * inv_h);
                        }
                    }
                }
                net_th -= fx_th;
                net_sm -= fx_sm;
                prev_x_th = fx_th;
                prev_x_sm = fx_sm;

                /* ----- +Y face (j+1): compute ONCE, subtract from cell j, carry to cell j+1 ----- */
                double fy_th = 0.0, fy_sm = 0.0;
                if (ky1r[i]) {
                    double vf = v1r[i];
                    if (has_jp1) {
                        size_t cR = row_c + (size_t)i + sy;
                        double th_R = th[cR], sm_R = sm[cR];
                        int th_diff = (th_R != th_c);
                        int sm_diff = (sm_R != sm_c);
                        if (th_diff || sm_diff) {
                            double nu_sum = nu_c + nu_sgs[cR];
                            if (vf >= 0.0) {
                                int has_uu = has_jm1 && (ky0r[i] != 0);
                                size_t cL = has_uu ? (row_c + (size_t)i - sy) : (row_c + (size_t)i);
                                if (th_diff) {
                                    double th_f = mc_face(th[cL], th_c, th_R, has_uu);
                                    double al_f = CFD_ALPHA0 + nu_sum * half_inv_prt;
                                    fy_th = vf * th_f - al_f * (th_R - th_c) * inv_h;
                                } else {
                                    fy_th = vf * th_c;
                                }
                                if (sm_diff) {
                                    double sm_f = mc_face(sm[cL], sm_c, sm_R, has_uu);
                                    double dc_f = CFD_DIFF0  + nu_sum * half_inv_sct;
                                    fy_sm = vf * sm_f - dc_f * (sm_R - sm_c) * inv_h;
                                } else {
                                    fy_sm = vf * sm_c;
                                }
                            } else {
                                int has_uu = has_jp2 && (kyp2[i] != 0);
                                size_t cRR = has_uu ? (cR + sy) : cR;
                                if (th_diff) {
                                    double th_f = mc_face(th[cRR], th_R, th_c, has_uu);
                                    double al_f = CFD_ALPHA0 + nu_sum * half_inv_prt;
                                    fy_th = vf * th_f - al_f * (th_R - th_c) * inv_h;
                                } else {
                                    fy_th = vf * th_c;
                                }
                                if (sm_diff) {
                                    double sm_f = mc_face(sm[cRR], sm_R, sm_c, has_uu);
                                    double dc_f = CFD_DIFF0  + nu_sum * half_inv_sct;
                                    fy_sm = vf * sm_f - dc_f * (sm_R - sm_c) * inv_h;
                                } else {
                                    fy_sm = vf * sm_c;
                                }
                            }
                        } else {
                            fy_th = vf * th_c;
                            fy_sm = vf * sm_c;
                        }
                    } else {
                        int is_inflow = (sim->bc_py == BC_INFLOW) || (vf < 0.0);
                        fy_th = vf * (is_inflow ? 0.0 : th_c);
                        fy_sm = vf * (is_inflow ? 0.0 : sm_c);
                        if (is_inflow) {
                            fy_th -= (CFD_ALPHA0 + nu_c * inv_prt) * (0.0 - th_c) * (2.0 * inv_h);
                            fy_sm -= (CFD_DIFF0  + nu_c * inv_sct) * (0.0 - sm_c) * (2.0 * inv_h);
                        }
                    }
                }
                net_th -= fy_th;
                net_sm -= fy_sm;
                prev_y_th[i] = fy_th;
                prev_y_sm[i] = fy_sm;

                /* ----- +Z face (k+1) ----- */
                if (kz1r[i]) {
                    double wf = w1r[i];
                    if (has_kp1) {
                        size_t cT = row_c + (size_t)i + sz;
                        double th_T = th[cT], sm_T = sm[cT];
                        int th_diff = (th_T != th_c);
                        int sm_diff = (sm_T != sm_c);
                        if (th_diff || sm_diff) {
                            double nu_sum = nu_c + nu_sgs[cT];
                            if (wf >= 0.0) {
                                int has_uu = has_km1 && (kz0r[i] != 0);
                                size_t cB = has_uu ? (row_c + (size_t)i - sz) : (row_c + (size_t)i);
                                if (th_diff) {
                                    double th_f = mc_face(th[cB], th_c, th_T, has_uu);
                                    double al_f = CFD_ALPHA0 + nu_sum * half_inv_prt;
                                    net_th -= wf * th_f - al_f * (th_T - th_c) * inv_h;
                                } else {
                                    net_th -= wf * th_c;
                                }
                                if (sm_diff) {
                                    double sm_f = mc_face(sm[cB], sm_c, sm_T, has_uu);
                                    double dc_f = CFD_DIFF0  + nu_sum * half_inv_sct;
                                    net_sm -= wf * sm_f - dc_f * (sm_T - sm_c) * inv_h;
                                } else {
                                    net_sm -= wf * sm_c;
                                }
                            } else {
                                int has_uu = has_kp2 && (kzp2[i] != 0);
                                size_t cTT = has_uu ? (cT + sz) : cT;
                                if (th_diff) {
                                    double th_f = mc_face(th[cTT], th_T, th_c, has_uu);
                                    double al_f = CFD_ALPHA0 + nu_sum * half_inv_prt;
                                    net_th -= wf * th_f - al_f * (th_T - th_c) * inv_h;
                                } else {
                                    net_th -= wf * th_c;
                                }
                                if (sm_diff) {
                                    double sm_f = mc_face(sm[cTT], sm_T, sm_c, has_uu);
                                    double dc_f = CFD_DIFF0  + nu_sum * half_inv_sct;
                                    net_sm -= wf * sm_f - dc_f * (sm_T - sm_c) * inv_h;
                                } else {
                                    net_sm -= wf * sm_c;
                                }
                            }
                        } else {
                            net_th -= wf * th_c;
                            net_sm -= wf * sm_c;
                        }
                    } else {
                        int is_inflow = (wf < 0.0);
                        net_th -= wf * (is_inflow ? 0.0 : th_c);
                        net_sm -= wf * (is_inflow ? 0.0 : sm_c);
                        if (is_inflow) {
                            net_th += (CFD_ALPHA0 + nu_c * inv_prt) * (0.0 - th_c) * (2.0 * inv_h);
                            net_sm += (CFD_DIFF0  + nu_c * inv_sct) * (0.0 - sm_c) * (2.0 * inv_h);
                        }
                    }
                }

                /* ----- -Z face (k) ----- */
                if (kz0r[i] && has_km1) {
                    double wf = w0r[i];
                    size_t cB = row_c + (size_t)i - sz;
                    double th_B = th[cB], sm_B = sm[cB];
                    int th_diff = (th_B != th_c);
                    int sm_diff = (sm_B != sm_c);
                    if (th_diff || sm_diff) {
                        double nu_sum = nu_c + nu_sgs[cB];
                        if (wf >= 0.0) {
                            int has_uu = has_km2 && (kzm1[i] != 0);
                            size_t cBB = has_uu ? (cB - sz) : cB;
                            if (th_diff) {
                                double th_f = mc_face(th[cBB], th_B, th_c, has_uu);
                                double al_f = CFD_ALPHA0 + nu_sum * half_inv_prt;
                                net_th += wf * th_f - al_f * (th_c - th_B) * inv_h;
                            } else {
                                net_th += wf * th_c;
                            }
                            if (sm_diff) {
                                double sm_f = mc_face(sm[cBB], sm_B, sm_c, has_uu);
                                double dc_f = CFD_DIFF0  + nu_sum * half_inv_sct;
                                net_sm += wf * sm_f - dc_f * (sm_c - sm_B) * inv_h;
                            } else {
                                net_sm += wf * sm_c;
                            }
                        } else {
                            int has_uu = has_kp1 && (kz1r[i] != 0);
                            size_t cT = has_uu ? (row_c + (size_t)i + sz) : (row_c + (size_t)i);
                            if (th_diff) {
                                double th_f = mc_face(th[cT], th_c, th_B, has_uu);
                                double al_f = CFD_ALPHA0 + nu_sum * half_inv_prt;
                                net_th += wf * th_f - al_f * (th_c - th_B) * inv_h;
                            } else {
                                net_th += wf * th_c;
                            }
                            if (sm_diff) {
                                double sm_f = mc_face(sm[cT], sm_c, sm_B, has_uu);
                                double dc_f = CFD_DIFF0  + nu_sum * half_inv_sct;
                                net_sm += wf * sm_f - dc_f * (sm_c - sm_B) * inv_h;
                            } else {
                                net_sm += wf * sm_c;
                            }
                        }
                    } else {
                        net_th += wf * th_c;
                        net_sm += wf * sm_c;
                    }
                }

                double rth = net_th * inv_h;
                double rsm = net_sm * inv_h;

                if (emit_on) {
                    double gval = emr[i];
                    if (gval > 0.0) {
                        double rate_c = smoke_rate * gval;
                        rsm += rate_c;
                        rth += rate_c * (theta_smoke - th_c);
                    }
                }

                if (fused_s1) {
                    othr[i] = th_c + dt * rth;
                    double sm1 = sm_c + dt * rsm;
                    osmr[i] = (sm1 > 0.0) ? sm1 : 0.0;
                } else {
                    othr[i] = rth;
                    osmr[i] = rsm;
                }
            }
        }
    }

    if (nx > 512) {
        free(prev_y_th);
        free(prev_y_sm);
    }
}

void cfd_compute_scalar_rhs(CFD3D *sim, const double *u, const double *v, const double *w,
                            const double *theta, const double *smoke, const double *nu_sgs,
                            double *rtheta, double *rsmoke) {
    ScalCtx ctx = { sim, u, v, w, theta, smoke, nu_sgs, rtheta, rsmoke, 0.0, 0 };
    cfd_parallel_for_k(sim, sim->nz, 8, &ctx, scalar_chunk_fn);
}

void cfd_scalar_stage1_step(CFD3D *sim, double dt, const double *u, const double *v, const double *w,
                            const double *theta0, const double *smoke0, const double *nu_sgs,
                            double *theta1, double *smoke1) {
    ScalCtx ctx = { sim, u, v, w, theta0, smoke0, nu_sgs, theta1, smoke1, dt, 1 };
    cfd_parallel_for_k(sim, sim->nz, 8, &ctx, scalar_chunk_fn);
}
