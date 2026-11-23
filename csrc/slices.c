#include "cfd3d_internal.h"

static inline double eval_cell_field(const CFD3D *sim, int field_id, int i, int j, int k) {
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    size_t c = idx_c(i, j, k, nx, ny);
    int is_solid = (sim->cell_type[c] != CFD_CELL_FLUID);

    if (field_id == CFD_FIELD_SOLID) {
        return is_solid ? 1.0 : 0.0;
    }
    if (is_solid) {
        return 0.0;
    }

    switch (field_id) {
        case CFD_FIELD_SMOKE:
            return sim->smoke[c];
        case CFD_FIELD_TEMP:
            return sim->params.temp_out_c + sim->theta[c];
        case CFD_FIELD_PRESSURE:
            return sim->rho0 * sim->phi[c];
        case CFD_FIELD_NU_SGS:
            return sim->nu_sgs[c];
        case CFD_FIELD_U:
            return 0.5 * (sim->u[idx_u(i, j, k, nx, ny)] + sim->u[idx_u(i + 1, j, k, nx, ny)]);
        case CFD_FIELD_V:
            return 0.5 * (sim->v[idx_v(i, j, k, nx, ny)] + sim->v[idx_v(i, j + 1, k, nx, ny)]);
        case CFD_FIELD_W:
            return 0.5 * (sim->w[idx_w(i, j, k, nx, ny)] + sim->w[idx_w(i, j, k + 1, nx, ny)]);
        case CFD_FIELD_SPEED: {
            double uc = 0.5 * (sim->u[idx_u(i, j, k, nx, ny)] + sim->u[idx_u(i + 1, j, k, nx, ny)]);
            double vc = 0.5 * (sim->v[idx_v(i, j, k, nx, ny)] + sim->v[idx_v(i, j + 1, k, nx, ny)]);
            double wc = 0.5 * (sim->w[idx_w(i, j, k, nx, ny)] + sim->w[idx_w(i, j, k + 1, nx, ny)]);
            return sqrt(uc * uc + vc * vc + wc * wc);
        }
        case CFD_FIELD_VORTICITY: {
            const double inv_2h = 0.5 * sim->inv_h;
            const double *u = sim->u, *v = sim->v, *w = sim->w;
            const uint8_t *kx = sim->kx, *ky = sim->ky, *kz = sim->kz;

            double uc = 0.5 * (u[idx_u(i, j, k, nx, ny)] + u[idx_u(i + 1, j, k, nx, ny)]);
            double vc = 0.5 * (v[idx_v(i, j, k, nx, ny)] + v[idx_v(i, j + 1, k, nx, ny)]);
            double wc = 0.5 * (w[idx_w(i, j, k, nx, ny)] + w[idx_w(i, j, k + 1, nx, ny)]);

            uint8_t kxm = kx[idx_u(i, j, k, nx, ny)];
            uint8_t kxp = kx[idx_u(i + 1, j, k, nx, ny)];
            uint8_t kym = ky[idx_v(i, j, k, nx, ny)];
            uint8_t kyp = ky[idx_v(i, j + 1, k, nx, ny)];
            uint8_t kzm = kz[idx_w(i, j, k, nx, ny)];
            uint8_t kzp = kz[idx_w(i, j, k + 1, nx, ny)];

            double vm = (i > 0 && kxm)
                      ? 0.5 * (v[idx_v(i - 1, j, k, nx, ny)] + v[idx_v(i - 1, j + 1, k, nx, ny)])
                      : (kxm ? vc : -vc);
            double vp = (i + 1 < nx && kxp)
                      ? 0.5 * (v[idx_v(i + 1, j, k, nx, ny)] + v[idx_v(i + 1, j + 1, k, nx, ny)])
                      : (kxp ? vc : -vc);
            double wm = (i > 0 && kxm)
                      ? 0.5 * (w[idx_w(i - 1, j, k, nx, ny)] + w[idx_w(i - 1, j, k + 1, nx, ny)])
                      : (kxm ? wc : -wc);
            double wp = (i + 1 < nx && kxp)
                      ? 0.5 * (w[idx_w(i + 1, j, k, nx, ny)] + w[idx_w(i + 1, j, k + 1, nx, ny)])
                      : (kxp ? wc : -wc);

            double um = (j > 0 && kym)
                      ? 0.5 * (u[idx_u(i, j - 1, k, nx, ny)] + u[idx_u(i + 1, j - 1, k, nx, ny)])
                      : (kym ? uc : -uc);
            double up = (j + 1 < ny && kyp)
                      ? 0.5 * (u[idx_u(i, j + 1, k, nx, ny)] + u[idx_u(i + 1, j + 1, k, nx, ny)])
                      : (kyp ? uc : -uc);
            double wym = (j > 0 && kym)
                      ? 0.5 * (w[idx_w(i, j - 1, k, nx, ny)] + w[idx_w(i, j - 1, k + 1, nx, ny)])
                      : (kym ? wc : -wc);
            double wyp = (j + 1 < ny && kyp)
                      ? 0.5 * (w[idx_w(i, j + 1, k, nx, ny)] + w[idx_w(i, j + 1, k + 1, nx, ny)])
                      : (kyp ? wc : -wc);

            double uzm = (k > 0 && kzm)
                      ? 0.5 * (u[idx_u(i, j, k - 1, nx, ny)] + u[idx_u(i + 1, j, k - 1, nx, ny)])
                      : (kzm ? uc : -uc);
            double uzp = (k + 1 < nz && kzp)
                      ? 0.5 * (u[idx_u(i, j, k + 1, nx, ny)] + u[idx_u(i + 1, j, k + 1, nx, ny)])
                      : (kzp ? uc : -uc);
            double vzm = (k > 0 && kzm)
                      ? 0.5 * (v[idx_v(i, j, k - 1, nx, ny)] + v[idx_v(i, j + 1, k - 1, nx, ny)])
                      : (kzm ? vc : -vc);
            double vzp = (k + 1 < nz && kzp)
                      ? 0.5 * (v[idx_v(i, j, k + 1, nx, ny)] + v[idx_v(i, j + 1, k + 1, nx, ny)])
                      : (kzp ? vc : -vc);

            double wx_c = ((wyp - wym) - (vzp - vzm)) * inv_2h;
            double wy_c = ((uzp - uzm) - (wp  - wm )) * inv_2h;
            double wz_c = ((vp  - vm ) - (up  - um )) * inv_2h;
            return sqrt(wx_c * wx_c + wy_c * wy_c + wz_c * wz_c);
        }
        default:
            return 0.0;
    }
}

int cfd_extract_slice_f32(const CFD3D *sim, int field_id, int plane, int index, float *out_buf) {
    if (!sim || !out_buf) return -1;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;

    if (plane == CFD_SLICE_XZ) {
        int j = cfd_iclamp(index, 0, ny - 1);
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                out_buf[(size_t)k * (size_t)nx + (size_t)i] = (float)eval_cell_field(sim, field_id, i, j, k);
            }
        }
        return 0;
    } else if (plane == CFD_SLICE_XY) {
        int k = cfd_iclamp(index, 0, nz - 1);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                out_buf[(size_t)j * (size_t)nx + (size_t)i] = (float)eval_cell_field(sim, field_id, i, j, k);
            }
        }
        return 0;
    } else if (plane == CFD_SLICE_YZ) {
        int i = cfd_iclamp(index, 0, nx - 1);
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                out_buf[(size_t)k * (size_t)ny + (size_t)j] = (float)eval_cell_field(sim, field_id, i, j, k);
            }
        }
        return 0;
    }
    return -1;
}

int cfd_extract_vec_slice_f32(const CFD3D *sim, int plane, int index, float *out_u1, float *out_u2) {
    if (!sim || !out_u1 || !out_u2) return -1;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;

    if (plane == CFD_SLICE_XZ) {
        /* (u, w) on (nz, nx) */
        int j = cfd_iclamp(index, 0, ny - 1);
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                size_t p = (size_t)k * (size_t)nx + (size_t)i;
                out_u1[p] = (float)eval_cell_field(sim, CFD_FIELD_U, i, j, k);
                out_u2[p] = (float)eval_cell_field(sim, CFD_FIELD_W, i, j, k);
            }
        }
        return 0;
    } else if (plane == CFD_SLICE_XY) {
        /* (u, v) on (ny, nx) */
        int k = cfd_iclamp(index, 0, nz - 1);
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t p = (size_t)j * (size_t)nx + (size_t)i;
                out_u1[p] = (float)eval_cell_field(sim, CFD_FIELD_U, i, j, k);
                out_u2[p] = (float)eval_cell_field(sim, CFD_FIELD_V, i, j, k);
            }
        }
        return 0;
    } else if (plane == CFD_SLICE_YZ) {
        /* (v, w) on (nz, ny) */
        int i = cfd_iclamp(index, 0, nx - 1);
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                size_t p = (size_t)k * (size_t)ny + (size_t)j;
                out_u1[p] = (float)eval_cell_field(sim, CFD_FIELD_V, i, j, k);
                out_u2[p] = (float)eval_cell_field(sim, CFD_FIELD_W, i, j, k);
            }
        }
        return 0;
    }
    return -1;
}

typedef struct PackCtx {
    const CFD3D *sim;
    int field_id;
    double vmin;
    double scale;
    uint8_t *out_u8;
} PackCtx;

static void pack_vol_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    const PackCtx *ctx = (const PackCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny;
    const int fid = ctx->field_id;
    const double vmin = ctx->vmin;
    const double scale = ctx->scale;
    uint8_t *restrict out = ctx->out_u8;

    for (int k = k0; k < k1; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t c = idx_c(i, j, k, nx, ny);
                if (fid != CFD_FIELD_SOLID && sim->cell_type[c] != CFD_CELL_FLUID) {
                    out[c] = 0;
                    continue;
                }
                double val = eval_cell_field(sim, fid, i, j, k);
                double norm = (val - vmin) * scale;
                if (norm <= 0.0) out[c] = 0;
                else if (norm >= 255.0) out[c] = 255;
                else out[c] = (uint8_t)(norm + 0.5);
            }
        }
    }
}

int cfd_pack_volume_u8(const CFD3D *sim, int field_id, double vmin, double vmax, uint8_t *out_u8) {
    if (!sim || !out_u8) return -1;
    double span = vmax - vmin;
    double scale = (fabs(span) > 1e-15) ? (255.0 / span) : 0.0;
    PackCtx ctx = { sim, field_id, vmin, scale, out_u8 };
    cfd_parallel_for_k(sim, sim->nz, 8, &ctx, pack_vol_chunk_fn);
    return 0;
}
