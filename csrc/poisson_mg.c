#include "cfd3d_internal.h"
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define CFD_USE_NEON 1
#endif

static void mg_level_free(MGLevel *L) {
    free(L->wx); L->wx = NULL;
    free(L->wy); L->wy = NULL;
    free(L->wz); L->wz = NULL;
    free(L->wd); L->wd = NULL;
    free(L->diag); L->diag = NULL;
    free(L->inv_diag); L->inv_diag = NULL;
    free(L->x); L->x = NULL;
    free(L->b); L->b = NULL;
    free(L->neumann_offsets); L->neumann_offsets = NULL;
    free(L->neumann_cells); L->neumann_cells = NULL;
    L->n_neumann = 0;
}

void cfd_mg_free(CFD3D *sim) {
    for (int l = 0; l < sim->n_mg_levels; ++l) {
        mg_level_free(&sim->mg[l]);
    }
    sim->n_mg_levels = 0;
    free(sim->l0_mask); sim->l0_mask = NULL;
    free(sim->l0_diag); sim->l0_diag = NULL;
}

static int mg_level_alloc(MGLevel *L, int nx, int ny, int nz) {
    memset(L, 0, sizeof(*L));
    L->nx = nx; L->ny = ny; L->nz = nz;
    size_t nc = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t nu = (size_t)(nx + 1) * (size_t)ny * (size_t)nz;
    size_t nv = (size_t)nx * (size_t)(ny + 1) * (size_t)nz;
    size_t nw = (size_t)nx * (size_t)ny * (size_t)(nz + 1);
    L->ncells = nc;

    L->wx       = (float *)calloc(nu, sizeof(float));
    L->wy       = (float *)calloc(nv, sizeof(float));
    L->wz       = (float *)calloc(nw, sizeof(float));
    L->wd       = (float *)calloc(nc, sizeof(float));
    L->diag     = (float *)calloc(nc, sizeof(float));
    L->inv_diag = (float *)calloc(nc, sizeof(float));
    L->x        = (float *)calloc(nc, sizeof(float));
    L->b        = (float *)calloc(nc, sizeof(float));

    if (!L->wx || !L->wy || !L->wz || !L->wd || !L->diag || !L->inv_diag ||
        !L->x || !L->b) {
        mg_level_free(L);
        return -1;
    }
    return 0;
}

static void mg_compute_diag(MGLevel *L) {
    const int nx = L->nx, ny = L->ny, nz = L->nz;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t c = idx_c(i, j, k, nx, ny);
                double d = (double)L->wx[idx_u(i,     j, k, nx, ny)]
                         + (double)L->wx[idx_u(i + 1, j, k, nx, ny)]
                         + (double)L->wy[idx_v(i, j,     k, nx, ny)]
                         + (double)L->wy[idx_v(i, j + 1, k, nx, ny)]
                         + (double)L->wz[idx_w(i, j, k,     nx, ny)]
                         + (double)L->wz[idx_w(i, j, k + 1, nx, ny)]
                         + (double)L->wd[c];
                L->diag[c] = (float)d;
                L->inv_diag[c] = (d > 1e-12) ? (float)(1.0 / d) : 0.0f;
            }
        }
    }
}

/* Detect disconnected pure-Neumann components (sum(wd) == 0) via BFS */
static int mg_find_neumann_components(MGLevel *L) {
    const int nx = L->nx, ny = L->ny, nz = L->nz;
    const size_t nc = L->ncells;

    uint8_t *visited = (uint8_t *)calloc(nc, sizeof(uint8_t));
    int *queue = (int *)malloc(nc * sizeof(int));
    int *comp_cells = (int *)malloc(nc * sizeof(int));
    int *offsets = (int *)malloc((nc + 1) * sizeof(int));
    if (!visited || !queue || !comp_cells || !offsets) {
        free(visited); free(queue); free(comp_cells); free(offsets);
        return -1;
    }

    int n_neu = 0;
    int total_neu_cells = 0;
    offsets[0] = 0;

    for (size_t start = 0; start < nc; ++start) {
        if (visited[start] || L->diag[start] <= 1e-12f) continue;

        int head = 0, tail = 0;
        queue[tail++] = (int)start;
        visited[start] = 1;
        int has_dirichlet = 0;

        while (head < tail) {
            int cur = queue[head++];
            if (L->wd[cur] > 1e-12f) has_dirichlet = 1;

            int i = cur % nx;
            int rem = cur / nx;
            int j = rem % ny;
            int k = rem / ny;

            if (i > 0 && L->wx[idx_u(i, j, k, nx, ny)] > 0.0f) {
                int nb = cur - 1;
                if (!visited[nb]) { visited[nb] = 1; queue[tail++] = nb; }
            }
            if (i + 1 < nx && L->wx[idx_u(i + 1, j, k, nx, ny)] > 0.0f) {
                int nb = cur + 1;
                if (!visited[nb]) { visited[nb] = 1; queue[tail++] = nb; }
            }
            if (j > 0 && L->wy[idx_v(i, j, k, nx, ny)] > 0.0f) {
                int nb = cur - nx;
                if (!visited[nb]) { visited[nb] = 1; queue[tail++] = nb; }
            }
            if (j + 1 < ny && L->wy[idx_v(i, j + 1, k, nx, ny)] > 0.0f) {
                int nb = cur + nx;
                if (!visited[nb]) { visited[nb] = 1; queue[tail++] = nb; }
            }
            if (k > 0 && L->wz[idx_w(i, j, k, nx, ny)] > 0.0f) {
                int nb = cur - nx * ny;
                if (!visited[nb]) { visited[nb] = 1; queue[tail++] = nb; }
            }
            if (k + 1 < nz && L->wz[idx_w(i, j, k + 1, nx, ny)] > 0.0f) {
                int nb = cur + nx * ny;
                if (!visited[nb]) { visited[nb] = 1; queue[tail++] = nb; }
            }
        }

        if (!has_dirichlet && tail > 0) {
            memcpy(&comp_cells[total_neu_cells], queue, (size_t)tail * sizeof(int));
            total_neu_cells += tail;
            n_neu++;
            offsets[n_neu] = total_neu_cells;
        }
    }

    L->n_neumann = n_neu;
    if (n_neu > 0) {
        L->neumann_offsets = (int *)malloc((size_t)(n_neu + 1) * sizeof(int));
        L->neumann_cells   = (int *)malloc((size_t)total_neu_cells * sizeof(int));
        if (!L->neumann_offsets || !L->neumann_cells) {
            free(visited); free(queue); free(comp_cells); free(offsets);
            return -1;
        }
        memcpy(L->neumann_offsets, offsets, (size_t)(n_neu + 1) * sizeof(int));
        memcpy(L->neumann_cells, comp_cells, (size_t)total_neu_cells * sizeof(int));
    }

    free(visited);
    free(queue);
    free(comp_cells);
    free(offsets);
    return 0;
}

static void subtract_neumann_mean_f64(const MGLevel *L, double *vec) {
    for (int comp = 0; comp < L->n_neumann; ++comp) {
        int s = L->neumann_offsets[comp];
        int e = L->neumann_offsets[comp + 1];
        int n = e - s;
        if (n <= 0) continue;
        double sum = 0.0;
        for (int idx = s; idx < e; ++idx) {
            sum += vec[L->neumann_cells[idx]];
        }
        double mean = sum / (double)n;
        for (int idx = s; idx < e; ++idx) {
            vec[L->neumann_cells[idx]] -= mean;
        }
    }
}

static void subtract_neumann_mean_f32(const MGLevel *L, float *vec) {
    for (int comp = 0; comp < L->n_neumann; ++comp) {
        int s = L->neumann_offsets[comp];
        int e = L->neumann_offsets[comp + 1];
        int n = e - s;
        if (n <= 0) continue;
        double sum = 0.0;
        for (int idx = s; idx < e; ++idx) {
            sum += (double)vec[L->neumann_cells[idx]];
        }
        float mean = (float)(sum / (double)n);
        for (int idx = s; idx < e; ++idx) {
            vec[L->neumann_cells[idx]] -= mean;
        }
    }
}

int cfd_mg_build(CFD3D *sim) {
    cfd_mg_free(sim);

    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    MGLevel *L0 = &sim->mg[0];
    if (mg_level_alloc(L0, nx, ny, nz) != 0) return -1;
    sim->n_mg_levels = 1;

    /* Populate level 0 face weights (exact integers 0/1) and Dirichlet weights (exact integers) */
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 1; i < nx; ++i) {
                size_t fu = idx_u(i, j, k, nx, ny);
                L0->wx[fu] = sim->kx[fu] ? 1.0f : 0.0f;
            }
        }
    }
    for (int k = 0; k < nz; ++k) {
        for (int j = 1; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t fv = idx_v(i, j, k, nx, ny);
                L0->wy[fv] = sim->ky[fv] ? 1.0f : 0.0f;
            }
        }
    }
    for (int k = 1; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t fw = idx_w(i, j, k, nx, ny);
                L0->wz[fw] = sim->kz[fw] ? 1.0f : 0.0f;
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t c = idx_c(i, j, k, nx, ny);
                if (sim->cell_type[c] != CFD_CELL_FLUID) continue;
                float wd = 0.0f;
                if (i == 0 && sim->kx[idx_u(0, j, k, nx, ny)]) {
                    wd += 2.0f; /* BC_VENT */
                }
                if (i == nx - 1 && sim->kx[idx_u(nx, j, k, nx, ny)] && sim->bc_px == BC_OPEN) {
                    wd += 2.0f;
                }
                if (j == 0 && sim->ky[idx_v(i, 0, k, nx, ny)] && sim->bc_my == BC_OPEN) {
                    wd += 2.0f;
                }
                if (j == ny - 1 && sim->ky[idx_v(i, ny, k, nx, ny)] && sim->bc_py == BC_OPEN) {
                    wd += 2.0f;
                }
                if (k == nz - 1 && sim->kz[idx_w(i, j, nz, nx, ny)] && sim->bc_pz == BC_OPEN) {
                    wd += 2.0f;
                }
                L0->wd[c] = wd;
            }
        }
    }

    mg_compute_diag(L0);
    if (mg_find_neumann_components(L0) != 0) {
        cfd_mg_free(sim);
        return -1;
    }

    /* Face-aware Galerkin coarsening: w_F^{2h} = 0.5 * sum w_f^h, w_D^{2h} = 0.5 * sum w_d^h */
    for (int lev = 0; lev < CFD_MAX_MG_LEVELS - 1; ++lev) {
        const MGLevel *Lf = &sim->mg[lev];
        int max_dim = Lf->nx;
        if (Lf->ny > max_dim) max_dim = Lf->ny;
        if (Lf->nz > max_dim) max_dim = Lf->nz;
        if (max_dim <= 4 || Lf->ncells <= 64) break;

        int nxc = (Lf->nx + 1) / 2;
        int nyc = (Lf->ny + 1) / 2;
        int nzc = (Lf->nz + 1) / 2;

        MGLevel *Lc = &sim->mg[lev + 1];
        if (mg_level_alloc(Lc, nxc, nyc, nzc) != 0) {
            cfd_mg_free(sim);
            return -1;
        }
        sim->n_mg_levels = lev + 2;

        /* Coarse interior x-faces */
        for (int K = 0; K < nzc; ++K) {
            int k0 = 2 * K;
            int k1 = (k0 + 1 < Lf->nz) ? (k0 + 1) : k0;
            for (int J = 0; J < nyc; ++J) {
                int j0 = 2 * J;
                int j1 = (j0 + 1 < Lf->ny) ? (j0 + 1) : j0;
                for (int I = 1; I < nxc; ++I) {
                    int ifa = 2 * I;
                    double sum = 0.0;
                    for (int kf = k0; kf <= k1; ++kf) {
                        for (int jf = j0; jf <= j1; ++jf) {
                            sum += (double)Lf->wx[idx_u(ifa, jf, kf, Lf->nx, Lf->ny)];
                        }
                    }
                    Lc->wx[idx_u(I, J, K, nxc, nyc)] = (float)(0.5 * sum);
                }
            }
        }

        /* Coarse interior y-faces */
        for (int K = 0; K < nzc; ++K) {
            int k0 = 2 * K;
            int k1 = (k0 + 1 < Lf->nz) ? (k0 + 1) : k0;
            for (int J = 1; J < nyc; ++J) {
                int jfa = 2 * J;
                for (int I = 0; I < nxc; ++I) {
                    int i0 = 2 * I;
                    int i1 = (i0 + 1 < Lf->nx) ? (i0 + 1) : i0;
                    double sum = 0.0;
                    for (int kf = k0; kf <= k1; ++kf) {
                        for (int if_ = i0; if_ <= i1; ++if_) {
                            sum += (double)Lf->wy[idx_v(if_, jfa, kf, Lf->nx, Lf->ny)];
                        }
                    }
                    Lc->wy[idx_v(I, J, K, nxc, nyc)] = (float)(0.5 * sum);
                }
            }
        }

        /* Coarse interior z-faces */
        for (int K = 1; K < nzc; ++K) {
            int kfa = 2 * K;
            for (int J = 0; J < nyc; ++J) {
                int j0 = 2 * J;
                int j1 = (j0 + 1 < Lf->ny) ? (j0 + 1) : j0;
                for (int I = 0; I < nxc; ++I) {
                    int i0 = 2 * I;
                    int i1 = (i0 + 1 < Lf->nx) ? (i0 + 1) : i0;
                    double sum = 0.0;
                    for (int jf = j0; jf <= j1; ++jf) {
                        for (int if_ = i0; if_ <= i1; ++if_) {
                            sum += (double)Lf->wz[idx_w(if_, jf, kfa, Lf->nx, Lf->ny)];
                        }
                    }
                    Lc->wz[idx_w(I, J, K, nxc, nyc)] = (float)(0.5 * sum);
                }
            }
        }

        /* Coarse Dirichlet weights */
        for (int K = 0; K < nzc; ++K) {
            int k0 = 2 * K;
            int k1 = (k0 + 1 < Lf->nz) ? (k0 + 1) : k0;
            for (int J = 0; J < nyc; ++J) {
                int j0 = 2 * J;
                int j1 = (j0 + 1 < Lf->ny) ? (j0 + 1) : j0;
                for (int I = 0; I < nxc; ++I) {
                    int i0 = 2 * I;
                    int i1 = (i0 + 1 < Lf->nx) ? (i0 + 1) : i0;
                    double sum = 0.0;
                    for (int kf = k0; kf <= k1; ++kf) {
                        for (int jf = j0; jf <= j1; ++jf) {
                            for (int if_ = i0; if_ <= i1; ++if_) {
                                sum += (double)Lf->wd[idx_c(if_, jf, kf, Lf->nx, Lf->ny)];
                            }
                        }
                    }
                    Lc->wd[idx_c(I, J, K, nxc, nyc)] = (float)(0.5 * sum);
                }
            }
        }

        mg_compute_diag(Lc);
    }

    /* Detect pure-Neumann components on coarsest level too */
    MGLevel *Llast = &sim->mg[sim->n_mg_levels - 1];
    if (sim->n_mg_levels > 1) {
        if (mg_find_neumann_components(Llast) != 0) {
            cfd_mg_free(sim);
            return -1;
        }
    }

    /* Pack Level-0 stencil into compact 1-byte mask + 1-byte integer diag (12x smaller than FP64!) */
    sim->l0_mask = (uint8_t *)calloc(L0->ncells, sizeof(uint8_t));
    sim->l0_diag = (uint8_t *)calloc(L0->ncells, sizeof(uint8_t));
    if (!sim->l0_mask || !sim->l0_diag) {
        cfd_mg_free(sim);
        return -1;
    }
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                size_t c = idx_c(i, j, k, nx, ny);
                uint8_t m = 0;
                if (i > 0      && L0->wx[idx_u(i,     j, k, nx, ny)] > 0.5f) m |= 1u;
                if (i + 1 < nx && L0->wx[idx_u(i + 1, j, k, nx, ny)] > 0.5f) m |= 2u;
                if (j > 0      && L0->wy[idx_v(i, j,     k, nx, ny)] > 0.5f) m |= 4u;
                if (j + 1 < ny && L0->wy[idx_v(i, j + 1, k, nx, ny)] > 0.5f) m |= 8u;
                if (k > 0      && L0->wz[idx_w(i, j, k,     nx, ny)] > 0.5f) m |= 16u;
                if (k + 1 < nz && L0->wz[idx_w(i, j, k + 1, nx, ny)] > 0.5f) m |= 32u;
                uint8_t d_u8 = (uint8_t)(L0->diag[c] + 0.5f);
                if (m == 0x3Fu && d_u8 == 6u) m |= 0x40u; /* Fast interior 6-neighbor flag */
                sim->l0_mask[c] = m;
                sim->l0_diag[c] = d_u8;
            }
        }
    }

    /* Free temporary Dirichlet weight arrays wd, and Level-0 redundant float weight arrays */
    for (int lev = 0; lev < sim->n_mg_levels; ++lev) {
        free(sim->mg[lev].wd);
        sim->mg[lev].wd = NULL;
    }
    free(L0->wx);   L0->wx   = NULL;
    free(L0->wy);   L0->wy   = NULL;
    free(L0->wz);   L0->wz   = NULL;
    free(L0->diag); L0->diag = NULL;

    sim->geom.n_mg_levels = sim->n_mg_levels;
    return 0;
}

/* ========================================================================= */
/* FP32 Parallel Multigrid Kernels (Branchless, NEON-Vectorizable)           */
/* ========================================================================= */

/* Zero-initial-guess fused first Red sweep: x[red] = b[red]*inv_d[red], x[black] = 0 */
static void first_red_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    const MGLevel *L = (const MGLevel *)vctx;
    const int nx = L->nx, ny = L->ny;
    const float *restrict inv_d = L->inv_diag;
    const float *restrict b = L->b;
    float *restrict x = L->x;

    for (int k = k0; k < k1; ++k) {
        for (int j = 0; j < ny; ++j) {
            size_t row = idx_c(0, j, k, nx, ny);
            const float *restrict br = b + row;
            const float *restrict idr = inv_d + row;
            float *restrict xr = x + row;
            int odd = (j + k) & 1;
            int i = 0;
            if (odd) {
                xr[0] = 0.0f;
                i = 1;
            }
            for (; i + 1 < nx; i += 2) {
                xr[i]     = br[i] * idr[i];
                xr[i + 1] = 0.0f;
            }
            if (i < nx) {
                xr[i] = br[i] * idr[i];
            }
        }
    }
}

typedef struct RBGSCtx {
    const CFD3D *sim;
    const MGLevel *L;
    int color;
} RBGSCtx;

/* Level-0 ultra-compact RBGS using 1-byte bitmask l0_mask (eliminates 4.7 MB of wx/wy/wz reads!) */
static void rbgs_L0_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    const RBGSCtx *ctx = (const RBGSCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const MGLevel *L = ctx->L;
    const int nx = L->nx, ny = L->ny, nz = L->nz;
    const int color = ctx->color;
    const uint8_t *restrict mask = sim->l0_mask;
    const float *restrict inv_d  = L->inv_diag;
    const float *restrict b      = L->b;
    float *restrict x            = L->x;
    const size_t stride_y = (size_t)nx;
    const size_t stride_z = (size_t)nx * (size_t)ny;

    for (int k = k0; k < k1; ++k) {
        const ptrdiff_t off_zm = (k > 0)      ? -(ptrdiff_t)stride_z : 0;
        const ptrdiff_t off_zp = (k + 1 < nz) ?  (ptrdiff_t)stride_z : 0;

        for (int j = 0; j < ny; ++j) {
            const ptrdiff_t off_ym = (j > 0)      ? -(ptrdiff_t)stride_y : 0;
            const ptrdiff_t off_yp = (j + 1 < ny) ?  (ptrdiff_t)stride_y : 0;

            const size_t row_c = idx_c(0, j, k, nx, ny);
            float *restrict xr = x + row_c;
            const float *restrict x_ym  = xr + off_ym;
            const float *restrict x_yp  = xr + off_yp;
            const float *restrict x_zm  = xr + off_zm;
            const float *restrict x_zp  = xr + off_zp;
            const float *restrict br    = b + row_c;
            const float *restrict idr   = inv_d + row_c;
            const uint8_t *restrict mr  = mask + row_c;

            int i_start = (j + k + color) & 1;

            if (i_start == 0) {
                uint8_t m = mr[0];
                float sum = br[0]
                          + ((m &  2u) ? xr[1]   : 0.0f)
                          + ((m &  4u) ? x_ym[0] : 0.0f)
                          + ((m &  8u) ? x_yp[0] : 0.0f)
                          + ((m & 16u) ? x_zm[0] : 0.0f)
                          + ((m & 32u) ? x_zp[0] : 0.0f);
                xr[0] = sum * idr[0];
                i_start = 2;
            }

            int i = i_start;
#ifdef CFD_USE_NEON
            const float32x4_t v_inv6 = vdupq_n_f32(1.0f / 6.0f);
            for (; i + 7 < nx; i += 8) {
                if ((mr[i] & mr[i + 2] & mr[i + 4] & mr[i + 6] & 0x40u) != 0u) {
                    float32x4x2_t xcur = vld2q_f32(xr + i);
                    float32x4_t x_prev = vdupq_n_f32(xr[i - 1]);
                    float32x4_t x_xm   = vextq_f32(x_prev, xcur.val[1], 3);
                    float32x4_t x_xp   = xcur.val[1];
                    float32x4_t b_ev   = vld2q_f32(br   + i).val[0];
                    float32x4_t ym_ev  = vld2q_f32(x_ym + i).val[0];
                    float32x4_t yp_ev  = vld2q_f32(x_yp + i).val[0];
                    float32x4_t zm_ev  = vld2q_f32(x_zm + i).val[0];
                    float32x4_t zp_ev  = vld2q_f32(x_zp + i).val[0];

                    float32x4_t sx  = vaddq_f32(x_xm, x_xp);
                    float32x4_t sy  = vaddq_f32(ym_ev, yp_ev);
                    float32x4_t sz  = vaddq_f32(zm_ev, zp_ev);
                    float32x4_t sum = vaddq_f32(vaddq_f32(b_ev, sx), vaddq_f32(sy, sz));
                    float32x4_t res = vmulq_f32(sum, v_inv6);

                    vst1q_lane_f32(xr + i,     res, 0);
                    vst1q_lane_f32(xr + i + 2, res, 1);
                    vst1q_lane_f32(xr + i + 4, res, 2);
                    vst1q_lane_f32(xr + i + 6, res, 3);
                } else {
                    for (int t = 0; t < 8; t += 2) {
                        int ii = i + t;
                        uint8_t m = mr[ii];
                        float sum = br[ii]
                                  + ((m &  1u) ? xr[ii - 1] : 0.0f)
                                  + ((m &  2u) ? xr[ii + 1] : 0.0f)
                                  + ((m &  4u) ? x_ym[ii]   : 0.0f)
                                  + ((m &  8u) ? x_yp[ii]   : 0.0f)
                                  + ((m & 16u) ? x_zm[ii]   : 0.0f)
                                  + ((m & 32u) ? x_zp[ii]   : 0.0f);
                        xr[ii] = sum * idr[ii];
                    }
                }
            }
#endif
            for (; i < nx - 1; i += 2) {
                uint8_t m = mr[i];
                float sum = br[i]
                          + ((m &  1u) ? xr[i - 1] : 0.0f)
                          + ((m &  2u) ? xr[i + 1] : 0.0f)
                          + ((m &  4u) ? x_ym[i]   : 0.0f)
                          + ((m &  8u) ? x_yp[i]   : 0.0f)
                          + ((m & 16u) ? x_zm[i]   : 0.0f)
                          + ((m & 32u) ? x_zp[i]   : 0.0f);
                xr[i] = sum * idr[i];
            }

            if (i == nx - 1) {
                uint8_t m = mr[i];
                float sum = br[i]
                          + ((m &  1u) ? xr[i - 1] : 0.0f)
                          + ((m &  4u) ? x_ym[i]   : 0.0f)
                          + ((m &  8u) ? x_yp[i]   : 0.0f)
                          + ((m & 16u) ? x_zm[i]   : 0.0f)
                          + ((m & 32u) ? x_zp[i]   : 0.0f);
                xr[i] = sum * idr[i];
            }
        }
    }
}

static void rbgs_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    const RBGSCtx *ctx = (const RBGSCtx *)vctx;
    const MGLevel *L = ctx->L;
    const int nx = L->nx, ny = L->ny, nz = L->nz;
    const int color = ctx->color;
    const float *restrict wx = L->wx;
    const float *restrict wy = L->wy;
    const float *restrict wz = L->wz;
    const float *restrict inv_d = L->inv_diag;
    const float *restrict b = L->b;
    float *restrict x = L->x;
    const size_t stride_y = (size_t)nx;
    const size_t stride_z = (size_t)nx * (size_t)ny;

    for (int k = k0; k < k1; ++k) {
        const ptrdiff_t off_zm = (k > 0)      ? -(ptrdiff_t)stride_z : 0;
        const ptrdiff_t off_zp = (k + 1 < nz) ?  (ptrdiff_t)stride_z : 0;

        for (int j = 0; j < ny; ++j) {
            const ptrdiff_t off_ym = (j > 0)      ? -(ptrdiff_t)stride_y : 0;
            const ptrdiff_t off_yp = (j + 1 < ny) ?  (ptrdiff_t)stride_y : 0;

            const size_t row_c  = idx_c(0, j, k, nx, ny);
            const size_t row_u  = idx_u(0, j, k, nx, ny);
            const size_t row_v0 = idx_v(0, j,     k, nx, ny);
            const size_t row_v1 = idx_v(0, j + 1, k, nx, ny);
            const size_t row_w0 = idx_w(0, j, k,     nx, ny);
            const size_t row_w1 = idx_w(0, j, k + 1, nx, ny);

            float *restrict xr = x + row_c;
            const float *restrict x_ym = xr + off_ym;
            const float *restrict x_yp = xr + off_yp;
            const float *restrict x_zm = xr + off_zm;
            const float *restrict x_zp = xr + off_zp;
            const float *restrict br   = b + row_c;
            const float *restrict idr  = inv_d + row_c;
            const float *restrict wxr  = wx + row_u;
            const float *restrict wy0  = wy + row_v0;
            const float *restrict wy1  = wy + row_v1;
            const float *restrict wz0  = wz + row_w0;
            const float *restrict wz1  = wz + row_w1;

            int i_start = (j + k + color) & 1;

            /* Left boundary i = 0 (when active color) */
            if (i_start == 0) {
                float sum = br[0]
                          + ((nx > 1) ? wxr[1] * xr[1] : 0.0f)
                          + wy0[0] * x_ym[0] + wy1[0] * x_yp[0]
                          + wz0[0] * x_zm[0] + wz1[0] * x_zp[0];
                xr[0] = sum * idr[0];
                i_start = 2;
            }

            /* Pure branchless interior loop: 1 <= i <= nx - 2 (step 2) */
            int i = i_start;
            for (; i < nx - 1; i += 2) {
                float sum = br[i]
                          + wxr[i]     * xr[i - 1]
                          + wxr[i + 1] * xr[i + 1]
                          + wy0[i]     * x_ym[i]
                          + wy1[i]     * x_yp[i]
                          + wz0[i]     * x_zm[i]
                          + wz1[i]     * x_zp[i];
                xr[i] = sum * idr[i];
            }

            /* Right boundary i = nx - 1 (when active color and nx > 1) */
            if (i == nx - 1) {
                float sum = br[i]
                          + wxr[i] * xr[i - 1]
                          + wy0[i] * x_ym[i] + wy1[i] * x_yp[i]
                          + wz0[i] * x_zm[i] + wz1[i] * x_zp[i];
                xr[i] = sum * idr[i];
            }
        }
    }
}

static inline void mg_rbgs_sweep(const CFD3D *sim, int lev, const MGLevel *L, int color) {
    RBGSCtx ctx = { sim, L, color };
    cfd_k_chunk_fn fn = (lev == 0) ? rbgs_L0_chunk_fn : rbgs_chunk_fn;
    if (L->ncells >= 24576) {
        cfd_parallel_for_k(sim, L->nz, 8, &ctx, fn);
    } else {
        fn(&ctx, 0, L->nz, 0);
    }
}

typedef struct TransCtx {
    const CFD3D *sim;
    const MGLevel *Lf;
    MGLevel *Lc;
} TransCtx;

static inline float eval_res_L0_safe_f32(const CFD3D *sim, const MGLevel *L, int i, int j, int k) {
    const int nx = L->nx, ny = L->ny;
    const size_t c = idx_c(i, j, k, nx, ny);
    uint8_t d = sim->l0_diag[c];
    if (d == 0) return 0.0f;
    uint8_t m = sim->l0_mask[c];
    const size_t sy = (size_t)nx;
    const size_t sz = (size_t)nx * (size_t)ny;
    float ax = (float)d * L->x[c];
    if (m &  1u) ax -= L->x[c - 1];
    if (m &  2u) ax -= L->x[c + 1];
    if (m &  4u) ax -= L->x[c - sy];
    if (m &  8u) ax -= L->x[c + sy];
    if (m & 16u) ax -= L->x[c - sz];
    if (m & 32u) ax -= L->x[c + sz];
    return L->b[c] - ax;
}

static void res_restrict_red_L0_chunk_fn(void *vctx, int K0, int K1, int tid) {
    (void)tid;
    const TransCtx *ctx = (const TransCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const MGLevel *Lf = ctx->Lf;
    MGLevel *Lc = ctx->Lc;
    const int fnx = Lf->nx, fny = Lf->ny, fnz = Lf->nz;
    const int cnx = Lc->nx, cny = Lc->ny;
    const size_t sy = (size_t)fnx;
    const size_t sz = (size_t)fnx * (size_t)fny;
    const float *restrict fx  = Lf->x;
    const float *restrict fb  = Lf->b;
    const uint8_t *restrict fm = sim->l0_mask;
    const uint8_t *restrict fd = sim->l0_diag;
    const float *restrict cid = Lc->inv_diag;
    float *restrict cb = Lc->b;

    for (int K = K0; K < K1; ++K) {
        const int k0 = 2 * K;
        const int k1 = (k0 + 1 < fnz) ? (k0 + 1) : k0;
        const ptrdiff_t off_zm0 = (k0 > 0)       ? -(ptrdiff_t)sz : 0;
        const ptrdiff_t off_zp1 = (k0 + 2 < fnz) ?  (ptrdiff_t)sz : 0;

        for (int J = 0; J < cny; ++J) {
            const int j0 = 2 * J;
            const int j1 = (j0 + 1 < fny) ? (j0 + 1) : j0;
            const size_t crow = idx_c(0, J, K, cnx, cny);
            float *restrict cbr = cb + crow;
            const float *restrict cidr = cid + crow;

            if (k1 > k0 && j1 > j0 && fnx >= 6) {
                const ptrdiff_t off_ym0 = (j0 > 0)       ? -(ptrdiff_t)sy : 0;
                const ptrdiff_t off_yp1 = (j0 + 2 < fny) ?  (ptrdiff_t)sy : 0;

                const size_t o00 = idx_c(0, j0, k0, fnx, fny);
                const size_t o10 = idx_c(0, j1, k0, fnx, fny);
                const size_t o01 = idx_c(0, j0, k1, fnx, fny);
                const size_t o11 = idx_c(0, j1, k1, fnx, fny);

                const float *restrict x00 = fx + o00, *restrict x10 = fx + o10;
                const float *restrict x01 = fx + o01, *restrict x11 = fx + o11;
                const float *restrict b00 = fb + o00, *restrict b10 = fb + o10;
                const float *restrict b01 = fb + o01, *restrict b11 = fb + o11;
                const uint8_t *restrict m00 = fm + o00, *restrict m10 = fm + o10;
                const uint8_t *restrict m01 = fm + o01, *restrict m11 = fm + o11;
                const uint8_t *restrict d00 = fd + o00, *restrict d10 = fd + o10;
                const uint8_t *restrict d01 = fd + o01, *restrict d11 = fd + o11;

                const float *restrict x00_ym = x00 + off_ym0;
                const float *restrict x10_yp = x10 + off_yp1;
                const float *restrict x01_ym = x01 + off_ym0;
                const float *restrict x11_yp = x11 + off_yp1;
                const float *restrict x00_zm = x00 + off_zm0;
                const float *restrict x10_zm = x10 + off_zm0;
                const float *restrict x01_zp = x01 + off_zp1;
                const float *restrict x11_zp = x11 + off_zp1;

                const int full_I = fnx / 2;

                if (cidr[0] > 0.0f) {
                    cbr[0] = eval_res_L0_safe_f32(sim, Lf, 0, j0, k0)
                           + eval_res_L0_safe_f32(sim, Lf, 1, j1, k0)
                           + eval_res_L0_safe_f32(sim, Lf, 1, j0, k1)
                           + eval_res_L0_safe_f32(sim, Lf, 0, j1, k1);
                } else {
                    cbr[0] = 0.0f;
                }

                for (int I = 1; I < full_I - 1; ++I) {
                    const int i0 = 2 * I;
                    const int i1 = i0 + 1;

                    const float xb_100 = x00[i1];
                    const float xb_010 = x10[i0];
                    const float xb_001 = x01[i0];
                    const float xb_111 = x11[i1];

                    uint8_t ma = m00[i0];
                    uint8_t mb = m10[i1];
                    uint8_t mc = m01[i1];
                    uint8_t md = m11[i0];

                    float sum;
                    if ((ma & mb & mc & md & 0x40u) != 0u) {
                        float r000 = b00[i0] - (6.0f * x00[i0]
                                   - (x00[i0 - 1] + xb_100 + x00_ym[i0] + xb_010 + x00_zm[i0] + xb_001));
                        float r110 = b10[i1] - (6.0f * x10[i1]
                                   - (xb_010 + x10[i1 + 1] + xb_100 + x10_yp[i1] + x10_zm[i1] + xb_111));
                        float r101 = b01[i1] - (6.0f * x01[i1]
                                   - (xb_001 + x01[i1 + 1] + x01_ym[i1] + xb_111 + xb_100 + x01_zp[i1]));
                        float r011 = b11[i0] - (6.0f * x11[i0]
                                   - (x11[i0 - 1] + xb_111 + xb_001 + x11_yp[i0] + xb_010 + x11_zp[i0]));
                        sum = (r000 + r110) + (r101 + r011);
                    } else {
                        float r000 = b00[i0] - ((float)d00[i0] * x00[i0]
                                   - ((ma &  1u) ? x00[i0 - 1] : 0.0f)
                                   - ((ma &  2u) ? xb_100      : 0.0f)
                                   - ((ma &  4u) ? x00_ym[i0]  : 0.0f)
                                   - ((ma &  8u) ? xb_010      : 0.0f)
                                   - ((ma & 16u) ? x00_zm[i0]  : 0.0f)
                                   - ((ma & 32u) ? xb_001      : 0.0f));
                        float r110 = b10[i1] - ((float)d10[i1] * x10[i1]
                                   - ((mb &  1u) ? xb_010      : 0.0f)
                                   - ((mb &  2u) ? x10[i1 + 1] : 0.0f)
                                   - ((mb &  4u) ? xb_100      : 0.0f)
                                   - ((mb &  8u) ? x10_yp[i1]  : 0.0f)
                                   - ((mb & 16u) ? x10_zm[i1]  : 0.0f)
                                   - ((mb & 32u) ? xb_111      : 0.0f));
                        float r101 = b01[i1] - ((float)d01[i1] * x01[i1]
                                   - ((mc &  1u) ? xb_001      : 0.0f)
                                   - ((mc &  2u) ? x01[i1 + 1] : 0.0f)
                                   - ((mc &  4u) ? x01_ym[i1]  : 0.0f)
                                   - ((mc &  8u) ? xb_111      : 0.0f)
                                   - ((mc & 16u) ? xb_100      : 0.0f)
                                   - ((mc & 32u) ? x01_zp[i1]  : 0.0f));
                        float r011 = b11[i0] - ((float)d11[i0] * x11[i0]
                                   - ((md &  1u) ? x11[i0 - 1] : 0.0f)
                                   - ((md &  2u) ? xb_111      : 0.0f)
                                   - ((md &  4u) ? xb_001      : 0.0f)
                                   - ((md &  8u) ? x11_yp[i0]  : 0.0f)
                                   - ((md & 16u) ? xb_010      : 0.0f)
                                   - ((md & 32u) ? x11_zp[i0]  : 0.0f));
                        sum = (r000 + r110) + (r101 + r011);
                    }
                    cbr[I] = (cidr[I] > 0.0f) ? sum : 0.0f;
                }

                for (int I = full_I - 1; I < cnx; ++I) {
                    if (cidr[I] == 0.0f) {
                        cbr[I] = 0.0f;
                        continue;
                    }
                    int i0 = 2 * I;
                    int i1 = (i0 + 1 < fnx) ? (i0 + 1) : i0;
                    float sum = 0.0f;
                    for (int kf = k0; kf <= k1; ++kf) {
                        for (int jf = j0; jf <= j1; ++jf) {
                            for (int if_ = i0; if_ <= i1; ++if_) {
                                if (((if_ + jf + kf) & 1) == 0) {
                                    sum += eval_res_L0_safe_f32(sim, Lf, if_, jf, kf);
                                }
                            }
                        }
                    }
                    cbr[I] = sum;
                }
            } else {
                for (int I = 0; I < cnx; ++I) {
                    if (cidr[I] == 0.0f) {
                        cbr[I] = 0.0f;
                        continue;
                    }
                    int i0 = 2 * I;
                    int i1 = (i0 + 1 < fnx) ? (i0 + 1) : i0;
                    float sum = 0.0f;
                    for (int kf = k0; kf <= k1; ++kf) {
                        for (int jf = j0; jf <= j1; ++jf) {
                            for (int if_ = i0; if_ <= i1; ++if_) {
                                if (((if_ + jf + kf) & 1) == 0) {
                                    sum += eval_res_L0_safe_f32(sim, Lf, if_, jf, kf);
                                }
                            }
                        }
                    }
                    cbr[I] = sum;
                }
            }
        }
    }
}

static inline float eval_res_safe_f32(const MGLevel *L, int i, int j, int k) {
    const int nx = L->nx, ny = L->ny, nz = L->nz;
    const size_t c = idx_c(i, j, k, nx, ny);
    float d = L->diag[c];
    if (d == 0.0f) return 0.0f;
    const size_t sy = (size_t)nx;
    const size_t sz = (size_t)nx * (size_t)ny;
    float ax = d * L->x[c];
    if (i > 0)      ax -= L->wx[idx_u(i,     j, k, nx, ny)] * L->x[c - 1];
    if (i + 1 < nx) ax -= L->wx[idx_u(i + 1, j, k, nx, ny)] * L->x[c + 1];
    if (j > 0)      ax -= L->wy[idx_v(i, j,     k, nx, ny)] * L->x[c - sy];
    if (j + 1 < ny) ax -= L->wy[idx_v(i, j + 1, k, nx, ny)] * L->x[c + sy];
    if (k > 0)      ax -= L->wz[idx_w(i, j, k,     nx, ny)] * L->x[c - sz];
    if (k + 1 < nz) ax -= L->wz[idx_w(i, j, k + 1, nx, ny)] * L->x[c + sz];
    return L->b[c] - ax;
}

/* Fused Red-only residual + restriction:
 * Immediately after a Black RBGS sweep, all Black cells have residual == 0.
 * Evaluates ONLY the 4 Red cells in each 2x2x2 block and writes their sum
 * directly to Lc->b, eliminating Lf->r storage, writes, reads, and 1 barrier! */
static void res_restrict_red_chunk_fn(void *vctx, int K0, int K1, int tid) {
    (void)tid;
    const TransCtx *ctx = (const TransCtx *)vctx;
    const MGLevel *Lf = ctx->Lf;
    MGLevel *Lc = ctx->Lc;
    const int fnx = Lf->nx, fny = Lf->ny, fnz = Lf->nz;
    const int cnx = Lc->nx, cny = Lc->ny;
    const size_t sy = (size_t)fnx;
    const size_t sz = (size_t)fnx * (size_t)fny;
    const float *restrict fx  = Lf->x;
    const float *restrict fb  = Lf->b;
    const float *restrict fd  = Lf->diag;
    const float *restrict fwx = Lf->wx;
    const float *restrict fwy = Lf->wy;
    const float *restrict fwz = Lf->wz;
    const float *restrict cid = Lc->inv_diag;
    float *restrict cb = Lc->b;

    for (int K = K0; K < K1; ++K) {
        const int k0 = 2 * K;
        const int k1 = (k0 + 1 < fnz) ? (k0 + 1) : k0;
        const ptrdiff_t off_zm0 = (k0 > 0)       ? -(ptrdiff_t)sz : 0;
        const ptrdiff_t off_zp1 = (k0 + 2 < fnz) ?  (ptrdiff_t)sz : 0;

        for (int J = 0; J < cny; ++J) {
            const int j0 = 2 * J;
            const int j1 = (j0 + 1 < fny) ? (j0 + 1) : j0;
            const size_t crow = idx_c(0, J, K, cnx, cny);
            float *restrict cbr = cb + crow;
            const float *restrict cidr = cid + crow;

            if (k1 > k0 && j1 > j0 && fnx >= 6) {
                const ptrdiff_t off_ym0 = (j0 > 0)       ? -(ptrdiff_t)sy : 0;
                const ptrdiff_t off_yp1 = (j0 + 2 < fny) ?  (ptrdiff_t)sy : 0;

                const size_t o00 = idx_c(0, j0, k0, fnx, fny);
                const size_t o10 = idx_c(0, j1, k0, fnx, fny);
                const size_t o01 = idx_c(0, j0, k1, fnx, fny);
                const size_t o11 = idx_c(0, j1, k1, fnx, fny);

                const float *restrict x00 = fx + o00, *restrict x10 = fx + o10;
                const float *restrict x01 = fx + o01, *restrict x11 = fx + o11;
                const float *restrict b00 = fb + o00, *restrict b10 = fb + o10;
                const float *restrict b01 = fb + o01, *restrict b11 = fb + o11;
                const float *restrict d00 = fd + o00, *restrict d10 = fd + o10;
                const float *restrict d01 = fd + o01, *restrict d11 = fd + o11;

                const float *restrict x00_ym = x00 + off_ym0;
                const float *restrict x10_yp = x10 + off_yp1;
                const float *restrict x01_ym = x01 + off_ym0;
                const float *restrict x11_yp = x11 + off_yp1;
                const float *restrict x00_zm = x00 + off_zm0;
                const float *restrict x10_zm = x10 + off_zm0;
                const float *restrict x01_zp = x01 + off_zp1;
                const float *restrict x11_zp = x11 + off_zp1;

                const float *restrict wx00 = fwx + idx_u(0, j0, k0, fnx, fny);
                const float *restrict wx10 = fwx + idx_u(0, j1, k0, fnx, fny);
                const float *restrict wx01 = fwx + idx_u(0, j0, k1, fnx, fny);
                const float *restrict wx11 = fwx + idx_u(0, j1, k1, fnx, fny);

                const float *restrict wy0_k0 = fwy + idx_v(0, j0,     k0, fnx, fny);
                const float *restrict wy1_k0 = fwy + idx_v(0, j0 + 1, k0, fnx, fny);
                const float *restrict wy2_k0 = fwy + idx_v(0, j0 + 2, k0, fnx, fny);
                const float *restrict wy0_k1 = fwy + idx_v(0, j0,     k1, fnx, fny);
                const float *restrict wy1_k1 = fwy + idx_v(0, j0 + 1, k1, fnx, fny);
                const float *restrict wy2_k1 = fwy + idx_v(0, j0 + 2, k1, fnx, fny);

                const float *restrict wz_j0_k0 = fwz + idx_w(0, j0, k0,     fnx, fny);
                const float *restrict wz_j0_k1 = fwz + idx_w(0, j0, k0 + 1, fnx, fny);
                const float *restrict wz_j0_k2 = fwz + idx_w(0, j0, k0 + 2, fnx, fny);
                const float *restrict wz_j1_k0 = fwz + idx_w(0, j1, k0,     fnx, fny);
                const float *restrict wz_j1_k1 = fwz + idx_w(0, j1, k0 + 1, fnx, fny);
                const float *restrict wz_j1_k2 = fwz + idx_w(0, j1, k0 + 2, fnx, fny);

                const int full_I = fnx / 2;

                /* I = 0 boundary block */
                if (cidr[0] > 0.0f) {
                    cbr[0] = eval_res_safe_f32(Lf, 0, j0, k0)
                           + eval_res_safe_f32(Lf, 1, j1, k0)
                           + eval_res_safe_f32(Lf, 1, j0, k1)
                           + eval_res_safe_f32(Lf, 0, j1, k1);
                } else {
                    cbr[0] = 0.0f;
                }

                /* Pure branchless interior blocks: 1 <= I < full_I - 1 */
                for (int I = 1; I < full_I - 1; ++I) {
                    const int i0 = 2 * I;
                    const int i1 = i0 + 1;

                    /* Black cells shared inside the 2x2x2 block */
                    const float xb_100 = x00[i1];
                    const float xb_010 = x10[i0];
                    const float xb_001 = x01[i0];
                    const float xb_111 = x11[i1];

                    /* Red (0,0,0) */
                    float r000 = b00[i0] - (d00[i0] * x00[i0]
                               - wx00[i0]     * x00[i0 - 1]
                               - wx00[i1]     * xb_100
                               - wy0_k0[i0]   * x00_ym[i0]
                               - wy1_k0[i0]   * xb_010
                               - wz_j0_k0[i0] * x00_zm[i0]
                               - wz_j0_k1[i0] * xb_001);

                    /* Red (1,1,0) */
                    float r110 = b10[i1] - (d10[i1] * x10[i1]
                               - wx10[i1]     * xb_010
                               - wx10[i1 + 1] * x10[i1 + 1]
                               - wy1_k0[i1]   * xb_100
                               - wy2_k0[i1]   * x10_yp[i1]
                               - wz_j1_k0[i1] * x10_zm[i1]
                               - wz_j1_k1[i1] * xb_111);

                    /* Red (1,0,1) */
                    float r101 = b01[i1] - (d01[i1] * x01[i1]
                               - wx01[i1]     * xb_001
                               - wx01[i1 + 1] * x01[i1 + 1]
                               - wy0_k1[i1]   * x01_ym[i1]
                               - wy1_k1[i1]   * xb_111
                               - wz_j0_k1[i1] * xb_100
                               - wz_j0_k2[i1] * x01_zp[i1]);

                    /* Red (0,1,1) */
                    float r011 = b11[i0] - (d11[i0] * x11[i0]
                               - wx11[i0]     * x11[i0 - 1]
                               - wx11[i1]     * xb_111
                               - wy1_k1[i0]   * xb_001
                               - wy2_k1[i0]   * x11_yp[i0]
                               - wz_j1_k1[i0] * xb_010
                               - wz_j1_k2[i0] * x11_zp[i0]);

                    float sum = (r000 + r110) + (r101 + r011);
                    cbr[I] = (cidr[I] > 0.0f) ? sum : 0.0f;
                }

                /* Tail blocks I = full_I - 1 .. cnx - 1 */
                for (int I = full_I - 1; I < cnx; ++I) {
                    if (cidr[I] == 0.0f) {
                        cbr[I] = 0.0f;
                        continue;
                    }
                    int i0 = 2 * I;
                    int i1 = (i0 + 1 < fnx) ? (i0 + 1) : i0;
                    float sum = 0.0f;
                    for (int kf = k0; kf <= k1; ++kf) {
                        for (int jf = j0; jf <= j1; ++jf) {
                            for (int if_ = i0; if_ <= i1; ++if_) {
                                if (((if_ + jf + kf) & 1) == 0) {
                                    sum += eval_res_safe_f32(Lf, if_, jf, kf);
                                }
                            }
                        }
                    }
                    cbr[I] = sum;
                }
            } else {
                for (int I = 0; I < cnx; ++I) {
                    if (cidr[I] == 0.0f) {
                        cbr[I] = 0.0f;
                        continue;
                    }
                    int i0 = 2 * I;
                    int i1 = (i0 + 1 < fnx) ? (i0 + 1) : i0;
                    float sum = 0.0f;
                    for (int kf = k0; kf <= k1; ++kf) {
                        for (int jf = j0; jf <= j1; ++jf) {
                            for (int if_ = i0; if_ <= i1; ++if_) {
                                if (((if_ + jf + kf) & 1) == 0) {
                                    sum += eval_res_safe_f32(Lf, if_, jf, kf);
                                }
                            }
                        }
                    }
                    cbr[I] = sum;
                }
            }
        }
    }
}

static void prolong_chunk_fn(void *vctx, int K0, int K1, int tid) {
    (void)tid;
    const TransCtx *ctx = (const TransCtx *)vctx;
    const MGLevel *Lf = ctx->Lf;
    const MGLevel *Lc = ctx->Lc;
    const int fnx = Lf->nx, fny = Lf->ny, fnz = Lf->nz;
    const int cnx = Lc->nx, cny = Lc->ny;
    const float *restrict cx = Lc->x;
    const float *restrict fid = Lf->inv_diag;
    float *restrict fx = Lf->x;

    for (int K = K0; K < K1; ++K) {
        const int k0 = 2 * K;
        const int k1 = (k0 + 1 < fnz) ? (k0 + 1) : k0;
        for (int J = 0; J < cny; ++J) {
            const int j0 = 2 * J;
            const int j1 = (j0 + 1 < fny) ? (j0 + 1) : j0;
            const float *restrict cxr = cx + idx_c(0, J, K, cnx, cny);

            if (k1 > k0 && j1 > j0) {
                const size_t o00 = idx_c(0, j0, k0, fnx, fny);
                const size_t o10 = idx_c(0, j1, k0, fnx, fny);
                const size_t o01 = idx_c(0, j0, k1, fnx, fny);
                const size_t o11 = idx_c(0, j1, k1, fnx, fny);
                float *restrict x00 = fx + o00, *restrict x10 = fx + o10;
                float *restrict x01 = fx + o01, *restrict x11 = fx + o11;
                const float *restrict d00 = fid + o00, *restrict d10 = fid + o10;
                const float *restrict d01 = fid + o01, *restrict d11 = fid + o11;
                const int full_I = fnx / 2;
                for (int I = 0; I < full_I; ++I) {
                    float val = cxr[I];
                    int i0 = 2 * I;
                    x00[i0]     += (d00[i0]     > 0.0f) ? val : 0.0f;
                    x00[i0 + 1] += (d00[i0 + 1] > 0.0f) ? val : 0.0f;
                    x10[i0]     += (d10[i0]     > 0.0f) ? val : 0.0f;
                    x10[i0 + 1] += (d10[i0 + 1] > 0.0f) ? val : 0.0f;
                    x01[i0]     += (d01[i0]     > 0.0f) ? val : 0.0f;
                    x01[i0 + 1] += (d01[i0 + 1] > 0.0f) ? val : 0.0f;
                    x11[i0]     += (d11[i0]     > 0.0f) ? val : 0.0f;
                    x11[i0 + 1] += (d11[i0 + 1] > 0.0f) ? val : 0.0f;
                }
                if (full_I < cnx) {
                    float val = cxr[full_I];
                    int i0 = 2 * full_I;
                    if (d00[i0] > 0.0f) x00[i0] += val;
                    if (d10[i0] > 0.0f) x10[i0] += val;
                    if (d01[i0] > 0.0f) x01[i0] += val;
                    if (d11[i0] > 0.0f) x11[i0] += val;
                }
            } else {
                for (int I = 0; I < cnx; ++I) {
                    float val = cxr[I];
                    if (val == 0.0f) continue;
                    int i0 = 2 * I;
                    int i1 = (i0 + 1 < fnx) ? (i0 + 1) : i0;
                    for (int kf = k0; kf <= k1; ++kf) {
                        for (int jf = j0; jf <= j1; ++jf) {
                            for (int if_ = i0; if_ <= i1; ++if_) {
                                size_t cf = idx_c(if_, jf, kf, fnx, fny);
                                if (fid[cf] > 0.0f) fx[cf] += val;
                            }
                        }
                    }
                }
            }
        }
    }
}

/* Strictly symmetric FP32 V-cycle:
 * On lev == 0, caller has already performed zero + first Red sweep into L0->x.
 * On lev > 0, L->x is initialized via fused zero + first Red sweep (or zeroed on coarsest). */
static void mg_vcycle_f32(CFD3D *sim, int lev) {
    MGLevel *L = &sim->mg[lev];
    const int n_smooth = sim->mg_smooth_sweeps;
    const int n_coarse = sim->mg_coarse_sweeps;

    if (lev == sim->n_mg_levels - 1) {
        if (lev > 0) {
            first_red_chunk_fn(L, 0, L->nz, 0);
        }
        mg_rbgs_sweep(sim, lev, L, 1);
        for (int s = 1; s < n_coarse; ++s) {
            mg_rbgs_sweep(sim, lev, L, 0);
            mg_rbgs_sweep(sim, lev, L, 1);
        }
        for (int s = 0; s < n_coarse; ++s) {
            mg_rbgs_sweep(sim, lev, L, 1);
            mg_rbgs_sweep(sim, lev, L, 0);
        }
        if (L->n_neumann > 0) {
            subtract_neumann_mean_f32(L, L->x);
        }
        return;
    }

    /* Pre-smooth #1: first Red already done on lev == 0; fused zero+Red on lev > 0 */
    if (lev > 0) {
        if (L->ncells >= 24576) {
            cfd_parallel_for_k(sim, L->nz, 8, L, first_red_chunk_fn);
        } else {
            first_red_chunk_fn(L, 0, L->nz, 0);
        }
    }
    mg_rbgs_sweep(sim, lev, L, 1);
    for (int s = 1; s < n_smooth; ++s) {
        mg_rbgs_sweep(sim, lev, L, 0);
        mg_rbgs_sweep(sim, lev, L, 1);
    }

    /* Fused Red-only residual + restriction directly into Lc->b */
    MGLevel *Lc = &sim->mg[lev + 1];
    TransCtx tctx = { sim, L, Lc };
    cfd_k_chunk_fn rr_fn = (lev == 0) ? res_restrict_red_L0_chunk_fn : res_restrict_red_chunk_fn;
    if (L->ncells >= 24576) {
        cfd_parallel_for_k(sim, Lc->nz, 8, &tctx, rr_fn);
    } else {
        rr_fn(&tctx, 0, Lc->nz, 0);
    }

    /* Coarse grid solve */
    mg_vcycle_f32(sim, lev + 1);

    /* Piecewise-constant prolongation x += P * x_c */
    if (Lc->ncells >= 8192) {
        cfd_parallel_for_k(sim, Lc->nz, 8, &tctx, prolong_chunk_fn);
    } else {
        prolong_chunk_fn(&tctx, 0, Lc->nz, 0);
    }

    /* Post-smooth (strictly symmetric B -> R) */
    for (int s = 0; s < n_smooth; ++s) {
        mg_rbgs_sweep(sim, lev, L, 1);
        mg_rbgs_sweep(sim, lev, L, 0);
    }
}

/* ========================================================================= */
/* Strict FP64 Outer Flexible-PCG Kernels (Fused & Cache-Line Padded)        */
/* ========================================================================= */

/* Strict FP64 finest-level operator y = b - A*x fused with (r_sq, r_inf) */
typedef struct ApplyResNormCtx {
    const CFD3D *sim;
    const double *x;
    const double *b;
    double *r;
    double sq_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
    double inf_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
} ApplyResNormCtx;

static void apply_a_L0_res_norm_chunk_fn(void *vctx, int k0, int k1, int tid) {
    ApplyResNormCtx *ctx = (ApplyResNormCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const uint8_t *restrict mask = sim->l0_mask;
    const uint8_t *restrict diag = sim->l0_diag;
    const double *restrict x     = ctx->x;
    const double *restrict b     = ctx->b;
    double *restrict r           = ctx->r;
    const size_t stride_y = (size_t)nx;
    const size_t stride_z = (size_t)nx * (size_t)ny;

    double sq = 0.0, inf = 0.0;

    for (int k = k0; k < k1; ++k) {
        const ptrdiff_t off_zm = (k > 0)      ? -(ptrdiff_t)stride_z : 0;
        const ptrdiff_t off_zp = (k + 1 < nz) ?  (ptrdiff_t)stride_z : 0;

        for (int j = 0; j < ny; ++j) {
            const ptrdiff_t off_ym = (j > 0)      ? -(ptrdiff_t)stride_y : 0;
            const ptrdiff_t off_yp = (j + 1 < ny) ?  (ptrdiff_t)stride_y : 0;

            const size_t row_c = idx_c(0, j, k, nx, ny);
            const double *restrict xr   = x + row_c;
            const double *restrict x_ym = xr + off_ym;
            const double *restrict x_yp = xr + off_yp;
            const double *restrict x_zm = xr + off_zm;
            const double *restrict x_zp = xr + off_zp;
            const double *restrict br   = b + row_c;
            const uint8_t *restrict mr  = mask + row_c;
            const uint8_t *restrict dr  = diag + row_c;
            double *restrict rr         = r + row_c;

            /* i = 0 */
            {
                uint8_t m = mr[0];
                double ax = (double)dr[0] * xr[0]
                          - ((m &  2u) ? xr[1]   : 0.0)
                          - ((m &  4u) ? x_ym[0] : 0.0)
                          - ((m &  8u) ? x_yp[0] : 0.0)
                          - ((m & 16u) ? x_zm[0] : 0.0)
                          - ((m & 32u) ? x_zp[0] : 0.0);
                double rv = br[0] - ax;
                rr[0] = rv;
                sq += rv * rv;
                inf = fmax(inf, fabs(rv));
            }

            for (int i = 1; i < nx - 1; ++i) {
                uint8_t m = mr[i];
                double ax = (double)dr[i] * xr[i]
                          - ((m &  1u) ? xr[i - 1] : 0.0)
                          - ((m &  2u) ? xr[i + 1] : 0.0)
                          - ((m &  4u) ? x_ym[i]   : 0.0)
                          - ((m &  8u) ? x_yp[i]   : 0.0)
                          - ((m & 16u) ? x_zm[i]   : 0.0)
                          - ((m & 32u) ? x_zp[i]   : 0.0);
                double rv = br[i] - ax;
                rr[i] = rv;
                sq += rv * rv;
                inf = fmax(inf, fabs(rv));
            }

            if (nx > 1) {
                int i = nx - 1;
                uint8_t m = mr[i];
                double ax = (double)dr[i] * xr[i]
                          - ((m &  1u) ? xr[i - 1] : 0.0)
                          - ((m &  4u) ? x_ym[i]   : 0.0)
                          - ((m &  8u) ? x_yp[i]   : 0.0)
                          - ((m & 16u) ? x_zm[i]   : 0.0)
                          - ((m & 32u) ? x_zp[i]   : 0.0);
                double rv = br[i] - ax;
                rr[i] = rv;
                sq += rv * rv;
                inf = fmax(inf, fabs(rv));
            }
        }
    }

    ctx->sq_t[tid * CFD_CACHE_PAD]  = sq;
    ctx->inf_t[tid * CFD_CACHE_PAD] = inf;
}

/* Strict FP64 finest-level operator ap = A*p fused with pap = dot(p, ap) */
typedef struct ApplyADotCtx {
    const CFD3D *sim;
    const double *p;
    double *ap;
    double pap_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
} ApplyADotCtx;

static void apply_a_L0_dot_chunk_fn(void *vctx, int k0, int k1, int tid) {
    ApplyADotCtx *ctx = (ApplyADotCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const uint8_t *restrict mask = sim->l0_mask;
    const uint8_t *restrict diag = sim->l0_diag;
    const double *restrict p     = ctx->p;
    double *restrict ap          = ctx->ap;
    const size_t stride_y = (size_t)nx;
    const size_t stride_z = (size_t)nx * (size_t)ny;

    double pap = 0.0;

    for (int k = k0; k < k1; ++k) {
        const ptrdiff_t off_zm = (k > 0)      ? -(ptrdiff_t)stride_z : 0;
        const ptrdiff_t off_zp = (k + 1 < nz) ?  (ptrdiff_t)stride_z : 0;

        for (int j = 0; j < ny; ++j) {
            const ptrdiff_t off_ym = (j > 0)      ? -(ptrdiff_t)stride_y : 0;
            const ptrdiff_t off_yp = (j + 1 < ny) ?  (ptrdiff_t)stride_y : 0;

            const size_t row_c = idx_c(0, j, k, nx, ny);
            const double *restrict pr   = p + row_c;
            const double *restrict p_ym = pr + off_ym;
            const double *restrict p_yp = pr + off_yp;
            const double *restrict p_zm = pr + off_zm;
            const double *restrict p_zp = pr + off_zp;
            const uint8_t *restrict mr  = mask + row_c;
            const uint8_t *restrict dr  = diag + row_c;
            double *restrict apr        = ap + row_c;

            /* i = 0 */
            {
                uint8_t m = mr[0];
                double ax = (double)dr[0] * pr[0]
                          - ((m &  2u) ? pr[1]   : 0.0)
                          - ((m &  4u) ? p_ym[0] : 0.0)
                          - ((m &  8u) ? p_yp[0] : 0.0)
                          - ((m & 16u) ? p_zm[0] : 0.0)
                          - ((m & 32u) ? p_zp[0] : 0.0);
                apr[0] = ax;
                pap += pr[0] * ax;
            }

            /* Pure branchless unit-stride interior i = 1..nx-2 */
            int i = 1;
#ifdef CFD_USE_NEON
            const float64x2_t v_six = vdupq_n_f64(6.0);
            float64x2_t v_pap0 = vdupq_n_f64(0.0);
            float64x2_t v_pap1 = vdupq_n_f64(0.0);
            for (; i + 4 < nx; i += 4) {
                if ((mr[i] & mr[i + 1] & mr[i + 2] & mr[i + 3] & 0x40u) != 0u) {
                    float64x2_t p0  = vld1q_f64(pr + i);
                    float64x2_t p1  = vld1q_f64(pr + i + 2);
                    float64x2_t nb0 = vaddq_f64(
                        vaddq_f64(vld1q_f64(pr + i - 1), vld1q_f64(pr + i + 1)),
                        vaddq_f64(vaddq_f64(vld1q_f64(p_ym + i), vld1q_f64(p_yp + i)),
                                  vaddq_f64(vld1q_f64(p_zm + i), vld1q_f64(p_zp + i))));
                    float64x2_t nb1 = vaddq_f64(
                        vaddq_f64(vld1q_f64(pr + i + 1), vld1q_f64(pr + i + 3)),
                        vaddq_f64(vaddq_f64(vld1q_f64(p_ym + i + 2), vld1q_f64(p_yp + i + 2)),
                                  vaddq_f64(vld1q_f64(p_zm + i + 2), vld1q_f64(p_zp + i + 2))));
                    float64x2_t ax0 = vsubq_f64(vmulq_f64(v_six, p0), nb0);
                    float64x2_t ax1 = vsubq_f64(vmulq_f64(v_six, p1), nb1);
                    vst1q_f64(apr + i,     ax0);
                    vst1q_f64(apr + i + 2, ax1);
                    v_pap0 = vfmaq_f64(v_pap0, p0, ax0);
                    v_pap1 = vfmaq_f64(v_pap1, p1, ax1);
                } else {
                    for (int t = 0; t < 4; ++t) {
                        int ii = i + t;
                        uint8_t m = mr[ii];
                        double ax = (double)dr[ii] * pr[ii]
                                  - ((m &  1u) ? pr[ii - 1] : 0.0)
                                  - ((m &  2u) ? pr[ii + 1] : 0.0)
                                  - ((m &  4u) ? p_ym[ii]   : 0.0)
                                  - ((m &  8u) ? p_yp[ii]   : 0.0)
                                  - ((m & 16u) ? p_zm[ii]   : 0.0)
                                  - ((m & 32u) ? p_zp[ii]   : 0.0);
                        apr[ii] = ax;
                        pap += pr[ii] * ax;
                    }
                }
            }
            pap += vaddvq_f64(vaddq_f64(v_pap0, v_pap1));
#endif
            for (; i < nx - 1; ++i) {
                uint8_t m = mr[i];
                double ax = (double)dr[i] * pr[i]
                          - ((m &  1u) ? pr[i - 1] : 0.0)
                          - ((m &  2u) ? pr[i + 1] : 0.0)
                          - ((m &  4u) ? p_ym[i]   : 0.0)
                          - ((m &  8u) ? p_yp[i]   : 0.0)
                          - ((m & 16u) ? p_zm[i]   : 0.0)
                          - ((m & 32u) ? p_zp[i]   : 0.0);
                apr[i] = ax;
                pap += pr[i] * ax;
            }

            /* i = nx - 1 */
            if (nx > 1) {
                int i = nx - 1;
                uint8_t m = mr[i];
                double ax = (double)dr[i] * pr[i]
                          - ((m &  1u) ? pr[i - 1] : 0.0)
                          - ((m &  4u) ? p_ym[i]   : 0.0)
                          - ((m &  8u) ? p_yp[i]   : 0.0)
                          - ((m & 16u) ? p_zm[i]   : 0.0)
                          - ((m & 32u) ? p_zp[i]   : 0.0);
                apr[i] = ax;
                pap += pr[i] * ax;
            }
        }
    }

    ctx->pap_t[tid * CFD_CACHE_PAD] = pap;
}

/* Fused FP64 r -> scaled FP32 L0->b + zero & first Red sweep into L0->x */
typedef struct PrepMGCtx {
    const MGLevel *L0;
    const double *r;
    double inv_scale;
} PrepMGCtx;

static void prep_mg_first_red_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    const PrepMGCtx *ctx = (const PrepMGCtx *)vctx;
    const MGLevel *L = ctx->L0;
    const int nx = L->nx, ny = L->ny;
    const double inv_s = ctx->inv_scale;
    const double *restrict r = ctx->r;
    const float *restrict inv_d = L->inv_diag;
    float *restrict lb = L->b;
    float *restrict lx = L->x;

    for (int k = k0; k < k1; ++k) {
        for (int j = 0; j < ny; ++j) {
            size_t row = idx_c(0, j, k, nx, ny);
            const double *restrict rr = r + row;
            const float *restrict idr = inv_d + row;
            float *restrict lbr = lb + row;
            float *restrict lxr = lx + row;
            int odd = (j + k) & 1;
            int i = 0;
            if (odd) {
                lbr[0] = (float)(rr[0] * inv_s);
                lxr[0] = 0.0f;
                i = 1;
            }
            for (; i + 1 < nx; i += 2) {
                float b0 = (float)(rr[i]     * inv_s);
                float b1 = (float)(rr[i + 1] * inv_s);
                lbr[i]     = b0;
                lbr[i + 1] = b1;
                lxr[i]     = b0 * idr[i];
                lxr[i + 1] = 0.0f;
            }
            if (i < nx) {
                float b0 = (float)(rr[i] * inv_s);
                lbr[i] = b0;
                lxr[i] = b0 * idr[i];
            }
        }
    }
}

/* Fused FP32 L0->x * scale -> FP64 z (and optionally p = z, rz = r.z, zap = ap.z) */
typedef struct PostMGCtx {
    const float *lx;
    const double *r;
    const double *ap; /* NULL on iter 0 (also copies p = z) */
    double *z;
    double *p;        /* non-NULL on iter 0 when n_neumann == 0 */
    double scale;
    size_t slab;
    double rz_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
    double zap_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
} PostMGCtx;

static void post_mg_convert_only_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    const PostMGCtx *ctx = (const PostMGCtx *)vctx;
    const size_t n0 = (size_t)k0 * ctx->slab;
    const size_t n1 = (size_t)k1 * ctx->slab;
    const float *restrict lx = ctx->lx + n0;
    double *restrict z = ctx->z + n0;
    const double s = ctx->scale;
    const size_t len = n1 - n0;
    for (size_t i = 0; i < len; ++i) {
        z[i] = (double)lx[i] * s;
    }
}

static void post_mg_init_chunk_fn(void *vctx, int k0, int k1, int tid) {
    PostMGCtx *ctx = (PostMGCtx *)vctx;
    const size_t n0 = (size_t)k0 * ctx->slab;
    const size_t n1 = (size_t)k1 * ctx->slab;
    const float *restrict lx = ctx->lx + n0;
    const double *restrict r = ctx->r + n0;
    double *restrict z = ctx->z + n0;
    double *restrict p = ctx->p + n0;
    const double s = ctx->scale;
    const size_t len = n1 - n0;
    double rz = 0.0;
    for (size_t i = 0; i < len; ++i) {
        double zv = (double)lx[i] * s;
        z[i] = zv;
        p[i] = zv;
        rz += r[i] * zv;
    }
    ctx->rz_t[tid * CFD_CACHE_PAD] = rz;
}

static void post_mg_dot2_chunk_fn(void *vctx, int k0, int k1, int tid) {
    PostMGCtx *ctx = (PostMGCtx *)vctx;
    const size_t n0 = (size_t)k0 * ctx->slab;
    const size_t n1 = (size_t)k1 * ctx->slab;
    const float *restrict lx = ctx->lx + n0;
    const double *restrict r = ctx->r + n0;
    const double *restrict ap = ctx->ap + n0;
    double *restrict z = ctx->z + n0;
    const double s = ctx->scale;
    const size_t len = n1 - n0;
    double rz = 0.0, zap = 0.0;
    for (size_t i = 0; i < len; ++i) {
        double zv = (double)lx[i] * s;
        z[i] = zv;
        rz  += r[i]  * zv;
        zap += ap[i] * zv;
    }
    ctx->rz_t[tid * CFD_CACHE_PAD]  = rz;
    ctx->zap_t[tid * CFD_CACHE_PAD] = zap;
}

typedef struct Dot2Ctx {
    const double *a;
    const double *b;
    const double *c; /* optional second vector paired with a */
    size_t slab;
    double ab_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
    double ac_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
} Dot2Ctx;

static void dot1_chunk_fn(void *vctx, int k0, int k1, int tid) {
    Dot2Ctx *ctx = (Dot2Ctx *)vctx;
    const size_t n0 = (size_t)k0 * ctx->slab;
    const size_t len = (size_t)(k1 - k0) * ctx->slab;
    const double *restrict a = ctx->a + n0;
    const double *restrict b = ctx->b + n0;
    double s = 0.0;
    for (size_t i = 0; i < len; ++i) s += a[i] * b[i];
    ctx->ab_t[tid * CFD_CACHE_PAD] = s;
}

static void dot2_chunk_fn(void *vctx, int k0, int k1, int tid) {
    Dot2Ctx *ctx = (Dot2Ctx *)vctx;
    const size_t n0 = (size_t)k0 * ctx->slab;
    const size_t len = (size_t)(k1 - k0) * ctx->slab;
    const double *restrict a = ctx->a + n0;
    const double *restrict b = ctx->b + n0;
    const double *restrict c = ctx->c + n0;
    double sab = 0.0, sac = 0.0;
    for (size_t i = 0; i < len; ++i) {
        double av = a[i];
        sab += av * b[i];
        sac += av * c[i];
    }
    ctx->ab_t[tid * CFD_CACHE_PAD] = sab;
    ctx->ac_t[tid * CFD_CACHE_PAD] = sac;
}

typedef struct AxpyNormCtx {
    double alpha;
    const double *p;
    const double *ap;
    double *phi;
    double *r;
    size_t slab;
    double sq_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
    double inf_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
} AxpyNormCtx;

static void axpy_norm_chunk_fn(void *vctx, int k0, int k1, int tid) {
    AxpyNormCtx *ctx = (AxpyNormCtx *)vctx;
    const size_t n0 = (size_t)k0 * ctx->slab;
    const size_t len = (size_t)(k1 - k0) * ctx->slab;
    const double alpha = ctx->alpha;
    const double *restrict p = ctx->p + n0;
    const double *restrict ap = ctx->ap + n0;
    double *restrict phi = ctx->phi + n0;
    double *restrict r = ctx->r + n0;
    double sq = 0.0, inf = 0.0;
    for (size_t i = 0; i < len; ++i) {
        phi[i] += alpha * p[i];
        double rv = r[i] - alpha * ap[i];
        r[i] = rv;
        sq += rv * rv;
        inf = fmax(inf, fabs(rv));
    }
    ctx->sq_t[tid * CFD_CACHE_PAD]  = sq;
    ctx->inf_t[tid * CFD_CACHE_PAD] = inf;
}

typedef struct PUpdateCtx {
    double beta;
    const double *z;
    double *p;
    size_t slab;
} PUpdateCtx;

static void p_update_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    const PUpdateCtx *ctx = (const PUpdateCtx *)vctx;
    const size_t n0 = (size_t)k0 * ctx->slab;
    const size_t len = (size_t)(k1 - k0) * ctx->slab;
    const double beta = ctx->beta;
    const double *restrict z = ctx->z + n0;
    double *restrict p = ctx->p + n0;
    for (size_t i = 0; i < len; ++i) {
        p[i] = z[i] + beta * p[i];
    }
}

typedef struct RhsCtx {
    const CFD3D *sim;
    double scale;
    const double *u, *v, *w;
    double *rhs;
    double bsq_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
} RhsCtx;

static void assemble_rhs_chunk_fn(void *vctx, int k0, int k1, int tid) {
    RhsCtx *ctx = (RhsCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double scale = ctx->scale;
    const double *restrict u = ctx->u;
    const double *restrict v = ctx->v;
    const double *restrict w = ctx->w;
    const uint8_t *restrict ct = sim->cell_type;
    const uint8_t *restrict kx = sim->kx;
    const uint8_t *restrict ky = sim->ky;
    const uint8_t *restrict kz = sim->kz;
    double *restrict rhs = ctx->rhs;
    const int open_px = (sim->bc_px == BC_OPEN);
    const int open_my = (sim->bc_my == BC_OPEN);
    const int open_py = (sim->bc_py == BC_OPEN);
    const int open_pz = (sim->bc_pz == BC_OPEN);
    double bsq = 0.0;

    for (int k = k0; k < k1; ++k) {
        const int is_k_top = (k == nz - 1) && open_pz;
        for (int j = 0; j < ny; ++j) {
            const size_t row_c  = idx_c(0, j, k, nx, ny);
            const size_t row_u  = idx_u(0, j, k, nx, ny);
            const size_t row_v0 = idx_v(0, j,     k, nx, ny);
            const size_t row_v1 = idx_v(0, j + 1, k, nx, ny);
            const size_t row_w0 = idx_w(0, j, k,     nx, ny);
            const size_t row_w1 = idx_w(0, j, k + 1, nx, ny);

            const double *restrict ur  = u + row_u;
            const double *restrict v0  = v + row_v0;
            const double *restrict v1  = v + row_v1;
            const double *restrict w0  = w + row_w0;
            const double *restrict w1  = w + row_w1;
            const uint8_t *restrict ctr = ct + row_c;
            double *restrict br = rhs + row_c;

            const int is_j_lo = (j == 0) && open_my;
            const int is_j_hi = (j == ny - 1) && open_py;

            for (int i = 0; i < nx; ++i) {
                if (ctr[i] != CFD_CELL_FLUID) {
                    br[i] = 0.0;
                    continue;
                }
                double div_unit = (ur[i + 1] - ur[i])
                                + (v1[i]     - v0[i])
                                + (w1[i]     - w0[i]);
                double bv = -scale * div_unit;

                if (i == 0 && kx[row_u]) {
                    bv += 2.0 * sim->phi_bc_mx[(size_t)k * (size_t)ny + (size_t)j];
                }
                if (i == nx - 1 && open_px && kx[row_u + (size_t)nx]) {
                    bv += 2.0 * sim->phi_bc_px[(size_t)k * (size_t)ny + (size_t)j];
                }
                if (is_j_lo && ky[row_v0 + (size_t)i]) {
                    bv += 2.0 * sim->phi_bc_my[(size_t)k * (size_t)nx + (size_t)i];
                }
                if (is_j_hi && ky[row_v1 + (size_t)i]) {
                    bv += 2.0 * sim->phi_bc_py[(size_t)k * (size_t)nx + (size_t)i];
                }
                if (is_k_top && kz[row_w1 + (size_t)i]) {
                    bv += 2.0 * sim->phi_bc_pz[(size_t)j * (size_t)nx + (size_t)i];
                }
                br[i] = bv;
                bsq += bv * bv;
            }
        }
    }

    ctx->bsq_t[tid * CFD_CACHE_PAD] = bsq;
}

int cfd_poisson_solve(CFD3D *sim, double dt, const double *u_star, const double *v_star,
                      const double *w_star, double *phi, int *out_iters, double *out_rel_res) {
    MGLevel *L0 = &sim->mg[0];
    const size_t nc = sim->ncells;
    const size_t slab = (size_t)sim->nx * (size_t)sim->ny;
    const int nz = sim->nz;
    double *b  = sim->pcg_b;
    double *r  = sim->pcg_r;
    double *z  = sim->pcg_z;
    double *p  = sim->pcg_p;
    double *ap = sim->pcg_ap;

    /* Assemble RHS b = -(h/dt)*div_unit(u*) + sum 2*phi_bc (fused with ||b||^2 when n_neumann == 0) */
    RhsCtx rctx;
    rctx.sim = sim;
    rctx.scale = sim->h / dt;
    rctx.u = u_star; rctx.v = v_star; rctx.w = w_star;
    rctx.rhs = b;
    int nt = cfd_parallel_for_k(sim, nz, 8, &rctx, assemble_rhs_chunk_fn);

    double b_sq = 0.0;
    if (L0->n_neumann > 0) {
        subtract_neumann_mean_f64(L0, b);
        subtract_neumann_mean_f64(L0, phi);
        Dot2Ctx dctx;
        dctx.a = b; dctx.b = b; dctx.c = NULL; dctx.slab = slab;
        int ntd = cfd_parallel_for_k(sim, nz, 8, &dctx, dot1_chunk_fn);
        for (int t = 0; t < ntd; ++t) b_sq += dctx.ab_t[t * CFD_CACHE_PAD];
    } else {
        for (int t = 0; t < nt; ++t) b_sq += rctx.bsq_t[t * CFD_CACHE_PAD];
    }

    double b_norm2 = sqrt(b_sq);
    double denom = (b_norm2 > 1e-15) ? b_norm2 : 1e-15;
    double div_scale = dt * sim->inv_h2;
    double rel_tol = sim->params.pcg_rel_tol;
    double div_tol = sim->params.pcg_div_tol;
    int max_iters  = sim->params.pcg_max_iters;

    /* Initial residual r = b - A*phi (strict FP64) fused with (r_sq, r_inf) */
    ApplyResNormCtx rnctx;
    rnctx.sim = sim; rnctx.x = phi; rnctx.b = b; rnctx.r = r;
    nt = cfd_parallel_for_k(sim, nz, 8, &rnctx, apply_a_L0_res_norm_chunk_fn);

    double r_sq = 0.0, r_inf = 0.0;
    for (int t = 0; t < nt; ++t) {
        r_sq += rnctx.sq_t[t * CFD_CACHE_PAD];
        if (rnctx.inf_t[t * CFD_CACHE_PAD] > r_inf) r_inf = rnctx.inf_t[t * CFD_CACHE_PAD];
    }

    double rel_res = sqrt(r_sq) / denom;
    if (rel_res < rel_tol && (div_scale * r_inf) < div_tol) {
        if (out_iters) *out_iters = 0;
        if (out_rel_res) *out_rel_res = rel_res;
        return 0;
    }

    /* Initial preconditioning z = M_fp32^{-1} r (normalized by r_inf to avoid FP32 under/overflow) */
    double r_scale = (r_inf > 1e-300) ? r_inf : 1.0;
    PrepMGCtx prep = { L0, r, 1.0 / r_scale };
    cfd_parallel_for_k(sim, nz, 8, &prep, prep_mg_first_red_chunk_fn);
    mg_vcycle_f32(sim, 0);

    double rz = 0.0;
    PostMGCtx post;
    post.lx = L0->x; post.r = r; post.ap = NULL; post.z = z; post.p = p;
    post.scale = r_scale; post.slab = slab;

    if (L0->n_neumann == 0) {
        nt = cfd_parallel_for_k(sim, nz, 8, &post, post_mg_init_chunk_fn);
        for (int t = 0; t < nt; ++t) rz += post.rz_t[t * CFD_CACHE_PAD];
    } else {
        cfd_parallel_for_k(sim, nz, 8, &post, post_mg_convert_only_chunk_fn);
        subtract_neumann_mean_f64(L0, z);
        memcpy(p, z, nc * sizeof(double));
        Dot2Ctx dctx;
        dctx.a = r; dctx.b = z; dctx.c = NULL; dctx.slab = slab;
        nt = cfd_parallel_for_k(sim, nz, 8, &dctx, dot1_chunk_fn);
        for (int t = 0; t < nt; ++t) rz += dctx.ab_t[t * CFD_CACHE_PAD];
    }

    /* Outer Flexible-CG (Notay FCG(1) / exact Polak-Ribiere) in strict FP64 */
    int iter = 0;
    for (iter = 1; iter <= max_iters; ++iter) {
        ApplyADotCtx adctx;
        adctx.sim = sim; adctx.p = p; adctx.ap = ap;
        nt = cfd_parallel_for_k(sim, nz, 8, &adctx, apply_a_L0_dot_chunk_fn);

        double pap = 0.0;
        for (int t = 0; t < nt; ++t) pap += adctx.pap_t[t * CFD_CACHE_PAD];
        if (pap <= 1e-30) break;

        double alpha = rz / pap;

        AxpyNormCtx actx;
        actx.alpha = alpha; actx.p = p; actx.ap = ap; actx.phi = phi; actx.r = r; actx.slab = slab;
        nt = cfd_parallel_for_k(sim, nz, 8, &actx, axpy_norm_chunk_fn);

        r_sq = 0.0;
        r_inf = 0.0;
        for (int t = 0; t < nt; ++t) {
            r_sq += actx.sq_t[t * CFD_CACHE_PAD];
            if (actx.inf_t[t * CFD_CACHE_PAD] > r_inf) r_inf = actx.inf_t[t * CFD_CACHE_PAD];
        }

        rel_res = sqrt(r_sq) / denom;
        if (rel_res < rel_tol && (div_scale * r_inf) < div_tol) {
            break;
        }

        /* z = M_fp32^{-1} r */
        r_scale = (r_inf > 1e-300) ? r_inf : 1.0;
        prep.inv_scale = 1.0 / r_scale;
        cfd_parallel_for_k(sim, nz, 8, &prep, prep_mg_first_red_chunk_fn);
        mg_vcycle_f32(sim, 0);

        double rz_new = 0.0, zap = 0.0;
        post.ap = ap;
        post.scale = r_scale;
        if (L0->n_neumann == 0) {
            nt = cfd_parallel_for_k(sim, nz, 8, &post, post_mg_dot2_chunk_fn);
            for (int t = 0; t < nt; ++t) {
                rz_new += post.rz_t[t * CFD_CACHE_PAD];
                zap    += post.zap_t[t * CFD_CACHE_PAD];
            }
        } else {
            cfd_parallel_for_k(sim, nz, 8, &post, post_mg_convert_only_chunk_fn);
            subtract_neumann_mean_f64(L0, z);
            Dot2Ctx dctx;
            dctx.a = z; dctx.b = r; dctx.c = ap; dctx.slab = slab;
            nt = cfd_parallel_for_k(sim, nz, 8, &dctx, dot2_chunk_fn);
            for (int t = 0; t < nt; ++t) {
                rz_new += dctx.ab_t[t * CFD_CACHE_PAD];
                zap    += dctx.ac_t[t * CFD_CACHE_PAD];
            }
        }

        if (rz_new <= 1e-300) break;

        /* Notay FCG(1) A-orthogonalization: p_{k+1} = z_{k+1} + beta * p_k, beta = -(z.Ap)/(p.Ap) */
        double beta_cg = -zap / pap;
        PUpdateCtx pctx = { beta_cg, z, p, slab };
        cfd_parallel_for_k(sim, nz, 8, &pctx, p_update_chunk_fn);
        rz = rz_new;
    }

    if (L0->n_neumann > 0) {
        subtract_neumann_mean_f64(L0, phi);
    }

    if (out_iters) *out_iters = (iter > max_iters) ? max_iters : iter;
    if (out_rel_res) *out_rel_res = rel_res;
    return 0;
}

/* ========================================================================= */
/* Exact FP64 Velocity Projection (Unit-Stride, Fused UVW Pass)              */
/* ========================================================================= */

typedef struct ProjCtx {
    const CFD3D *sim;
    double dt_inv_h;
    const double *phi;
    double *u, *v, *w;
    double max_div_t[CFD_MAX_CHUNKS * CFD_CACHE_PAD];
} ProjCtx;

static void proj_uvw_chunk_fn(void *vctx, int k0, int k1, int tid) {
    (void)tid;
    ProjCtx *ctx = (ProjCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny, nz = sim->nz;
    const double c1 = ctx->dt_inv_h;
    const double c2 = 2.0 * c1;
    const double *restrict phi = ctx->phi;
    const uint8_t *restrict kx = sim->kx;
    const uint8_t *restrict ky = sim->ky;
    const uint8_t *restrict kz = sim->kz;
    double *restrict u = ctx->u;
    double *restrict v = ctx->v;
    double *restrict w = ctx->w;
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
            const uint8_t *restrict kxr = kx + row_u;
            double *restrict ur = u + row_u;

            if (kxr[0]) {
                double phi_bc = sim->phi_bc_mx[(size_t)k * (size_t)ny + (size_t)j];
                ur[0] -= c2 * (pr[0] - phi_bc);
            } else {
                ur[0] = 0.0;
            }

            for (int i = 1; i < nx; ++i) {
                double u_new = ur[i] - c1 * (pr[i] - pr[i - 1]);
                ur[i] = kxr[i] ? u_new : 0.0;
            }

            if (kxr[nx] && open_px) {
                double phi_bc = sim->phi_bc_px[(size_t)k * (size_t)ny + (size_t)j];
                ur[nx] -= c2 * (phi_bc - pr[nx - 1]);
            } else if (!kxr[nx]) {
                ur[nx] = 0.0;
            }
        }

        /* 2. v-faces (100% unit-stride: j outer, i inner!) */
        {
            /* j = 0 */
            const size_t row_c0 = idx_c(0, 0, k, nx, ny);
            const size_t row_v0 = idx_v(0, 0, k, nx, ny);
            const double *restrict pr0 = phi + row_c0;
            const double *restrict bc_my = sim->phi_bc_my + (size_t)k * (size_t)nx;
            const uint8_t *restrict kyr0 = ky + row_v0;
            double *restrict vr0 = v + row_v0;
            if (open_my) {
                for (int i = 0; i < nx; ++i) {
                    vr0[i] = kyr0[i] ? (vr0[i] - c2 * (pr0[i] - bc_my[i])) : 0.0;
                }
            } else {
                for (int i = 0; i < nx; ++i) {
                    if (!kyr0[i]) vr0[i] = 0.0;
                }
            }

            /* Interior j = 1..ny-1 */
            for (int j = 1; j < ny; ++j) {
                const size_t row_c  = idx_c(0, j,     k, nx, ny);
                const size_t row_cm = idx_c(0, j - 1, k, nx, ny);
                const size_t row_v  = idx_v(0, j,     k, nx, ny);
                const double *restrict pr  = phi + row_c;
                const double *restrict prm = phi + row_cm;
                const uint8_t *restrict kyr = ky + row_v;
                double *restrict vr = v + row_v;
                for (int i = 0; i < nx; ++i) {
                    double v_new = vr[i] - c1 * (pr[i] - prm[i]);
                    vr[i] = kyr[i] ? v_new : 0.0;
                }
            }

            /* j = ny */
            const size_t row_cn = idx_c(0, ny - 1, k, nx, ny);
            const size_t row_vn = idx_v(0, ny,     k, nx, ny);
            const double *restrict prn = phi + row_cn;
            const double *restrict bc_py = sim->phi_bc_py + (size_t)k * (size_t)nx;
            const uint8_t *restrict kyrn = ky + row_vn;
            double *restrict vrn = v + row_vn;
            if (open_py) {
                for (int i = 0; i < nx; ++i) {
                    vrn[i] = kyrn[i] ? (vrn[i] - c2 * (bc_py[i] - prn[i])) : 0.0;
                }
            } else {
                for (int i = 0; i < nx; ++i) {
                    if (!kyrn[i]) vrn[i] = 0.0;
                }
            }
        }

        /* 3. w-faces for slab k (unit stride in i) */
        if (k == 0) {
            memset(&w[idx_w(0, 0, 0, nx, ny)], 0, (size_t)nx * (size_t)ny * sizeof(double));
        } else {
            for (int j = 0; j < ny; ++j) {
                const size_t row_c  = idx_c(0, j, k,     nx, ny);
                const size_t row_cm = idx_c(0, j, k - 1, nx, ny);
                const size_t row_w  = idx_w(0, j, k,     nx, ny);
                const double *restrict pr  = phi + row_c;
                const double *restrict prm = phi + row_cm;
                const uint8_t *restrict kzr = kz + row_w;
                double *restrict wr = w + row_w;
                for (int i = 0; i < nx; ++i) {
                    double w_new = wr[i] - c1 * (pr[i] - prm[i]);
                    wr[i] = kzr[i] ? w_new : 0.0;
                }
            }
        }
    }

    /* Top sky boundary w-face k = nz handled by last chunk */
    if (k1 == nz) {
        for (int j = 0; j < ny; ++j) {
            const size_t row_cn = idx_c(0, j, nz - 1, nx, ny);
            const size_t row_wn = idx_w(0, j, nz,     nx, ny);
            const double *restrict prn = phi + row_cn;
            const double *restrict bc_pz = sim->phi_bc_pz + (size_t)j * (size_t)nx;
            const uint8_t *restrict kzr = kz + row_wn;
            double *restrict wr = w + row_wn;
            if (open_pz) {
                for (int i = 0; i < nx; ++i) {
                    wr[i] = kzr[i] ? (wr[i] - c2 * (bc_pz[i] - prn[i])) : 0.0;
                }
            } else {
                for (int i = 0; i < nx; ++i) {
                    if (!kzr[i]) wr[i] = 0.0;
                }
            }
        }
    }
}

static void check_div_chunk_fn(void *vctx, int k0, int k1, int tid) {
    ProjCtx *ctx = (ProjCtx *)vctx;
    const CFD3D *sim = ctx->sim;
    const int nx = sim->nx, ny = sim->ny;
    const double inv_h = sim->inv_h;
    const double *restrict u = ctx->u;
    const double *restrict v = ctx->v;
    const double *restrict w = ctx->w;
    const uint8_t *restrict ct = sim->cell_type;
    double md = 0.0;

    for (int k = k0; k < k1; ++k) {
        for (int j = 0; j < ny; ++j) {
            const uint8_t *restrict ctr = ct + idx_c(0, j, k, nx, ny);
            const double *restrict ur   = u  + idx_u(0, j, k, nx, ny);
            const double *restrict v0   = v  + idx_v(0, j,     k, nx, ny);
            const double *restrict v1   = v  + idx_v(0, j + 1, k, nx, ny);
            const double *restrict w0   = w  + idx_w(0, j, k,     nx, ny);
            const double *restrict w1   = w  + idx_w(0, j, k + 1, nx, ny);
            for (int i = 0; i < nx; ++i) {
                if (ctr[i] != CFD_CELL_FLUID) continue;
                double d = fabs(((ur[i + 1] - ur[i]) + (v1[i] - v0[i]) + (w1[i] - w0[i])) * inv_h);
                if (d > md) md = d;
            }
        }
    }
    ctx->max_div_t[tid * CFD_CACHE_PAD] = md;
}

void cfd_project_velocity(CFD3D *sim, double dt, const double *phi,
                          double *u, double *v, double *w, double *out_max_div) {
    const int nz = sim->nz;
    ProjCtx ctx;
    ctx.sim = sim;
    ctx.dt_inv_h = dt * sim->inv_h;
    ctx.phi = phi;
    ctx.u = u; ctx.v = v; ctx.w = w;

    cfd_parallel_for_k(sim, nz, 8, &ctx, proj_uvw_chunk_fn);
    if (out_max_div) {
        int nt = cfd_parallel_for_k(sim, nz, 8, &ctx, check_div_chunk_fn);
        double max_div = 0.0;
        for (int t = 0; t < nt; ++t) {
            if (ctx.max_div_t[t * CFD_CACHE_PAD] > max_div) {
                max_div = ctx.max_div_t[t * CFD_CACHE_PAD];
            }
        }
        *out_max_div = max_div;
    }
}
