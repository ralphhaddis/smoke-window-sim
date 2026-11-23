#ifndef CFD3D_H
#define CFD3D_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Window hinge modes */
enum {
    CFD_HINGE_LEFT  = 0, /* Side-hung at y = win_y0, swings outward into +x */
    CFD_HINGE_RIGHT = 1, /* Side-hung at y = win_y1, swings outward into +x */
    CFD_HINGE_TOP   = 2  /* Top-hung awning at z = win_z1, swings outward into +x */
};

/* Cell types */
enum {
    CFD_CELL_FLUID = 0,
    CFD_CELL_SOLID = 1
};

/* Cell zones */
enum {
    CFD_ZONE_INDOOR  = 0,
    CFD_ZONE_REVEAL  = 1,
    CFD_ZONE_OUTDOOR = 2,
    CFD_ZONE_SOLID   = 3
};

/* Field IDs for slice extraction and volume packing */
enum {
    CFD_FIELD_SMOKE     = 0,
    CFD_FIELD_TEMP      = 1, /* Degrees Celsius: temp_out_c + theta */
    CFD_FIELD_SPEED     = 2, /* |u| in m/s */
    CFD_FIELD_PRESSURE  = 3, /* Perturbation pressure p' = rho0 * phi in Pa */
    CFD_FIELD_VORTICITY = 4, /* |curl(u)| in 1/s */
    CFD_FIELD_U         = 5, /* Cell-centered u in m/s */
    CFD_FIELD_V         = 6, /* Cell-centered v in m/s */
    CFD_FIELD_W         = 7, /* Cell-centered w in m/s */
    CFD_FIELD_NU_SGS    = 8, /* Subgrid eddy viscosity in m^2/s */
    CFD_FIELD_SOLID     = 9  /* 0.0 fluid, 1.0 solid */
};

/* Slice planes */
enum {
    CFD_SLICE_XZ = 0, /* Fixed j, shape (nz, nx) */
    CFD_SLICE_XY = 1, /* Fixed k, shape (ny, nx) */
    CFD_SLICE_YZ = 2  /* Fixed i, shape (nz, ny) */
};

/* Raw FP64 field pointer IDs */
enum {
    CFD_PTR_U         = 0,
    CFD_PTR_V         = 1,
    CFD_PTR_W         = 2,
    CFD_PTR_PHI       = 3,
    CFD_PTR_THETA     = 4,
    CFD_PTR_SMOKE     = 5,
    CFD_PTR_NU_SGS    = 6,
    CFD_PTR_WALL_DIST = 7
};

/* Raw uint8 mask pointer IDs */
enum {
    CFD_MASK_CELL_TYPE = 0,
    CFD_MASK_CELL_ZONE = 1,
    CFD_MASK_KX        = 2,
    CFD_MASK_KY        = 3,
    CFD_MASK_KZ        = 4
};

typedef struct CFDParams {
    /* Grid dimensions (cannot change via cfd_update_params; requires cfd_create) */
    int nx;
    int ny;
    int nz;
    double h; /* Uniform cell size [m] */

    /* Room & wall geometry [m] */
    double room_lx;     /* Indoor length along x (default 3.60) */
    double room_ly;     /* Indoor width along y (default 3.00) */
    double room_lz;     /* Indoor ceiling height along z (default 2.50) */
    double wall_thick;  /* Brick facade wall reveal thickness (default 0.25) */

    /* UK casement window geometry [m, deg] */
    double win_width;      /* Opening width along y (default 0.90) */
    double win_height;     /* Opening height along z (default 1.20) */
    double win_sill;       /* Sill height above floor (default 0.90) */
    double win_y_offset;   /* Offset from room y-center (default 0.0) */
    int    win_hinge_mode; /* CFD_HINGE_LEFT / RIGHT / TOP */
    double win_angle_deg;  /* Opening angle in [0, 90] degrees */

    /* Optional rear door / vent on x = 0 indoor wall */
    int    vent_enabled;   /* 0 = closed, 1 = open */
    double vent_y0;        /* Room-local y start [m] (default 1.05) */
    double vent_y1;        /* Room-local y end [m] (default 1.95) */
    double vent_z0;        /* z start [m] (default 0.0) */
    double vent_z1;        /* z end [m] (default 2.00) */
    double delta_p_pa;     /* Hallway-to-outdoor gauge pressure Delta P [Pa] */
    double vent_k_loss;    /* Minor loss coefficient K_loss (default 0.5) */

    /* Thermal & ambient conditions */
    double temp_in_c;      /* Initial indoor & hallway temperature [deg C] */
    double temp_out_c;     /* Outdoor reference temperature [deg C] */
    double wind_speed;     /* Ambient wind speed at z_ref [m/s] */
    double wind_dir_deg;   /* 0 = blowing -x into window, 90 = +y, 180 = +x, 270 = -y */
    int    wind_use_abl;   /* 0 = uniform wind, 1 = suburban log-law ABL profile */
    double wind_z_ref;     /* Reference height for wind_speed [m] (default 2.0) */

    /* Smoke emitter (room-local coordinates: x in [0, room_lx], y in [0, room_ly], z in [0, room_lz]) */
    int    smoke_enabled;      /* 0 = off, 1 = on */
    double smoke_x;            /* [m] */
    double smoke_y;            /* [m] */
    double smoke_z;            /* [m] */
    double smoke_radius;       /* Gaussian radius [m] (default 0.18) */
    double smoke_temp_c;       /* Source temperature [deg C] */
    double smoke_rate;         /* Volumetric source rate [m^3/s] */
    double smoke_alpha_s;      /* Excess density coefficient alpha_s (default 0.0) */
    double smoke_jet_speed;    /* Optional jet speed [m/s] (default 0.0) */
    double smoke_jet_azim_deg; /* Azimuth in xy plane (0 = +x toward window, 90 = +y) */
    double smoke_jet_elev_deg; /* Elevation angle from horizontal [-90, 90] */

    /* Numerics & LES parameters */
    double cfl_max;        /* Max 3D advection CFL (default 0.45) */
    double dt_min;         /* Min dt [s] (default 1e-4) */
    double dt_max;         /* Max dt [s] (default 0.05) */
    double cs_smag;        /* Smagorinsky constant Cs (default 0.17) */
    double sponge_sigma;   /* Max Rayleigh sponge rate [1/s] (default 1.5) */
    int    pcg_max_iters;  /* Max MG-PCG iterations (default 50) */
    double pcg_rel_tol;    /* Relative L2 residual tolerance (default 1e-6) */
    double pcg_div_tol;    /* Max post-projection divergence tol [1/s] (default 1e-5) */
} CFDParams;

typedef struct CFDMetrics {
    double sim_time;
    double dt;
    int    step_count;

    /* Window & vent exchange diagnostics */
    double q_out;            /* Window outflow rate [m^3/s] */
    double q_in;             /* Window inflow rate [m^3/s] */
    double q_net;            /* q_out - q_in [m^3/s] */
    double q_vent;           /* Rear vent net inflow rate [m^3/s] */
    double ach;              /* Air changes per hour [1/h] */
    double z_npp;            /* Neutral pressure plane height [m], or -1.0 if none */

    /* Smoke & thermal diagnostics */
    double smoke_indoor;     /* Integrated smoke mass indoors [m^3] */
    double smoke_outdoor;    /* Integrated smoke mass outdoors [m^3] */
    double smoke_max_indoor; /* Peak indoor smoke concentration */
    double temp_mean_in_c;   /* Mean indoor temperature [deg C] */
    double kinetic_energy;   /* Total fluid kinetic energy [J] */

    /* Numerical & performance diagnostics */
    double max_speed;        /* Max fluid speed [m/s] */
    double cfl;              /* Achieved 3D CFL number */
    double max_div;          /* Max |div(u)| over fluid cells [1/s] */
    int    pcg_iters;        /* MG-PCG iterations taken */
    double pcg_rel_res;      /* Final relative residual ||r||_2 / max(||b||_2, 1e-15) */
    double step_ms;          /* Wall time of step [ms] */
} CFDMetrics;

typedef struct CFDGeomInfo {
    int nx, ny, nz;
    double h;
    double lx, ly, lz;

    /* Snapped grid indices */
    int ix_wall, ix_ext;
    int iy_min, iy_max;
    int iz_ceil;
    int j_w0, j_w1, k_w0, k_w1;
    int j_v0, j_v1, k_v0, k_v1;

    /* Physical bounds [m] */
    double room_x0, room_x1;
    double wall_x1;
    double room_y0, room_y1;
    double room_z0, room_z1;
    double win_y0, win_y1, win_z0, win_z1;
    double vent_y0, vent_y1, vent_z0, vent_z1;
    double room_volume;

    /* 4 corners of angled sash in world coordinates (p00, p10, p11, p01) */
    double sash_corners[4][3];
    double sash_normal[3];

    /* Cell counts */
    int n_fluid_cells;
    int n_indoor_cells;
    int n_sash_blocked_faces;
    int n_mg_levels;
} CFDGeomInfo;

typedef struct CFD3D CFD3D;

void     cfd_default_params(CFDParams *p);
CFD3D   *cfd_create(const CFDParams *p);
void     cfd_destroy(CFD3D *sim);
void     cfd_reset(CFD3D *sim);
int      cfd_update_params(CFD3D *sim, const CFDParams *p);
int      cfd_step(CFD3D *sim, CFDMetrics *out_metrics);
int      cfd_step_n(CFD3D *sim, int n_steps, CFDMetrics *out_metrics);
void     cfd_get_metrics(const CFD3D *sim, CFDMetrics *out_metrics);
void     cfd_get_geom_info(const CFD3D *sim, CFDGeomInfo *out_info);
double  *cfd_get_field_ptr(CFD3D *sim, int ptr_id, int *out_nz, int *out_ny, int *out_nx);
uint8_t *cfd_get_mask_ptr(CFD3D *sim, int mask_id, int *out_nz, int *out_ny, int *out_nx);
int      cfd_extract_slice_f32(const CFD3D *sim, int field_id, int plane, int index, float *out_buf);
int      cfd_extract_vec_slice_f32(const CFD3D *sim, int plane, int index, float *out_u1, float *out_u2);
int      cfd_pack_volume_u8(const CFD3D *sim, int field_id, double vmin, double vmax, uint8_t *out_u8);

#ifdef __cplusplus
}
#endif

#endif /* CFD3D_H */
