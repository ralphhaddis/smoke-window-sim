"""Python ctypes + zero-copy NumPy wrapper for the 3D incompressible Boussinesq CFD engine."""

from __future__ import annotations

import ctypes as ct
from typing import Any, Dict, Optional, Tuple
import numpy as np

from .build import build_library

# Enum constants matching csrc/cfd3d.h
HINGE_LEFT = 0
HINGE_RIGHT = 1
HINGE_TOP = 2

CELL_FLUID = 0
CELL_SOLID = 1

ZONE_INDOOR = 0
ZONE_REVEAL = 1
ZONE_OUTDOOR = 2
ZONE_SOLID = 3

FIELD_SMOKE = 0
FIELD_TEMP = 1
FIELD_SPEED = 2
FIELD_PRESSURE = 3
FIELD_VORTICITY = 4
FIELD_U = 5
FIELD_V = 6
FIELD_W = 7
FIELD_NU_SGS = 8
FIELD_SOLID = 9

SLICE_XZ = 0
SLICE_XY = 1
SLICE_YZ = 2

PTR_U = 0
PTR_V = 1
PTR_W = 2
PTR_PHI = 3
PTR_THETA = 4
PTR_SMOKE = 5
PTR_NU_SGS = 6
PTR_WALL_DIST = 7

MASK_CELL_TYPE = 0
MASK_CELL_ZONE = 1
MASK_KX = 2
MASK_KY = 3
MASK_KZ = 4

FIELD_NAME_TO_ID = {
    "smoke": FIELD_SMOKE,
    "temp": FIELD_TEMP,
    "speed": FIELD_SPEED,
    "pressure": FIELD_PRESSURE,
    "vorticity": FIELD_VORTICITY,
    "u": FIELD_U,
    "v": FIELD_V,
    "w": FIELD_W,
    "nu_sgs": FIELD_NU_SGS,
    "solid": FIELD_SOLID,
}

PLANE_NAME_TO_ID = {
    "xz": SLICE_XZ,
    "xy": SLICE_XY,
    "yz": SLICE_YZ,
}

PTR_NAME_TO_ID = {
    "u": PTR_U,
    "v": PTR_V,
    "w": PTR_W,
    "phi": PTR_PHI,
    "theta": PTR_THETA,
    "smoke": PTR_SMOKE,
    "nu_sgs": PTR_NU_SGS,
    "wall_dist": PTR_WALL_DIST,
}

MASK_NAME_TO_ID = {
    "cell_type": MASK_CELL_TYPE,
    "cell_zone": MASK_CELL_ZONE,
    "kx": MASK_KX,
    "ky": MASK_KY,
    "kz": MASK_KZ,
}

GRID_PRESETS: Dict[str, Dict[str, Any]] = {
    "turbo":   {"nx": 64,  "ny": 48, "nz": 40, "h": 0.10},
    "fast":    {"nx": 80,  "ny": 64, "nz": 56, "h": 0.0825},
    "default": {"nx": 96,  "ny": 64, "nz": 64, "h": 0.06875},
    "high":    {"nx": 128, "ny": 96, "nz": 80, "h": 0.055},
}

SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "cold_day": {
        "temp_in_c": 20.0,
        "temp_out_c": 5.0,
        "win_hinge_mode": HINGE_LEFT,
        "win_angle_deg": 45.0,
        "wind_speed": 0.0,
        "vent_enabled": 0,
        "smoke_enabled": 1,
        "smoke_temp_c": 30.0,
        "smoke_rate": 0.08,
    },
    "quiescent": {
        "temp_in_c": 20.0,
        "temp_out_c": 20.0,
        "win_hinge_mode": HINGE_LEFT,
        "win_angle_deg": 45.0,
        "wind_speed": 0.0,
        "vent_enabled": 0,
        "delta_p_pa": 0.0,
        "smoke_enabled": 0,
        "smoke_rate": 0.0,
    },
    "crosswind": {
        "temp_in_c": 20.0,
        "temp_out_c": 10.0,
        "win_hinge_mode": HINGE_LEFT,
        "win_angle_deg": 45.0,
        "wind_speed": 2.5,
        "wind_dir_deg": 75.0,
        "wind_use_abl": 1,
        "vent_enabled": 0,
        "smoke_enabled": 1,
        "smoke_temp_c": 28.0,
        "smoke_rate": 0.08,
    },
    "pressurized_cross_vent": {
        "temp_in_c": 20.0,
        "temp_out_c": 8.0,
        "win_hinge_mode": HINGE_LEFT,
        "win_angle_deg": 60.0,
        "wind_speed": 0.0,
        "vent_enabled": 1,
        "delta_p_pa": 3.0,
        "smoke_enabled": 1,
        "smoke_temp_c": 28.0,
        "smoke_rate": 0.08,
    },
    "awning_rain": {
        "temp_in_c": 20.0,
        "temp_out_c": 5.0,
        "win_hinge_mode": HINGE_TOP,
        "win_angle_deg": 30.0,
        "wind_speed": 1.0,
        "wind_dir_deg": 45.0,
        "wind_use_abl": 1,
        "vent_enabled": 0,
        "smoke_enabled": 1,
        "smoke_temp_c": 30.0,
        "smoke_rate": 0.08,
    },
    "closed_window": {
        "temp_in_c": 20.0,
        "temp_out_c": 5.0,
        "win_hinge_mode": HINGE_LEFT,
        "win_angle_deg": 0.0,
        "wind_speed": 0.0,
        "vent_enabled": 0,
        "smoke_enabled": 0,
        "smoke_rate": 0.0,
    },
}


class CFDParams(ct.Structure):
    _fields_ = [
        ("nx", ct.c_int),
        ("ny", ct.c_int),
        ("nz", ct.c_int),
        ("h", ct.c_double),
        ("room_lx", ct.c_double),
        ("room_ly", ct.c_double),
        ("room_lz", ct.c_double),
        ("wall_thick", ct.c_double),
        ("win_width", ct.c_double),
        ("win_height", ct.c_double),
        ("win_sill", ct.c_double),
        ("win_y_offset", ct.c_double),
        ("win_hinge_mode", ct.c_int),
        ("win_angle_deg", ct.c_double),
        ("vent_enabled", ct.c_int),
        ("vent_y0", ct.c_double),
        ("vent_y1", ct.c_double),
        ("vent_z0", ct.c_double),
        ("vent_z1", ct.c_double),
        ("delta_p_pa", ct.c_double),
        ("vent_k_loss", ct.c_double),
        ("temp_in_c", ct.c_double),
        ("temp_out_c", ct.c_double),
        ("wind_speed", ct.c_double),
        ("wind_dir_deg", ct.c_double),
        ("wind_use_abl", ct.c_int),
        ("wind_z_ref", ct.c_double),
        ("smoke_enabled", ct.c_int),
        ("smoke_x", ct.c_double),
        ("smoke_y", ct.c_double),
        ("smoke_z", ct.c_double),
        ("smoke_radius", ct.c_double),
        ("smoke_temp_c", ct.c_double),
        ("smoke_rate", ct.c_double),
        ("smoke_alpha_s", ct.c_double),
        ("smoke_jet_speed", ct.c_double),
        ("smoke_jet_azim_deg", ct.c_double),
        ("smoke_jet_elev_deg", ct.c_double),
        ("cfl_max", ct.c_double),
        ("dt_min", ct.c_double),
        ("dt_max", ct.c_double),
        ("cs_smag", ct.c_double),
        ("sponge_sigma", ct.c_double),
        ("pcg_max_iters", ct.c_int),
        ("pcg_rel_tol", ct.c_double),
        ("pcg_div_tol", ct.c_double),
    ]

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name, _ in self._fields_}


class CFDMetrics(ct.Structure):
    _fields_ = [
        ("sim_time", ct.c_double),
        ("dt", ct.c_double),
        ("step_count", ct.c_int),
        ("q_out", ct.c_double),
        ("q_in", ct.c_double),
        ("q_net", ct.c_double),
        ("q_vent", ct.c_double),
        ("ach", ct.c_double),
        ("z_npp", ct.c_double),
        ("smoke_indoor", ct.c_double),
        ("smoke_outdoor", ct.c_double),
        ("smoke_max_indoor", ct.c_double),
        ("temp_mean_in_c", ct.c_double),
        ("kinetic_energy", ct.c_double),
        ("max_speed", ct.c_double),
        ("cfl", ct.c_double),
        ("max_div", ct.c_double),
        ("pcg_iters", ct.c_int),
        ("pcg_rel_res", ct.c_double),
        ("step_ms", ct.c_double),
    ]

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name, _ in self._fields_}


class CFDGeomInfo(ct.Structure):
    _fields_ = [
        ("nx", ct.c_int),
        ("ny", ct.c_int),
        ("nz", ct.c_int),
        ("h", ct.c_double),
        ("lx", ct.c_double),
        ("ly", ct.c_double),
        ("lz", ct.c_double),
        ("ix_wall", ct.c_int),
        ("ix_ext", ct.c_int),
        ("iy_min", ct.c_int),
        ("iy_max", ct.c_int),
        ("iz_ceil", ct.c_int),
        ("j_w0", ct.c_int),
        ("j_w1", ct.c_int),
        ("k_w0", ct.c_int),
        ("k_w1", ct.c_int),
        ("j_v0", ct.c_int),
        ("j_v1", ct.c_int),
        ("k_v0", ct.c_int),
        ("k_v1", ct.c_int),
        ("room_x0", ct.c_double),
        ("room_x1", ct.c_double),
        ("wall_x1", ct.c_double),
        ("room_y0", ct.c_double),
        ("room_y1", ct.c_double),
        ("room_z0", ct.c_double),
        ("room_z1", ct.c_double),
        ("win_y0", ct.c_double),
        ("win_y1", ct.c_double),
        ("win_z0", ct.c_double),
        ("win_z1", ct.c_double),
        ("vent_y0", ct.c_double),
        ("vent_y1", ct.c_double),
        ("vent_z0", ct.c_double),
        ("vent_z1", ct.c_double),
        ("room_volume", ct.c_double),
        ("sash_corners", (ct.c_double * 3) * 4),
        ("sash_normal", ct.c_double * 3),
        ("n_fluid_cells", ct.c_int),
        ("n_indoor_cells", ct.c_int),
        ("n_sash_blocked_faces", ct.c_int),
        ("n_mg_levels", ct.c_int),
    ]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, _ in self._fields_:
            val = getattr(self, name)
            if name == "sash_corners":
                out[name] = [[val[i][j] for j in range(3)] for i in range(4)]
            elif name == "sash_normal":
                out[name] = [val[j] for j in range(3)]
            else:
                out[name] = val
        return out


_LIB: Optional[ct.CDLL] = None


def load_lib(force_rebuild: bool = False) -> ct.CDLL:
    global _LIB
    if _LIB is not None and not force_rebuild:
        return _LIB

    lib_path = build_library(force=force_rebuild, verbose=False)
    lib = ct.CDLL(str(lib_path))

    lib.cfd_default_params.argtypes = [ct.POINTER(CFDParams)]
    lib.cfd_default_params.restype = None

    lib.cfd_create.argtypes = [ct.POINTER(CFDParams)]
    lib.cfd_create.restype = ct.c_void_p

    lib.cfd_destroy.argtypes = [ct.c_void_p]
    lib.cfd_destroy.restype = None

    lib.cfd_reset.argtypes = [ct.c_void_p]
    lib.cfd_reset.restype = None

    lib.cfd_update_params.argtypes = [ct.c_void_p, ct.POINTER(CFDParams)]
    lib.cfd_update_params.restype = ct.c_int

    lib.cfd_step.argtypes = [ct.c_void_p, ct.POINTER(CFDMetrics)]
    lib.cfd_step.restype = ct.c_int

    lib.cfd_step_n.argtypes = [ct.c_void_p, ct.c_int, ct.POINTER(CFDMetrics)]
    lib.cfd_step_n.restype = ct.c_int

    lib.cfd_get_metrics.argtypes = [ct.c_void_p, ct.POINTER(CFDMetrics)]
    lib.cfd_get_metrics.restype = None

    lib.cfd_get_geom_info.argtypes = [ct.c_void_p, ct.POINTER(CFDGeomInfo)]
    lib.cfd_get_geom_info.restype = None

    lib.cfd_get_field_ptr.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_int),
    ]
    lib.cfd_get_field_ptr.restype = ct.POINTER(ct.c_double)

    lib.cfd_get_mask_ptr.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_int),
    ]
    lib.cfd_get_mask_ptr.restype = ct.POINTER(ct.c_uint8)

    lib.cfd_extract_slice_f32.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.POINTER(ct.c_float),
    ]
    lib.cfd_extract_slice_f32.restype = ct.c_int

    lib.cfd_extract_vec_slice_f32.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.c_int,
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float),
    ]
    lib.cfd_extract_vec_slice_f32.restype = ct.c_int

    lib.cfd_pack_volume_u8.argtypes = [
        ct.c_void_p,
        ct.c_int,
        ct.c_double,
        ct.c_double,
        ct.POINTER(ct.c_uint8),
    ]
    lib.cfd_pack_volume_u8.restype = ct.c_int

    _LIB = lib
    return lib


def default_params(
    grid: str = "default",
    preset: Optional[str] = None,
    **overrides: Any,
) -> CFDParams:
    lib = load_lib()
    p = CFDParams()
    lib.cfd_default_params(ct.byref(p))

    if grid not in GRID_PRESETS:
        raise ValueError(f"Unknown grid preset '{grid}'. Choices: {list(GRID_PRESETS)}")
    for k, v in GRID_PRESETS[grid].items():
        setattr(p, k, v)

    if preset is not None:
        if preset not in SCENARIO_PRESETS:
            raise ValueError(f"Unknown scenario preset '{preset}'. Choices: {list(SCENARIO_PRESETS)}")
        for k, v in SCENARIO_PRESETS[preset].items():
            setattr(p, k, v)

    valid_fields = {name for name, _ in CFDParams._fields_}
    for k, v in overrides.items():
        if k not in valid_fields:
            raise KeyError(f"Unknown CFDParams field '{k}'")
        setattr(p, k, v)
    return p


class CFD3DSolver:
    """High-level Python interface to the C99 3D Boussinesq CFD engine."""

    def __init__(
        self,
        params: Optional[CFDParams] = None,
        grid: str = "default",
        preset: Optional[str] = None,
        **overrides: Any,
    ) -> None:
        self._lib = load_lib()
        if params is None:
            self._params = default_params(grid=grid, preset=preset, **overrides)
        else:
            self._params = CFDParams()
            ct.memmove(ct.byref(self._params), ct.byref(params), ct.sizeof(CFDParams))
            valid_fields = {f[0] for f in CFDParams._fields_}
            for k, v in overrides.items():
                if k not in valid_fields:
                    raise KeyError(f"Unknown CFDParams field: {k}")
                setattr(self._params, k, v)

        self._ptr = self._lib.cfd_create(ct.byref(self._params))
        if not self._ptr:
            raise RuntimeError("cfd_create failed (invalid parameters or out of memory)")

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            self._lib.cfd_destroy(self._ptr)
            self._ptr = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "CFD3DSolver":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @property
    def params(self) -> CFDParams:
        return self._params

    def reset(self) -> None:
        self._lib.cfd_reset(self._ptr)

    def update_params(self, **kwargs: Any) -> None:
        valid_fields = {name for name, _ in CFDParams._fields_}
        for k, v in kwargs.items():
            if k not in valid_fields:
                raise KeyError(f"Unknown CFDParams field '{k}'")
            setattr(self._params, k, v)
        rc = self._lib.cfd_update_params(self._ptr, ct.byref(self._params))
        if rc != 0:
            raise RuntimeError("cfd_update_params failed (grid dimensions cannot change live)")

    def step(self) -> Dict[str, Any]:
        m = CFDMetrics()
        rc = self._lib.cfd_step(self._ptr, ct.byref(m))
        if rc != 0:
            raise RuntimeError("cfd_step failed")
        return m.to_dict()

    def step_n(self, n_steps: int) -> Dict[str, Any]:
        m = CFDMetrics()
        rc = self._lib.cfd_step_n(self._ptr, int(n_steps), ct.byref(m))
        if rc != 0:
            raise RuntimeError("cfd_step_n failed")
        return m.to_dict()

    def metrics(self) -> Dict[str, Any]:
        m = CFDMetrics()
        self._lib.cfd_get_metrics(self._ptr, ct.byref(m))
        return m.to_dict()

    def geom_info(self) -> Dict[str, Any]:
        g = CFDGeomInfo()
        self._lib.cfd_get_geom_info(self._ptr, ct.byref(g))
        return g.to_dict()

    def field_view(self, name: str) -> np.ndarray:
        """Return a zero-copy float64 NumPy view of shape (nz, ny, nx) (or staggered)."""
        if name not in PTR_NAME_TO_ID:
            raise KeyError(f"Unknown field '{name}'. Choices: {list(PTR_NAME_TO_ID)}")
        nz = ct.c_int()
        ny = ct.c_int()
        nx = ct.c_int()
        ptr = self._lib.cfd_get_field_ptr(
            self._ptr, PTR_NAME_TO_ID[name], ct.byref(nz), ct.byref(ny), ct.byref(nx)
        )
        if not ptr:
            raise RuntimeError(f"cfd_get_field_ptr failed for '{name}'")
        arr = np.ctypeslib.as_array(ptr, shape=(nz.value, ny.value, nx.value))
        return arr

    def mask_view(self, name: str) -> np.ndarray:
        """Return a zero-copy uint8 NumPy view of cell/face geometry masks."""
        if name not in MASK_NAME_TO_ID:
            raise KeyError(f"Unknown mask '{name}'. Choices: {list(MASK_NAME_TO_ID)}")
        nz = ct.c_int()
        ny = ct.c_int()
        nx = ct.c_int()
        ptr = self._lib.cfd_get_mask_ptr(
            self._ptr, MASK_NAME_TO_ID[name], ct.byref(nz), ct.byref(ny), ct.byref(nx)
        )
        if not ptr:
            raise RuntimeError(f"cfd_get_mask_ptr failed for '{name}'")
        return np.ctypeslib.as_array(ptr, shape=(nz.value, ny.value, nx.value))

    def extract_slice(
        self,
        field: str = "smoke",
        plane: str = "xz",
        index: Optional[int] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        fid = FIELD_NAME_TO_ID[field] if isinstance(field, str) else int(field)
        pid = PLANE_NAME_TO_ID[plane.lower()] if isinstance(plane, str) else int(plane)

        if pid == SLICE_XZ:
            shape = (self._params.nz, self._params.nx)
            if index is None:
                g = self.geom_info()
                index = (g["j_w0"] + g["j_w1"]) // 2
        elif pid == SLICE_XY:
            shape = (self._params.ny, self._params.nx)
            if index is None:
                g = self.geom_info()
                index = (g["k_w0"] + g["k_w1"]) // 2
        elif pid == SLICE_YZ:
            shape = (self._params.nz, self._params.ny)
            if index is None:
                g = self.geom_info()
                index = g["ix_wall"]
        else:
            raise ValueError(f"Invalid slice plane '{plane}'")

        if out is None:
            buf = np.empty(shape, dtype=np.float32)
        else:
            if out.shape != shape or out.dtype != np.float32 or not out.flags["C_CONTIGUOUS"]:
                raise ValueError(f"out must be C-contiguous float32 array of shape {shape}")
            buf = out

        rc = self._lib.cfd_extract_slice_f32(
            self._ptr, fid, pid, int(index), buf.ctypes.data_as(ct.POINTER(ct.c_float))
        )
        if rc != 0:
            raise RuntimeError("cfd_extract_slice_f32 failed")
        return buf

    def extract_vec_slice(
        self,
        plane: str = "xz",
        index: Optional[int] = None,
        out: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pid = PLANE_NAME_TO_ID[plane.lower()] if isinstance(plane, str) else int(plane)

        if pid == SLICE_XZ:
            shape = (self._params.nz, self._params.nx)
            if index is None:
                g = self.geom_info()
                index = (g["j_w0"] + g["j_w1"]) // 2
        elif pid == SLICE_XY:
            shape = (self._params.ny, self._params.nx)
            if index is None:
                g = self.geom_info()
                index = (g["k_w0"] + g["k_w1"]) // 2
        elif pid == SLICE_YZ:
            shape = (self._params.nz, self._params.ny)
            if index is None:
                g = self.geom_info()
                index = g["ix_wall"]
        else:
            raise ValueError(f"Invalid slice plane '{plane}'")

        if out is None:
            u1 = np.empty(shape, dtype=np.float32)
            u2 = np.empty(shape, dtype=np.float32)
        else:
            u1, u2 = out
            if (
                u1.shape != shape
                or u2.shape != shape
                or u1.dtype != np.float32
                or u2.dtype != np.float32
                or not u1.flags["C_CONTIGUOUS"]
                or not u2.flags["C_CONTIGUOUS"]
            ):
                raise ValueError(f"out arrays must be C-contiguous float32 arrays of shape {shape}")

        rc = self._lib.cfd_extract_vec_slice_f32(
            self._ptr,
            pid,
            int(index),
            u1.ctypes.data_as(ct.POINTER(ct.c_float)),
            u2.ctypes.data_as(ct.POINTER(ct.c_float)),
        )
        if rc != 0:
            raise RuntimeError("cfd_extract_vec_slice_f32 failed")
        return u1, u2

    def pack_volume_u8(
        self,
        field: str = "smoke",
        vmin: float = 0.0,
        vmax: float = 1.0,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        fid = FIELD_NAME_TO_ID[field] if isinstance(field, str) else int(field)
        shape = (self._params.nz, self._params.ny, self._params.nx)
        if out is None:
            buf = np.empty(shape, dtype=np.uint8)
        else:
            if out.shape != shape or out.dtype != np.uint8 or not out.flags["C_CONTIGUOUS"]:
                raise ValueError(f"out must be C-contiguous uint8 array of shape {shape}")
            buf = out
        rc = self._lib.cfd_pack_volume_u8(
            self._ptr,
            fid,
            float(vmin),
            float(vmax),
            buf.ctypes.data_as(ct.POINTER(ct.c_uint8)),
        )
        if rc != 0:
            raise RuntimeError("cfd_pack_volume_u8 failed")
        return buf


def _cli_smoke_test() -> None:
    print("=== 3D Boussinesq CFD Engine CLI Smoke Test ===")
    with CFD3DSolver(grid="default", preset="quiescent") as sim:
        m = sim.step_n(10)
        print(
            f"[Quiescent 10 steps] max_speed={m['max_speed']:.3e} m/s, "
            f"max_div={m['max_div']:.3e} 1/s, pcg_iters={m['pcg_iters']}, "
            f"step_ms={m['step_ms']:.2f} ms"
        )

    with CFD3DSolver(grid="default", preset="cold_day") as sim:
        g = sim.geom_info()
        print(
            f"[Cold Day Geometry] grid={g['nx']}x{g['ny']}x{g['nz']} h={g['h']}m, "
            f"room={g['room_x1']:.2f}x{g['room_y1']-g['room_y0']:.2f}x{g['room_z1']:.2f}m, "
            f"sash_blocked_faces={g['n_sash_blocked_faces']}, mg_levels={g['n_mg_levels']}"
        )
        ms_list = []
        iters_list = []
        for _ in range(40):
            m = sim.step()
            ms_list.append(m["step_ms"])
            iters_list.append(m["pcg_iters"])
        print(
            f"[Cold Day 40 steps] t={m['sim_time']:.2f}s, dt={m['dt']:.4f}s, "
            f"Q_out={m['q_out']:.4f} m3/s, Q_in={m['q_in']:.4f} m3/s, "
            f"z_NPP={m['z_npp']:.3f}m, ACH={m['ach']:.2f}/h, "
            f"max_div={m['max_div']:.2e}, avg_pcg={np.mean(iters_list):.1f}, "
            f"avg_ms={np.mean(ms_list):.2f} ms/step"
        )


if __name__ == "__main__":
    _cli_smoke_test()
