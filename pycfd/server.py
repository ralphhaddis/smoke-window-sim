"""Zero-dependency Python stdlib HTTP server & binary frame streamer for the 3D CFD engine.

Usage:
    python3 -m pycfd.server --port 8080 --grid fast
"""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import mimetypes
import struct
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

from .solver import (
    CFD3DSolver,
    CFDParams,
    FIELD_NAME_TO_ID,
    GRID_PRESETS,
    HINGE_LEFT,
    HINGE_RIGHT,
    HINGE_TOP,
    SCENARIO_PRESETS,
    default_params,
)

# Project root & static web directory
ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT_DIR / "web"

# Extended scenario presets (all reset key physical controls cleanly)
_BASE_PRESET_RESET: Dict[str, Any] = {
    "room_lx": 3.60,
    "room_ly": 3.00,
    "room_lz": 2.50,
    "wall_thick": 0.25,
    "win_width": 0.90,
    "win_height": 1.20,
    "win_sill": 0.90,
    "win_y_offset": 0.0,
    "win_hinge_mode": HINGE_LEFT,
    "win_angle_deg": 45.0,
    "vent_enabled": 0,
    "delta_p_pa": 0.0,
    "temp_in_c": 20.0,
    "temp_out_c": 5.0,
    "wind_speed": 0.0,
    "wind_dir_deg": 0.0,
    "wind_use_abl": 0,
    "smoke_enabled": 1,
    "smoke_x": 1.80,
    "smoke_y": 1.50,
    "smoke_z": 1.10,
    "smoke_radius": 0.18,
    "smoke_temp_c": 30.0,
    "smoke_rate": 0.08,
    "smoke_alpha_s": 0.0,
    "smoke_jet_speed": 0.0,
    "smoke_jet_azim_deg": 0.0,
    "smoke_jet_elev_deg": 0.0,
}

WEB_SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {}
for _name, _cfg in SCENARIO_PRESETS.items():
    WEB_SCENARIO_PRESETS[_name] = {**_BASE_PRESET_RESET, **_cfg}

# Additional requested scientific presets
WEB_SCENARIO_PRESETS["hot_summer"] = {
    **_BASE_PRESET_RESET,
    "temp_in_c": 20.0,
    "temp_out_c": 33.0,
    "win_hinge_mode": HINGE_LEFT,
    "win_angle_deg": 45.0,
    "wind_speed": 0.0,
    "vent_enabled": 0,
    "smoke_enabled": 1,
    "smoke_x": 2.20,
    "smoke_y": 1.50,
    "smoke_z": 1.40,
    "smoke_temp_c": 24.0,
    "smoke_rate": 0.08,
}

WEB_SCENARIO_PRESETS["heavy_cold_smoke"] = {
    **_BASE_PRESET_RESET,
    "temp_in_c": 20.0,
    "temp_out_c": 12.0,
    "win_hinge_mode": HINGE_LEFT,
    "win_angle_deg": 45.0,
    "wind_speed": 0.0,
    "vent_enabled": 0,
    "smoke_enabled": 1,
    "smoke_x": 2.00,
    "smoke_y": 1.50,
    "smoke_z": 1.65,
    "smoke_temp_c": 16.0,
    "smoke_alpha_s": 0.12,
    "smoke_rate": 0.08,
}

PRESET_ORDER: List[str] = [
    "cold_day",
    "hot_summer",
    "crosswind",
    "pressurized_cross_vent",
    "awning_rain",
    "heavy_cold_smoke",
    "quiescent",
    "closed_window",
]

PRESET_LABELS: Dict[str, str] = {
    "cold_day": "Cold UK Day (Stack Flow)",
    "hot_summer": "Hot Summer Day (Inverted Buoyancy)",
    "crosswind": "Oblique Crosswind (ABL)",
    "pressurized_cross_vent": "Pressurized Cross-Vent (+3 Pa)",
    "awning_rain": "Top-Hung Awning + Breeze",
    "heavy_cold_smoke": "Dense Sinking Smoke (α_s = 0.12)",
    "quiescent": "Quiescent Baseline (No Flow)",
    "closed_window": "Closed Window (0°)",
}

# Field type map for safe JSON coercion
_PARAM_TYPES: Dict[str, type] = {}
for _fname, _ctype in CFDParams._fields_:
    if _ctype is ct.c_int:
        _PARAM_TYPES[_fname] = int
    else:
        _PARAM_TYPES[_fname] = float

_NON_LIVE_FIELDS = {"nx", "ny", "nz", "h"}
_GEOM_AFFECTING_FIELDS = {
    "room_lx",
    "room_ly",
    "room_lz",
    "wall_thick",
    "win_width",
    "win_height",
    "win_sill",
    "win_y_offset",
    "win_hinge_mode",
    "win_angle_deg",
    "vent_enabled",
    "vent_y0",
    "vent_y1",
    "vent_z0",
    "vent_z1",
}
_LE_F4 = np.dtype("<f4")

# Physical fields supported by the web UI
SLICE_FIELDS = {"smoke", "temp", "speed", "pressure", "vorticity", "nu_sgs", "u", "v", "w"}
VOL_FIELDS = {"smoke", "temp", "speed", "vorticity"}


def _bytes_view(arr: np.ndarray, expect_le_f4: bool = False) -> memoryview:
    if expect_le_f4 and (arr.dtype != _LE_F4 or not arr.flags["C_CONTIGUOUS"]):
        arr = np.ascontiguousarray(arr, dtype=_LE_F4)
    return memoryview(arr).cast("B")


def _enrich_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """Add derived display metrics (indoor smoke fraction, L/s flows)."""
    s_in = float(m.get("smoke_indoor", 0.0))
    s_out = float(m.get("smoke_outdoor", 0.0))
    s_tot = s_in + s_out
    out = dict(m)
    out["smoke_indoor_frac"] = (s_in / s_tot) if s_tot > 1e-9 else 0.0
    out["q_out_ls"] = float(m.get("q_out", 0.0)) * 1000.0
    out["q_in_ls"] = float(m.get("q_in", 0.0)) * 1000.0
    out["q_vent_ls"] = float(m.get("q_vent", 0.0)) * 1000.0
    return out


def _fluid_minmax(arr: np.ndarray, fluid_mask: np.ndarray, has_fluid: bool) -> Tuple[float, float]:
    if not has_fluid:
        return 0.0, 1.0
    fluid = arr[fluid_mask]
    vmin = float(np.min(fluid))
    vmax = float(np.max(fluid))
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    return vmin, vmax


class SimulationManager:
    """Thread-safe controller wrapping CFD3DSolver with a background stepper thread."""

    def __init__(
        self,
        grid: str = "fast",
        preset: str = "cold_day",
        auto_start: bool = True,
        idle_timeout_s: float = 12.0,
    ) -> None:
        if grid not in GRID_PRESETS:
            raise ValueError(f"Unknown grid '{grid}'. Choices: {list(GRID_PRESETS)}")
        if preset not in WEB_SCENARIO_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choices: {list(WEB_SCENARIO_PRESETS)}")

        # Lock ordering: always acquire _frame_lock before _lock if both are needed
        self._frame_lock = threading.Lock()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()

        self._grid_name = grid
        self._preset_name = preset
        self._running = bool(auto_start)
        self._time_scale: float = 1.0  # 0.0 means uncapped Max speed
        self._achieved_rt: float = 0.0
        self._history: Deque[Dict[str, float]] = deque(maxlen=240)
        self._last_hist_time = -1.0

        # Version tracking & idle-pause watchdog
        self._state_version: int = 1
        self._geom_version: int = 1
        self._step_count: int = 0
        self._idle_timeout_s: float = max(0.2, float(idle_timeout_s))
        self._last_client_seen: float = time.monotonic()
        self._idle_paused: bool = False

        # Cached geometry & reusable zero-allocation frame buffers
        self._geom_cache: Optional[Dict[str, Any]] = None
        self._solid_cache: Dict[Tuple[int, str, int], Tuple[np.ndarray, bool, bytes]] = {}

        self._solver = self._make_solver(grid, WEB_SCENARIO_PRESETS[preset])
        self._alloc_frame_buffers()

        # Record initial telemetry sample
        init_m = _enrich_metrics(self._solver.metrics())
        self._step_count = int(init_m["step_count"])
        self._append_history(init_m, force=True)

        self._thread = threading.Thread(target=self._worker_loop, name="cfd3d-worker", daemon=True)
        self._thread.start()

    @property
    def idle_paused(self) -> bool:
        return self._idle_paused

    @property
    def state_version(self) -> int:
        return self._state_version

    def _alloc_frame_buffers(self) -> None:
        nx = self._solver.params.nx
        ny = self._solver.params.ny
        nz = self._solver.params.nz
        self._vol_buf = np.empty((nz, ny, nx), dtype=np.uint8)
        self._xz_val = np.empty((nz, nx), dtype=np.float32)
        self._xy_val = np.empty((ny, nx), dtype=np.float32)
        self._yz_val = np.empty((nz, ny), dtype=np.float32)
        self._xz_u1 = np.empty((nz, nx), dtype=np.float32)
        self._xz_u2 = np.empty((nz, nx), dtype=np.float32)
        self._xy_u1 = np.empty((ny, nx), dtype=np.float32)
        self._xy_u2 = np.empty((ny, nx), dtype=np.float32)
        self._yz_u1 = np.empty((nz, ny), dtype=np.float32)
        self._yz_u2 = np.empty((nz, ny), dtype=np.float32)
        self._geom_cache = None
        self._solid_cache.clear()

    def _get_geom_unlocked(self) -> Dict[str, Any]:
        if self._geom_cache is None:
            self._geom_cache = self._solver.geom_info()
        return self._geom_cache

    def _get_solid_slice_unlocked(self, plane: str, index: int) -> Tuple[np.ndarray, bool, bytes]:
        key = (self._geom_version, plane, index)
        cached = self._solid_cache.get(key)
        if cached is not None:
            return cached

        cell_type = self._solver.mask_view("cell_type")
        if plane == "xz":
            solid = np.ascontiguousarray(cell_type[:, index, :], dtype=np.uint8)
        elif plane == "xy":
            solid = np.ascontiguousarray(cell_type[index, :, :], dtype=np.uint8)
        else:
            solid = np.ascontiguousarray(cell_type[:, :, index], dtype=np.uint8)

        fluid_mask = solid == 0
        has_fluid = bool(fluid_mask.any())
        entry = (fluid_mask, has_fluid, solid.tobytes())
        if len(self._solid_cache) >= 64:
            self._solid_cache.clear()
        self._solid_cache[key] = entry
        return entry

    def touch_client(self) -> None:
        """Record client activity and seamlessly wake worker if it was auto-idle-paused."""
        self._last_client_seen = time.monotonic()
        if self._idle_paused:
            self._idle_paused = False
            self._wake_event.set()

    @staticmethod
    def _make_solver(grid: str, overrides: Dict[str, Any]) -> CFD3DSolver:
        clean = {k: v for k, v in overrides.items() if k not in _NON_LIVE_FIELDS}
        params = default_params(grid=grid, **clean)
        return CFD3DSolver(params=params)

    def _append_history(self, m: Dict[str, Any], force: bool = False) -> None:
        t = float(m["sim_time"])
        if not force and (t - self._last_hist_time) < 0.08:
            return
        self._last_hist_time = t
        self._history.append(
            {
                "t": round(t, 3),
                "q_out_ls": round(float(m["q_out_ls"]), 2),
                "q_in_ls": round(float(m["q_in_ls"]), 2),
                "ach": round(float(m["ach"]), 2),
                "smoke_frac": round(float(m["smoke_indoor_frac"]) * 100.0, 2),
                "smoke_in": round(float(m["smoke_indoor"]), 5),
                "z_npp": round(float(m["z_npp"]), 3),
                "max_speed": round(float(m["max_speed"]), 3),
            }
        )

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.clear()
            if not self._running:
                self._achieved_rt = 0.0
                self._wake_event.wait(timeout=0.5)
                continue

            if (time.monotonic() - self._last_client_seen) > self._idle_timeout_s:
                self._idle_paused = True
                self._achieved_rt = 0.0
                self._wake_event.wait(timeout=0.5)
                continue

            self._idle_paused = False

            t0 = time.perf_counter()
            sim_dt_burst = 0.0
            scale = self._time_scale

            # Run larger C bursts when speeding up time so HTTP/lock overhead stays negligible
            if scale == 0.0:
                n_burst = 6
            elif scale >= 4.0:
                n_burst = 5
            elif scale >= 2.0:
                n_burst = 3
            else:
                n_burst = 1

            with self._lock:
                if not self._running:
                    continue
                try:
                    t_before = float(self._solver.metrics()["sim_time"])
                    if n_burst == 1:
                        m = _enrich_metrics(self._solver.step())
                    else:
                        m = _enrich_metrics(self._solver.step_n(n_burst))
                    self._step_count = int(m["step_count"])
                    sim_dt_burst = max(0.0, float(m["sim_time"]) - t_before)
                    self._append_history(m)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[cfd3d-server] solver step error: {exc}", flush=True)
                    self._running = False
                    continue

            wall_elapsed = time.perf_counter() - t0
            if scale > 0.0 and sim_dt_burst > 0.0:
                target_wall = sim_dt_burst / scale
                remain = target_wall - wall_elapsed
                # Sleep in short slices so pause/slider updates respond immediately
                while remain > 0.002 and self._running and not self._stop_event.is_set():
                    if self._time_scale != scale:
                        break
                    if (time.monotonic() - self._last_client_seen) > self._idle_timeout_s:
                        break
                    chunk = min(remain, 0.015)
                    time.sleep(chunk)
                    remain = target_wall - (time.perf_counter() - t0)
            else:
                # Brief yield so HTTP frame/param requests acquire the lock immediately
                time.sleep(0.0005)

            total_wall = max(1e-6, time.perf_counter() - t0)
            inst_rt = sim_dt_burst / total_wall
            self._achieved_rt = inst_rt if self._achieved_rt <= 0.0 else (0.75 * self._achieved_rt + 0.25 * inst_rt)

    def shutdown(self) -> None:
        self._running = False
        self._stop_event.set()
        self._wake_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._frame_lock:
            with self._lock:
                self._solver.close()

    def get_state(self, include_history: bool = True) -> Dict[str, Any]:
        self.touch_client()
        with self._lock:
            return self._build_state_unlocked(include_history=include_history)

    def _build_state_unlocked(self, include_history: bool = True) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "running": self._running,
            "state_ver": self._state_version,
            "time_scale": self._time_scale,
            "achieved_rt": round(self._achieved_rt, 2),
            "grid": self._grid_name,
            "preset": self._preset_name,
            "grid_presets": GRID_PRESETS,
            "scenario_presets": [
                {"id": pid, "label": PRESET_LABELS.get(pid, pid)}
                for pid in PRESET_ORDER
                if pid in WEB_SCENARIO_PRESETS
            ],
            "params": self._solver.params.to_dict(),
            "geom": self._get_geom_unlocked(),
            "metrics": _enrich_metrics(self._solver.metrics()),
        }
        if include_history:
            state["history"] = list(self._history)
        return state

    def update_params(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(updates, dict):
            raise ValueError("Payload must be a JSON object")

        self.touch_client()
        coerced: Dict[str, Any] = {}
        for k, v in updates.items():
            if k in _NON_LIVE_FIELDS:
                continue
            if k not in _PARAM_TYPES:
                raise KeyError(f"Unknown parameter '{k}'")
            t = _PARAM_TYPES[k]
            coerced[k] = int(round(float(v))) if t is int else float(v)

        with self._lock:
            if coerced:
                self._solver.update_params(**coerced)
                self._state_version += 1
                if any(k in _GEOM_AFFECTING_FIELDS for k in coerced):
                    self._geom_version += 1
                    self._geom_cache = None
                    self._solid_cache.clear()
            return self._build_state_unlocked(include_history=False)

    def control(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object")
        self.touch_client()
        action = str(payload.get("action", "")).strip().lower()

        if action == "grid":
            grid_name = str(payload.get("grid", "")).strip()
            if grid_name not in GRID_PRESETS:
                raise ValueError(
                    f"Unknown grid '{grid_name}'. Choices: {list(GRID_PRESETS)}"
                )
            with self._frame_lock:
                with self._lock:
                    if grid_name != self._grid_name:
                        cur_params = self._solver.params.to_dict()
                        was_running = self._running
                        self._running = False
                        self._idle_paused = False
                        self._solver.close()
                        self._grid_name = grid_name
                        self._solver = self._make_solver(grid_name, cur_params)
                        self._geom_version += 1
                        self._alloc_frame_buffers()
                        self._history.clear()
                        self._last_hist_time = -1.0
                        init_m = _enrich_metrics(self._solver.metrics())
                        self._step_count = int(init_m["step_count"])
                        self._append_history(init_m, force=True)
                        self._state_version += 1
                        self._running = was_running
                        if was_running:
                            self._wake_event.set()
                    return self._build_state_unlocked(include_history=True)

        with self._lock:
            if action == "play":
                self._running = True
                self._idle_paused = False
                self._state_version += 1
                self._wake_event.set()
            elif action == "pause":
                self._running = False
                self._idle_paused = False
                self._state_version += 1
            elif action == "speed":
                raw = float(payload.get("scale", 1.0))
                self._time_scale = 0.0 if raw <= 0.0 else max(0.05, min(20.0, raw))
                self._state_version += 1
            elif action == "step":
                self._running = False
                self._idle_paused = False
                n_steps = max(1, min(20, int(payload.get("n", 1))))
                for _ in range(n_steps):
                    m = _enrich_metrics(self._solver.step())
                    self._step_count = int(m["step_count"])
                    self._append_history(m, force=True)
                self._state_version += 1
            elif action == "reset":
                self._solver.reset()
                self._history.clear()
                self._last_hist_time = -1.0
                init_m = _enrich_metrics(self._solver.metrics())
                self._step_count = int(init_m["step_count"])
                self._append_history(init_m, force=True)
                self._geom_version += 1
                self._geom_cache = None
                self._solid_cache.clear()
                self._state_version += 1
            elif action == "preset":
                preset_name = str(payload.get("preset", "")).strip()
                if preset_name not in WEB_SCENARIO_PRESETS:
                    raise ValueError(
                        f"Unknown preset '{preset_name}'. Choices: {list(WEB_SCENARIO_PRESETS)}"
                    )
                self._preset_name = preset_name
                self._solver.update_params(**WEB_SCENARIO_PRESETS[preset_name])
                self._solver.reset()
                self._history.clear()
                self._last_hist_time = -1.0
                init_m = _enrich_metrics(self._solver.metrics())
                self._step_count = int(init_m["step_count"])
                self._append_history(init_m, force=True)
                self._geom_version += 1
                self._geom_cache = None
                self._solid_cache.clear()
                self._state_version += 1
            else:
                raise ValueError(
                    f"Unknown control action '{action}'. "
                    "Use play|pause|step|reset|preset|grid."
                )
            return self._build_state_unlocked(include_history=True)

    def build_binary_frame(
        self,
        field: str = "smoke",
        vol_field: str = "smoke",
        xz_j: Optional[int] = None,
        xy_k: Optional[int] = None,
        yz_i: Optional[int] = None,
        vectors: bool = True,
        include_vol: bool = True,
        vol_vmin: Optional[float] = None,
        vol_vmax: Optional[float] = None,
        last_step: Optional[int] = None,
        last_ver: Optional[int] = None,
        last_key: Optional[str] = None,
    ) -> Optional[bytes]:
        self.touch_client()
        field = field if field in SLICE_FIELDS else "smoke"
        vol_field = vol_field if vol_field in VOL_FIELDS else "smoke"

        with self._frame_lock:
            with self._lock:
                g = self._get_geom_unlocked()
                nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])

                j_idx = (
                    max(0, min(ny - 1, int(xz_j)))
                    if xz_j is not None
                    else (int(g["j_w0"]) + int(g["j_w1"])) // 2
                )
                k_idx = (
                    max(0, min(nz - 1, int(xy_k)))
                    if xy_k is not None
                    else (int(g["k_w0"]) + int(g["k_w1"])) // 2
                )
                i_idx = (
                    max(0, min(nx - 1, int(yz_i)))
                    if yz_i is not None
                    else int(g["ix_wall"])
                )

                cur_key = (
                    f"{field}|{vol_field}|{j_idx}|{k_idx}|{i_idx}|"
                    f"{1 if vectors else 0}|{1 if include_vol else 0}|{vol_vmin}|{vol_vmax}"
                )

                step_count = self._step_count
                state_ver = self._state_version
                if (
                    last_step is not None
                    and last_ver is not None
                    and step_count == last_step
                    and state_ver == last_ver
                    and (last_key is None or last_key == cur_key)
                ):
                    return None

                p = self._solver.params.to_dict()
                m = _enrich_metrics(self._solver.metrics())
                self._step_count = int(m["step_count"])

                # Determine volume packing bounds
                if vol_vmin is None or vol_vmax is None:
                    if vol_field == "smoke":
                        v_lo, v_hi = 0.0, 0.85
                    elif vol_field == "temp":
                        # Symmetric strictly around T_out so outdoor ambient is exact 0.5 (transparent),
                        # warmer-than-outdoor air is always > 0.5 (amber/red), colder is always < 0.5 (blue).
                        t_out = float(p["temp_out_c"])
                        span = max(
                            10.0,
                            abs(float(p["temp_in_c"]) - t_out) + 2.0,
                            abs(float(p["smoke_temp_c"]) - t_out) + 2.0,
                        )
                        v_lo, v_hi = t_out - span, t_out + span
                    elif vol_field == "speed":
                        v_lo, v_hi = 0.0, max(1.0, float(m["max_speed"]) * 0.9)
                    else:  # vorticity
                        v_lo, v_hi = 0.0, 12.0
                else:
                    v_lo, v_hi = float(vol_vmin), float(vol_vmax)

                # Extract 3D volume into reusable uint8 buffer only when requested
                if include_vol:
                    self._solver.pack_volume_u8(vol_field, vmin=v_lo, vmax=v_hi, out=self._vol_buf)

                # Extract 2D scalar slices into pre-allocated reusable float32 buffers
                self._solver.extract_slice(field, "xz", j_idx, out=self._xz_val)
                self._solver.extract_slice(field, "xy", k_idx, out=self._xy_val)
                self._solver.extract_slice(field, "yz", i_idx, out=self._yz_val)

                # Fetch cached solid masks & precomputed fluid boolean masks
                xz_fmask, xz_has_fluid, xz_solid_bytes = self._get_solid_slice_unlocked("xz", j_idx)
                xy_fmask, xy_has_fluid, xy_solid_bytes = self._get_solid_slice_unlocked("xy", k_idx)
                yz_fmask, yz_has_fluid, yz_solid_bytes = self._get_solid_slice_unlocked("yz", i_idx)

                # Optional velocity vectors into pre-allocated reusable float32 buffers
                if vectors:
                    self._solver.extract_vec_slice("xz", j_idx, out=(self._xz_u1, self._xz_u2))
                    self._solver.extract_vec_slice("xy", k_idx, out=(self._xy_u1, self._xy_u2))
                    self._solver.extract_vec_slice("yz", i_idx, out=(self._yz_u1, self._yz_u2))

                hist_tail = list(self._history)[-160:]
                running = self._running
                time_scale = self._time_scale
                achieved_rt = round(self._achieved_rt, 2)
                grid_name = self._grid_name
                preset_name = self._preset_name

            # Compute min/max & assemble frame outside solver lock (still protected by _frame_lock)
            xz_mm = _fluid_minmax(self._xz_val, xz_fmask, xz_has_fluid)
            xy_mm = _fluid_minmax(self._xy_val, xy_fmask, xy_has_fluid)
            yz_mm = _fluid_minmax(self._yz_val, yz_fmask, yz_has_fluid)
            g_min = min(xz_mm[0], xy_mm[0], yz_mm[0])
            g_max = max(xz_mm[1], xy_mm[1], yz_mm[1])

            # Assemble 4-byte-aligned binary payload chunks using zero-copy memoryviews
            raw_chunks: List[Tuple[str, str, List[int], Any]] = []
            if include_vol:
                raw_chunks.append(("volume", "u8", [nz, ny, nx], _bytes_view(self._vol_buf)))

            raw_chunks.extend(
                [
                    ("xz_val", "f32", [nz, nx], _bytes_view(self._xz_val, expect_le_f4=True)),
                    ("xy_val", "f32", [ny, nx], _bytes_view(self._xy_val, expect_le_f4=True)),
                    ("yz_val", "f32", [nz, ny], _bytes_view(self._yz_val, expect_le_f4=True)),
                    ("xz_solid", "u8", [nz, nx], xz_solid_bytes),
                    ("xy_solid", "u8", [ny, nx], xy_solid_bytes),
                    ("yz_solid", "u8", [nz, ny], yz_solid_bytes),
                ]
            )

            if vectors:
                raw_chunks.extend(
                    [
                        ("xz_u1", "f32", [nz, nx], _bytes_view(self._xz_u1, expect_le_f4=True)),
                        ("xz_u2", "f32", [nz, nx], _bytes_view(self._xz_u2, expect_le_f4=True)),
                        ("xy_u1", "f32", [ny, nx], _bytes_view(self._xy_u1, expect_le_f4=True)),
                        ("xy_u2", "f32", [ny, nx], _bytes_view(self._xy_u2, expect_le_f4=True)),
                        ("yz_u1", "f32", [nz, ny], _bytes_view(self._yz_u1, expect_le_f4=True)),
                        ("yz_u2", "f32", [nz, ny], _bytes_view(self._yz_u2, expect_le_f4=True)),
                    ]
                )

            chunk_manifest: Dict[str, Dict[str, Any]] = {}
            payload_parts: List[Any] = []
            offset = 0
            for name, dtype, shape, data in raw_chunks:
                nbytes = len(data)
                chunk_manifest[name] = {
                    "dtype": dtype,
                    "shape": shape,
                    "offset": offset,
                    "bytes": nbytes,
                }
                payload_parts.append(data)
                offset += nbytes
                pad = (-offset) & 3
                if pad:
                    payload_parts.append(b"\x00" * pad)
                    offset += pad

            header_obj = {
                "running": running,
                "state_ver": state_ver,
                "frame_key": cur_key,
                "time_scale": time_scale,
                "achieved_rt": achieved_rt,
                "grid": grid_name,
                "preset": preset_name,
                "field": field,
                "vol_field": vol_field,
                "vol_vmin": v_lo,
                "vol_vmax": v_hi,
                "xz_j": j_idx,
                "xy_k": k_idx,
                "yz_i": i_idx,
                "slices_minmax": {
                    "xz": [xz_mm[0], xz_mm[1]],
                    "xy": [xy_mm[0], xy_mm[1]],
                    "yz": [yz_mm[0], yz_mm[1]],
                    "global": [g_min, g_max],
                },
                "metrics": m,
                "geom": g,
                "params": p,
                "history": hist_tail,
                "chunks": chunk_manifest,
            }

            header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")
            # Pad header with ASCII spaces so payload starts at a 4-byte aligned offset
            hpad = (-len(header_bytes)) & 3
            if hpad:
                header_bytes += b" " * hpad

            prefix = struct.pack("<I", len(header_bytes))
            return b"".join([prefix, header_bytes, *payload_parts])


def _parse_opt_int(qs: Dict[str, List[str]], key: str) -> Optional[int]:
    vals = qs.get(key)
    if not vals or vals[0] == "":
        return None
    try:
        return int(vals[0])
    except ValueError:
        return None


def _parse_opt_float(qs: Dict[str, List[str]], key: str) -> Optional[float]:
    vals = qs.get(key)
    if not vals or vals[0] == "":
        return None
    try:
        return float(vals[0])
    except ValueError:
        return None


class CFDRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler serving static frontend assets and live CFD JSON/binary API."""

    protocol_version = "HTTP/1.1"
    manager: SimulationManager  # Attached dynamically to subclass

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep console clean (suppress noisy 30 FPS /api/frame logs)
        if args and isinstance(args[0], str) and "/api/frame" in args[0]:
            return
        super().log_message(fmt, *args)

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: int = HTTPStatus.OK,
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status=status)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        if length > 2_000_000:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/state":
            try:
                state = self.manager.get_state(include_history=True)
                self._send_json(state)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/api/frame":
            try:
                qs = parse_qs(parsed.query)
                field = (qs.get("field", ["smoke"])[0] or "smoke").lower()
                vol_field = (qs.get("vol_field", ["smoke"])[0] or "smoke").lower()
                xz_j = _parse_opt_int(qs, "xz_j")
                xy_k = _parse_opt_int(qs, "xy_k")
                yz_i = _parse_opt_int(qs, "yz_i")
                vectors = (qs.get("vectors", ["1"])[0] or "1") not in ("0", "false", "no")
                include_vol = (qs.get("vol", ["1"])[0] or "1") not in ("0", "false", "no")
                vol_vmin = _parse_opt_float(qs, "vol_vmin")
                vol_vmax = _parse_opt_float(qs, "vol_vmax")
                last_step = _parse_opt_int(qs, "last_step")
                last_ver = _parse_opt_int(qs, "last_ver")
                last_key_list = qs.get("last_key")
                last_key = last_key_list[0] if (last_key_list and last_key_list[0] != "") else None

                frame = self.manager.build_binary_frame(
                    field=field,
                    vol_field=vol_field,
                    xz_j=xz_j,
                    xy_k=xy_k,
                    yz_i=yz_i,
                    vectors=vectors,
                    include_vol=include_vol,
                    vol_vmin=vol_vmin,
                    vol_vmax=vol_vmax,
                    last_step=last_step,
                    last_ver=last_ver,
                    last_key=last_key,
                )
                if frame is None:
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.send_header("Content-Length", "0")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                else:
                    self._send_bytes(frame, "application/octet-stream")
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        # Serve static files from web/
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        try:
            target = (WEB_DIR / rel).resolve()
            web_root = WEB_DIR.resolve()
            if not target.is_relative_to(web_root):
                self._send_json({"error": "Forbidden"}, status=HTTPStatus.FORBIDDEN)
                return
            if not target.is_file():
                self._send_json({"error": f"Not found: {path}"}, status=HTTPStatus.NOT_FOUND)
                return

            ctype, _ = mimetypes.guess_type(str(target))
            if target.suffix == ".js":
                ctype = "application/javascript; charset=utf-8"
            elif target.suffix == ".css":
                ctype = "text/css; charset=utf-8"
            elif target.suffix == ".html":
                ctype = "text/html; charset=utf-8"
            elif not ctype:
                ctype = "application/octet-stream"

            data = target.read_bytes()
            self._send_bytes(data, ctype, cache_control="no-cache")
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/params":
            try:
                body = self._read_json_body()
                state = self.manager.update_params(body)
                self._send_json(state)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/api/control":
            try:
                body = self._read_json_body()
                state = self.manager.control(body)
                self._send_json(state)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json({"error": f"Unknown endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    grid: str = "fast",
    preset: str = "cold_day",
    auto_start: bool = True,
    idle_timeout_s: float = 12.0,
) -> Tuple[ThreadingHTTPServer, SimulationManager]:
    manager = SimulationManager(
        grid=grid,
        preset=preset,
        auto_start=auto_start,
        idle_timeout_s=idle_timeout_s,
    )

    class _BoundHandler(CFDRequestHandler):
        pass

    _BoundHandler.manager = manager
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer((host, port), _BoundHandler)
    httpd.daemon_threads = True
    return httpd, manager


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live 3D Boussinesq Smoke & UK Casement Window CFD Web Server"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument(
        "--grid",
        default="fast",
        choices=list(GRID_PRESETS.keys()),
        help="Initial grid resolution preset (default: fast)",
    )
    parser.add_argument(
        "--preset",
        default="cold_day",
        choices=PRESET_ORDER,
        help="Initial physical scenario preset (default: cold_day)",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Start with simulation paused",
    )
    args = parser.parse_args()

    httpd, manager = create_server(
        host=args.host,
        port=args.port,
        grid=args.grid,
        preset=args.preset,
        auto_start=not args.paused,
    )
    g = GRID_PRESETS[args.grid]
    print(
        f"3D CFD Live Server running at http://{args.host}:{args.port}/ "
        f"(grid={args.grid} {g['nx']}x{g['ny']}x{g['nz']}, preset={args.preset})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down 3D CFD server...", flush=True)
    finally:
        httpd.server_close()
        manager.shutdown()


if __name__ == "__main__":
    main()
