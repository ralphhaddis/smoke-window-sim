"""Rigorous Verification & Validation (V&V) Test Suite for the 3D Boussinesq CFD Engine.

Tests implemented:
  1. CTypes ABI Layout & Field Offset Check (C sizeof/offsetof vs Python ctypes)
  2. Quiescent Ambient Equilibrium (50 steps, 45 deg open window)
  3. Closed-Window Hydrostatic Warm Room (50 steps, dp'/dz validation)
  4. Sealed-Room Smoke Conservation & Positivity (20 emit + 60 free-circulate steps)
  5. Angled UK Sash Impermeability (0, 20, 45, 70, 90 deg; all 3 hinges; zero diagonal leakage + dynamic zero velocity)
  6. Analytical Linden / Brown-Solvason Single-Sided Buoyancy Exchange + Hot Summer Reversal
  7. Long-Horizon Open-Boundary Stability Stress Tests:
       7a. Cold Winter Day Long Run (t >= 28 s, 45 deg sash, hot smoke, wind=0)
       7b. Strong Oblique Crosswind + Head-on Wind Long Run (t >= 15 s, U_inf=2.5 m/s)
       7c. Heavy Cold Smoke + Top-Hung Awning Sash (30 deg, T_smoke=8 C, alpha_s=0.05)
  8. Rear Vent Bernoulli Orifice Check (Delta P = +3 Pa and -3 Pa)
  9. Performance Benchmark (ms/step, Mcells/sec)
"""

from __future__ import annotations

import ctypes as ct
import math
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .solver import (
    CFD3DSolver,
    CFDGeomInfo,
    CFDMetrics,
    CFDParams,
    CELL_FLUID,
    HINGE_LEFT,
    HINGE_RIGHT,
    HINGE_TOP,
    ZONE_INDOOR,
    ZONE_OUTDOOR,
    ZONE_REVEAL,
    default_params,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
CSRC_DIR = ROOT_DIR / "csrc"

CFD_G = 9.80665
CFD_P_ATM = 101325.0
CFD_R_AIR = 287.05


class TestResult:
    def __init__(self, name: str, passed: bool, summary: str, details: List[str] | None = None):
        self.name = name
        self.passed = passed
        self.summary = summary
        self.details = details or []


def check_finite_fields(sim: CFD3DSolver) -> Tuple[bool, str]:
    for fname in ("u", "v", "w", "phi", "theta", "smoke", "nu_sgs"):
        arr = sim.field_view(fname)
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            return False, f"Field '{fname}' has {n_bad} non-finite values!"
    return True, "all finite"


# =============================================================================
# TEST 1: CTypes ABI Layout & Field Offset Check
# =============================================================================
def test_1_ctypes_abi() -> TestResult:
    print("\n" + "=" * 78)
    print("TEST 1: CTypes ABI Layout & Field Offset Verification")
    print("=" * 78)

    structs = [
        ("CFDParams", CFDParams),
        ("CFDMetrics", CFDMetrics),
        ("CFDGeomInfo", CFDGeomInfo),
    ]

    lines = [
        '#include <stdio.h>',
        '#include <stddef.h>',
        '#include "cfd3d.h"',
        'int main(void) {',
    ]
    for sname, scls in structs:
        lines.append(f'    printf("SIZE {sname} %zu\\n", sizeof({sname}));')
        for fname, _ in scls._fields_:
            lines.append(f'    printf("OFF {sname} {fname} %zu\\n", offsetof({sname}, {fname}));')
    lines.append("    return 0;")
    lines.append("}")
    c_src = "\n".join(lines) + "\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "abi_check.c"
        bin_path = Path(tmpdir) / "abi_check"
        src_path.write_text(c_src)
        cmd = ["clang", "-std=c99", "-I", str(CSRC_DIR), str(src_path), "-o", str(bin_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return TestResult(
                "1. CTypes ABI Layout",
                False,
                f"Failed to compile ABI probe: {res.stderr}",
            )
        run = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)

    c_sizes: Dict[str, int] = {}
    c_offsets: Dict[Tuple[str, str], int] = {}
    for line in run.stdout.strip().splitlines():
        parts = line.split()
        if parts[0] == "SIZE":
            c_sizes[parts[1]] = int(parts[2])
        elif parts[0] == "OFF":
            c_offsets[(parts[1], parts[2])] = int(parts[3])

    mismatches: List[str] = []
    details: List[str] = []
    for sname, scls in structs:
        py_sz = ct.sizeof(scls)
        c_sz = c_sizes[sname]
        status = "OK" if py_sz == c_sz else "MISMATCH"
        msg = f"  sizeof({sname}): C={c_sz} bytes, Python={py_sz} bytes [{status}]"
        print(msg)
        details.append(msg)
        if py_sz != c_sz:
            mismatches.append(msg)
        for fname, _ in scls._fields_:
            py_off = getattr(scls, fname).offset
            c_off = c_offsets[(sname, fname)]
            if py_off != c_off:
                m = f"  offsetof({sname}.{fname}): C={c_off}, Python={py_off} [MISMATCH]"
                print(m)
                mismatches.append(m)

    passed = len(mismatches) == 0
    total_fields = sum(len(scls._fields_) for _, scls in structs)
    summary = (
        f"All 3 structs ({total_fields} fields) match C ABI sizes & offsets exactly"
        if passed
        else f"{len(mismatches)} ABI mismatches found!"
    )
    print(f"-> {'PASS' if passed else 'FAIL'}: {summary}")
    return TestResult("1. CTypes ABI Layout", passed, summary, details + mismatches)


# =============================================================================
# TEST 2: Quiescent Ambient Equilibrium
# =============================================================================
def test_2_quiescent_equilibrium() -> TestResult:
    print("\n" + "=" * 78)
    print("TEST 2: Quiescent Ambient Equilibrium (50 steps, 45 deg open window)")
    print("=" * 78)

    with CFD3DSolver(
        grid="fast",
        preset="quiescent",
        temp_in_c=20.0,
        temp_out_c=20.0,
        win_angle_deg=45.0,
        wind_speed=0.0,
        vent_enabled=0,
        smoke_enabled=0,
    ) as sim:
        max_spd_all = 0.0
        max_div_all = 0.0
        max_pcg = 0
        for _ in range(50):
            m = sim.step()
            max_spd_all = max(max_spd_all, m["max_speed"])
            max_div_all = max(max_div_all, m["max_div"])
            max_pcg = max(max_pcg, m["pcg_iters"])

        raw_u_max = float(np.max(np.abs(sim.field_view("u"))))
        raw_v_max = float(np.max(np.abs(sim.field_view("v"))))
        raw_w_max = float(np.max(np.abs(sim.field_view("w"))))
        raw_vel_max = max(raw_u_max, raw_v_max, raw_w_max)
        raw_phi_max = float(np.max(np.abs(sim.field_view("phi"))))
        fin_ok, fin_msg = check_finite_fields(sim)

    passed = fin_ok and (max_spd_all < 1e-13) and (raw_vel_max < 1e-13) and (max_div_all < 1e-13)
    summary = (
        f"max|u|_metric={max_spd_all:.3e}, max|u|_raw={raw_vel_max:.3e} m/s, "
        f"max|div u|={max_div_all:.3e} 1/s, max|phi|={raw_phi_max:.3e}, max_pcg={max_pcg}"
    )
    print(f"  {summary}")
    print(f"-> {'PASS' if passed else 'FAIL'}")
    return TestResult("2. Quiescent Equilibrium", passed, summary)


# =============================================================================
# TEST 3: Closed-Window Hydrostatic Warm Room
# =============================================================================
def test_3_closed_hydrostatic() -> TestResult:
    print("\n" + "=" * 78)
    print("TEST 3: Closed-Window Hydrostatic Warm Room (50 steps, T_in=22C, T_out=2C, 0 deg)")
    print("=" * 78)

    temp_in_c = 22.0
    temp_out_c = 2.0
    t_out_k = temp_out_c + 273.15
    beta = 1.0 / t_out_k
    rho0 = CFD_P_ATM / (CFD_R_AIR * t_out_k)
    dT = temp_in_c - temp_out_c
    dpdz_theory = rho0 * CFD_G * beta * dT

    with CFD3DSolver(
        grid="fast",
        temp_in_c=temp_in_c,
        temp_out_c=temp_out_c,
        win_angle_deg=0.0,
        wind_speed=0.0,
        vent_enabled=0,
        smoke_enabled=0,
        smoke_rate=0.0,
    ) as sim:
        g = sim.geom_info()
        h = g["h"]
        max_spd_all = 0.0
        max_div_all = 0.0
        for _ in range(50):
            m = sim.step()
            max_spd_all = max(max_spd_all, m["max_speed"])
            max_div_all = max(max_div_all, m["max_div"])

        u = sim.field_view("u")
        v = sim.field_view("v")
        w = sim.field_view("w")
        phi = sim.field_view("phi")
        zone = sim.mask_view("cell_zone")
        raw_vel_max = max(float(np.max(np.abs(u))), float(np.max(np.abs(v))), float(np.max(np.abs(w))))

        # Indoor vertical pressure gradient: dp'/dz = rho0 * (phi[k+1] - phi[k]) / h
        indoor_mask = (zone == ZONE_INDOOR)
        pair_mask = indoor_mask[:-1, :, :] & indoor_mask[1:, :, :]
        dpdz_num = rho0 * (phi[1:, :, :] - phi[:-1, :, :]) / h
        ratios = dpdz_num[pair_mask] / dpdz_theory
        ratio_mean = float(np.mean(ratios))
        ratio_min = float(np.min(ratios))
        ratio_max = float(np.max(ratios))
        max_ratio_err = float(np.max(np.abs(ratios - 1.0)))

        # Outdoor phi should remain ~0
        outdoor_phi_max = float(np.max(np.abs(phi[zone == ZONE_OUTDOOR])))
        fin_ok, _ = check_finite_fields(sim)

    passed = (
        fin_ok
        and (max_spd_all < 1e-12)
        and (raw_vel_max < 1e-12)
        and (max_div_all < 1e-12)
        and (max_ratio_err < 1e-4)
        and (outdoor_phi_max < 1e-12)
    )
    summary = (
        f"max|u|_metric={max_spd_all:.3e}, max|u|_raw={raw_vel_max:.3e} m/s, "
        f"max|div|={max_div_all:.3e}, dp'/dz / (rho0*g*beta*dT) = {ratio_mean:.8f} "
        f"(min={ratio_min:.8f}, max={ratio_max:.8f}, max_err={max_ratio_err:.3e}), "
        f"max|phi_out|={outdoor_phi_max:.3e}"
    )
    print(f"  Theory dp'/dz = {dpdz_theory:.6f} Pa/m")
    print(f"  {summary}")
    print(f"-> {'PASS' if passed else 'FAIL'}")
    return TestResult("3. Closed-Window Hydrostatic", passed, summary)


# =============================================================================
# TEST 4: Sealed-Room Smoke Conservation & Positivity
# =============================================================================
def test_4_smoke_conservation() -> TestResult:
    print("\n" + "=" * 78)
    print("TEST 4: Sealed-Room Smoke Conservation & Positivity (20 emit + 60 free steps)")
    print("=" * 78)

    with CFD3DSolver(
        grid="fast",
        temp_in_c=20.0,
        temp_out_c=20.0,
        win_angle_deg=0.0,
        wind_speed=0.0,
        vent_enabled=0,
        smoke_enabled=1,
        smoke_x=1.80,
        smoke_y=1.50,
        smoke_z=1.10,
        smoke_radius=0.20,
        smoke_temp_c=45.0,
        smoke_rate=0.08,
        smoke_jet_speed=0.4,
        smoke_jet_azim_deg=30.0,
        smoke_jet_elev_deg=20.0,
    ) as sim:
        g = sim.geom_info()
        h3 = g["h"] ** 3

        # Phase 1: 20 steps with emitter ON
        min_smoke_all = 0.0
        injected_expected = 0.0
        for _ in range(20):
            m = sim.step()
            injected_expected += 0.08 * m["dt"]
            sm = sim.field_view("smoke")
            min_smoke_all = min(min_smoke_all, float(np.min(sm)))

        zone = sim.mask_view("cell_zone")
        sm = sim.field_view("smoke")
        mass_t0 = float(np.sum(sm)) * h3
        outdoor_t0 = float(np.sum(sm[zone == ZONE_OUTDOOR])) * h3
        emit_rel_err = abs(mass_t0 - injected_expected) / max(injected_expected, 1e-15)

        # Turn emitter OFF and keep plume circulating buoyantly for 60 steps
        sim.update_params(smoke_enabled=0, smoke_rate=0.0, smoke_jet_speed=0.0)

        max_rel_drift = 0.0
        max_div_free = 0.0
        max_spd_free = 0.0
        for _ in range(60):
            m = sim.step()
            sm = sim.field_view("smoke")
            min_smoke_all = min(min_smoke_all, float(np.min(sm)))
            mass_now = float(np.sum(sm)) * h3
            rel_drift = abs(mass_now - mass_t0) / max(mass_t0, 1e-15)
            max_rel_drift = max(max_rel_drift, rel_drift)
            max_div_free = max(max_div_free, m["max_div"])
            max_spd_free = max(max_spd_free, m["max_speed"])

        mass_final = float(np.sum(sim.field_view("smoke"))) * h3
        outdoor_final = float(np.sum(sim.field_view("smoke")[zone == ZONE_OUTDOOR])) * h3
        signed_rel_drift = (mass_final - mass_t0) / max(mass_t0, 1e-15)
        fin_ok, _ = check_finite_fields(sim)

    passed = (
        fin_ok
        and (min_smoke_all >= -1e-14)
        and (max_rel_drift < 1e-12)
        and (outdoor_final == 0.0)
        and (max_spd_free > 0.05)  # verify fluid was actually circulating!
    )
    summary = (
        f"M_emit={mass_t0:.6f} (expected {injected_expected:.6f}, err={emit_rel_err:.2e}), "
        f"M_final={mass_final:.6f}, max_rel_drift={max_rel_drift:.3e} (signed={signed_rel_drift:+.3e}), "
        f"min(C)={min_smoke_all:.3e}, M_out={outdoor_final:.3e}, "
        f"peak_circ_speed={max_spd_free:.3f} m/s, max_div={max_div_free:.2e}"
    )
    print(f"  {summary}")
    print(f"-> {'PASS' if passed else 'FAIL'}")
    return TestResult("4. Sealed-Room Smoke Conservation", passed, summary)


# =============================================================================
# TEST 5: Angled UK Sash Impermeability (Zero Diagonal Leakage)
# =============================================================================
def _segment_hits_sash_interior(
    a: np.ndarray,
    b: np.ndarray,
    p0: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    n: np.ndarray,
    h: float,
    margin: float = 1e-6,
) -> bool:
    da = float(np.dot(a - p0, n))
    db = float(np.dot(b - p0, n))
    # Require both cell centers to lie strictly on opposite sides of the plane
    # (a cell center lying mathematically ON the 45-deg plane has |d| ~ 1e-17 and is
    # assigned consistently to one side by the half-open cut, verified by the BFS test).
    eps = 1e-12 * h
    if not ((da < -eps and db > eps) or (da > eps and db < -eps)):
        return False
    tau = da / (da - db)
    ip = a + tau * (b - a) - p0
    u1 = float(np.dot(ip, e1) / np.dot(e1, e1))
    u2 = float(np.dot(ip, e2) / np.dot(e2, e2))
    return (margin < u1 < 1.0 - margin) and (margin < u2 < 1.0 - margin)


def test_5_sash_impermeability() -> TestResult:
    print("\n" + "=" * 78)
    print("TEST 5: Angled UK Sash Impermeability (Zero Diagonal Leakage & Blocked Face u=0)")
    print("=" * 78)

    hinges = [("LEFT", HINGE_LEFT), ("RIGHT", HINGE_RIGHT), ("TOP", HINGE_TOP)]
    angles = [0.0, 20.0, 45.0, 70.0, 90.0]

    failures: List[str] = []
    details: List[str] = []

    with CFD3DSolver(grid="fast", win_hinge_mode=HINGE_LEFT, win_angle_deg=45.0) as sim:
        for hname, hmode in hinges:
            for ang in angles:
                sim.update_params(win_hinge_mode=hmode, win_angle_deg=ang)
                g = sim.geom_info()
                h = g["h"]
                nx, ny, nz = g["nx"], g["ny"], g["nz"]
                ct_arr = sim.mask_view("cell_type")
                kx = sim.mask_view("kx")
                ky = sim.mask_view("ky")
                kz = sim.mask_view("kz")

                corners = np.array(g["sash_corners"], dtype=np.float64)
                p0 = corners[0]
                e1 = corners[1] - p0
                e2 = corners[3] - p0
                n = np.array(g["sash_normal"], dtype=np.float64)

                # Bounding box of sash in index space (padded by 2 cells)
                mins = np.min(corners, axis=0)
                maxs = np.max(corners, axis=0)
                i0 = max(1, int(math.floor(mins[0] / h)) - 2)
                i1 = min(nx - 2, int(math.ceil(maxs[0] / h)) + 2)
                j0 = max(1, int(math.floor(mins[1] / h)) - 2)
                j1 = min(ny - 2, int(math.ceil(maxs[1] / h)) + 2)
                k0 = max(1, int(math.floor(mins[2] / h)) - 2)
                k1 = min(nz - 2, int(math.ceil(maxs[2] / h)) + 2)

                # 1) Direct open-face crossing test: NO open face dual segment may cross sash interior
                leaks = 0
                for k in range(k0, k1 + 1):
                    zc = (k + 0.5) * h
                    for j in range(j0, j1 + 1):
                        yc = (j + 0.5) * h
                        for i in range(i0, i1 + 1):
                            xc = (i + 0.5) * h
                            a = np.array([xc, yc, zc])
                            if kx[k, j, i + 1]:
                                b = np.array([xc + h, yc, zc])
                                if _segment_hits_sash_interior(a, b, p0, e1, e2, n, h):
                                    leaks += 1
                            if ky[k, j + 1, i]:
                                b = np.array([xc, yc + h, zc])
                                if _segment_hits_sash_interior(a, b, p0, e1, e2, n, h):
                                    leaks += 1
                            if kz[k + 1, j, i]:
                                b = np.array([xc, yc, zc + h])
                                if _segment_hits_sash_interior(a, b, p0, e1, e2, n, h):
                                    leaks += 1

                # 2) 6-connected BFS watertight test inside the prism extruded along sash normal
                # Restrict BFS to cells whose orthogonal projection onto sash lies strictly inside sash
                e1_sq = float(np.dot(e1, e1))
                e2_sq = float(np.dot(e2, e2))
                in_prism = set()
                neg_seeds = []
                pos_targets = set()
                for k in range(k0, k1 + 1):
                    zc = (k + 0.5) * h
                    for j in range(j0, j1 + 1):
                        yc = (j + 0.5) * h
                        for i in range(i0, i1 + 1):
                            if ct_arr[k, j, i] != CELL_FLUID:
                                continue
                            rel = np.array([ (i + 0.5) * h, yc, zc ]) - p0
                            u1 = float(np.dot(rel, e1) / e1_sq)
                            u2 = float(np.dot(rel, e2) / e2_sq)
                            dn = float(np.dot(rel, n))
                            # Strictly inside sash footprint by > h so paths cannot legally wrap around edges
                            m1 = h / math.sqrt(e1_sq)
                            m2 = h / math.sqrt(e2_sq)
                            if (m1 < u1 < 1.0 - m1) and (m2 < u2 < 1.0 - m2) and abs(dn) <= 2.5 * h:
                                in_prism.add((i, j, k))
                                if dn < -0.25 * h:
                                    neg_seeds.append((i, j, k))
                                elif dn > 0.25 * h:
                                    pos_targets.add((i, j, k))

                bfs_crossed = False
                if neg_seeds and pos_targets:
                    visited = set(neg_seeds)
                    q = deque(neg_seeds)
                    while q and not bfs_crossed:
                        ci, cj, ck = q.popleft()
                        if (ci, cj, ck) in pos_targets:
                            bfs_crossed = True
                            break
                        # 6 neighbors gated by open faces kx, ky, kz
                        nbrs = [
                            (ci + 1, cj, ck, kx[ck, cj, ci + 1]),
                            (ci - 1, cj, ck, kx[ck, cj, ci]),
                            (ci, cj + 1, ck, ky[ck, cj + 1, ci]),
                            (ci, cj - 1, ck, ky[ck, cj, ci]),
                            (ci, cj, ck + 1, kz[ck + 1, cj, ci]),
                            (ci, cj, ck - 1, kz[ck, cj, ci]),
                        ]
                        for ni, nj, nk, is_open in nbrs:
                            if is_open and (ni, nj, nk) in in_prism and (ni, nj, nk) not in visited:
                                visited.add((ni, nj, nk))
                                q.append((ni, nj, nk))

                # For closed window (0 deg), also verify indoor+reveal is completely disconnected from outdoor!
                closed_leak = False
                if ang == 0.0:
                    # Check all kx faces at ix_ext
                    ix_ext = g["ix_ext"]
                    if int(np.sum(kx[:, :, ix_ext])) != 0:
                        closed_leak = True

                line = (
                    f"  {hname:5s} {ang:4.0f}deg: blocked_faces={g['n_sash_blocked_faces']:3d}, "
                    f"crossing_open_faces={leaks}, bfs_crossed={bfs_crossed}"
                    + (f", closed_ix_ext_open={closed_leak}" if ang == 0.0 else "")
                )
                print(line)
                details.append(line)
                if leaks > 0 or bfs_crossed or closed_leak:
                    failures.append(line)

    # 3) Dynamic test: strong wind normal to 45 deg sash -> verify velocity is IDENTICALLY 0.0 on all blocked faces
    with CFD3DSolver(
        grid="fast",
        temp_in_c=22.0,
        temp_out_c=5.0,
        win_hinge_mode=HINGE_LEFT,
        win_angle_deg=45.0,
        wind_speed=3.0,
        wind_dir_deg=45.0,
        wind_use_abl=1,
        smoke_enabled=0,
    ) as sim:
        sim.step_n(30)
        u = sim.field_view("u")
        v = sim.field_view("v")
        w = sim.field_view("w")
        kx = sim.mask_view("kx")
        ky = sim.mask_view("ky")
        kz = sim.mask_view("kz")
        max_blocked_u = float(np.max(np.abs(u[kx == 0])))
        max_blocked_v = float(np.max(np.abs(v[ky == 0])))
        max_blocked_w = float(np.max(np.abs(w[kz == 0])))
        max_blocked = max(max_blocked_u, max_blocked_v, max_blocked_w)
        dyn_line = (
            f"  Dynamic 3 m/s wind @ 45deg (30 steps): max|u| on blocked faces = "
            f"({max_blocked_u:.1e}, {max_blocked_v:.1e}, {max_blocked_w:.1e})"
        )
        print(dyn_line)
        details.append(dyn_line)
        if max_blocked != 0.0:
            failures.append(dyn_line)

    passed = len(failures) == 0
    summary = (
        "Zero diagonal leaks across all 15 (hinge x angle) configs; blocked face velocities identically 0.0"
        if passed
        else f"{len(failures)} impermeability failures!"
    )
    print(f"-> {'PASS' if passed else 'FAIL'}: {summary}")
    return TestResult("5. Angled Sash Impermeability", passed, summary, details)


# =============================================================================
# TEST 6: Linden / Brown-Solvason Single-Sided Buoyancy Exchange + Hot Summer Reversal
# =============================================================================
def _run_linden_case(label: str, temp_in_c: float, temp_out_c: float, t_target: float) -> Dict[str, float]:
    with CFD3DSolver(
        grid="fast",
        temp_in_c=temp_in_c,
        temp_out_c=temp_out_c,
        win_hinge_mode=HINGE_LEFT,
        win_angle_deg=90.0,
        wind_speed=0.0,
        vent_enabled=0,
        smoke_enabled=0,
        smoke_rate=0.0,
    ) as sim:
        g = sim.geom_info()
        W = g["win_y1"] - g["win_y0"]
        H = g["win_z1"] - g["win_z0"]
        t_out_k = temp_out_c + 273.15
        beta = 1.0 / t_out_k
        g_prime = CFD_G * beta * abs(temp_in_c - temp_out_c)
        q_ideal = (1.0 / 3.0) * W * H * math.sqrt(g_prime * H)

        q_out_hist, q_in_hist, znpp_hist = [], [], []
        max_div_all = 0.0
        max_q_err = 0.0
        while True:
            m = sim.step()
            max_div_all = max(max_div_all, m["max_div"])
            max_q_err = max(max_q_err, abs(m["q_out"] - m["q_in"]))
            if m["sim_time"] >= t_target - 2.0:
                q_out_hist.append(m["q_out"])
                q_in_hist.append(m["q_in"])
                if m["z_npp"] > 0.0:
                    znpp_hist.append(m["z_npp"])
            if m["sim_time"] >= t_target:
                break

        # Inspect top-half vs bottom-half mean u across window reveal plane i = ix_wall
        u = sim.field_view("u")
        ix = g["ix_wall"]
        jw0, jw1 = g["j_w0"], g["j_w1"]
        kw0, kw1 = g["k_w0"], g["k_w1"]
        k_mid = (kw0 + kw1) // 2
        u_win_bot = float(np.mean(u[kw0:k_mid, jw0:jw1, ix]))
        u_win_top = float(np.mean(u[k_mid:kw1, jw0:jw1, ix]))

        q_out_avg = float(np.mean(q_out_hist))
        q_in_avg = float(np.mean(q_in_hist))
        q_avg = 0.5 * (q_out_avg + q_in_avg)
        cd = q_avg / q_ideal
        # Also compute peak quasi-steady Cd (before finite room cools/warms significantly)
        cd_peak = max(0.5 * (a + b) for a, b in zip(q_out_hist, q_in_hist)) / q_ideal
        z_npp_avg = float(np.mean(znpp_hist)) if znpp_hist else -1.0
        z_mid = 0.5 * (g["win_z0"] + g["win_z1"])

        print(
            f"  [{label}] t={m['sim_time']:.2f}s (steps={m['step_count']}): "
            f"W={W:.3f}m, H={H:.3f}m, Q_ideal={q_ideal:.4f} m3/s"
        )
        print(
            f"    Q_out={q_out_avg:.4f}, Q_in={q_in_avg:.4f} m3/s, "
            f"max|Q_out-Q_in|={max_q_err:.3e}, Cd={cd:.3f} (peak={cd_peak:.3f}), "
            f"z_NPP={z_npp_avg:.3f}m (mid={z_mid:.3f}m)"
        )
        print(
            f"    u_top={u_win_top:+.4f} m/s, u_bot={u_win_bot:+.4f} m/s, "
            f"T_in_final={m['temp_mean_in_c']:.2f}C, max_div={max_div_all:.2e}"
        )
        return {
            "q_out": q_out_avg,
            "q_in": q_in_avg,
            "max_q_err": max_q_err,
            "cd": cd,
            "cd_peak": cd_peak,
            "z_npp": z_npp_avg,
            "z_mid": z_mid,
            "u_top": u_win_top,
            "u_bot": u_win_bot,
            "max_div": max_div_all,
            "t_in_final": m["temp_mean_in_c"],
        }


def test_6_linden_buoyancy_exchange() -> TestResult:
    print("\n" + "=" * 78)
    print("TEST 6: Linden / Brown-Solvason Buoyancy Exchange & Hot Summer Reversal")
    print("=" * 78)

    # Measure quasi-steady window ~6-10 s (warm winter room & hot summer reversal)
    winter = _run_linden_case("Cold Winter 22C/2C", 22.0, 2.0, t_target=10.0)
    summer = _run_linden_case("Hot Summer 18C/35C", 18.0, 35.0, t_target=8.0)

    w_ok = (
        winter["max_q_err"] < 1e-5
        and 0.45 <= winter["cd"] <= 0.75
        and winter["u_top"] > 0.1
        and winter["u_bot"] < -0.1
    )
    s_ok = (
        summer["max_q_err"] < 1e-5
        and 0.45 <= summer["cd"] <= 0.75
        and summer["u_top"] < -0.1
        and summer["u_bot"] > 0.1
    )
    passed = w_ok and s_ok
    summary = (
        f"Winter: Cd={winter['cd']:.3f}, |dQ|={winter['max_q_err']:.2e}, u_top={winter['u_top']:+.3f}, u_bot={winter['u_bot']:+.3f} | "
        f"Summer: Cd={summer['cd']:.3f}, |dQ|={summer['max_q_err']:.2e}, u_top={summer['u_top']:+.3f}, u_bot={summer['u_bot']:+.3f}"
    )
    print(f"-> {'PASS' if passed else 'FAIL'}: {summary}")
    return TestResult("6. Linden Buoyancy & Summer Reversal", passed, summary)


# =============================================================================
# TEST 7: Long-Horizon Open-Boundary Stability Stress Tests
# =============================================================================
def test_7_long_horizon_stability() -> TestResult:
    print("\n" + "=" * 78)
    print("TEST 7: Long-Horizon Open-Boundary Stability Stress Tests")
    print("=" * 78)

    all_ok = True
    details: List[str] = []

    # ---- 7a: Cold Winter Day Long Run (t >= 28 s) ----
    print("\n  [7a] Cold Winter Day Long Run (80x64x56, 22C/2C, 45deg sash, 45C smoke, wind=0, t>=28s)")
    t0 = time.perf_counter()
    with CFD3DSolver(
        grid="fast",
        temp_in_c=22.0,
        temp_out_c=2.0,
        win_hinge_mode=HINGE_LEFT,
        win_angle_deg=45.0,
        wind_speed=0.0,
        vent_enabled=0,
        smoke_enabled=1,
        smoke_temp_c=45.0,
        smoke_rate=0.08,
    ) as sim:
        max_spd_7a = 0.0
        max_div_7a = 0.0
        max_ke_7a = 0.0
        max_pcg_7a = 0
        max_open_face_spd = 0.0
        step_ms_list = []
        next_log_t = 5.0
        while True:
            m = sim.step()
            step_ms_list.append(m["step_ms"])
            max_spd_7a = max(max_spd_7a, m["max_speed"])
            max_div_7a = max(max_div_7a, m["max_div"])
            max_ke_7a = max(max_ke_7a, m["kinetic_energy"])
            max_pcg_7a = max(max_pcg_7a, m["pcg_iters"])
            if m["sim_time"] >= next_log_t or m["sim_time"] >= 28.0:
                u = sim.field_view("u")
                v = sim.field_view("v")
                w = sim.field_view("w")
                b_spd = max(
                    float(np.max(np.abs(u[:, :, -1]))),
                    float(np.max(np.abs(v[:, 0, :]))),
                    float(np.max(np.abs(v[:, -1, :]))),
                    float(np.max(np.abs(w[-1, :, :]))),
                )
                max_open_face_spd = max(max_open_face_spd, b_spd)
                print(
                    f"    t={m['sim_time']:5.2f}s (step {m['step_count']:4d}): "
                    f"max_u={m['max_speed']:.3f} m/s, open_bc_u={b_spd:.3f} m/s, "
                    f"KE={m['kinetic_energy']:.2f} J, Q_out={m['q_out']:.4f}, "
                    f"div={m['max_div']:.2e}, pcg={m['pcg_iters']:2d}"
                )
                next_log_t += 5.0
            if m["sim_time"] >= 28.0:
                break

        fin_7a, fin_msg_7a = check_finite_fields(sim)
        wall_7a = time.perf_counter() - t0
        ok_7a = fin_7a and (max_spd_7a < 3.0) and (max_div_7a < 1e-4) and (max_ke_7a < 200.0)
        line_7a = (
            f"7a Cold Winter ({m['step_count']} steps, t={m['sim_time']:.2f}s, {wall_7a:.1f}s wall, "
            f"{np.mean(step_ms_list):.1f} ms/step): max_u={max_spd_7a:.3f} m/s, "
            f"max_open_bc_u={max_open_face_spd:.3f} m/s, max_KE={max_ke_7a:.2f} J, "
            f"max_div={max_div_7a:.2e}, max_pcg={max_pcg_7a}, finite={fin_msg_7a} -> {'PASS' if ok_7a else 'FAIL'}"
        )
        print("  " + line_7a)
        details.append(line_7a)
        all_ok = all_ok and ok_7a

    # ---- 7b: Strong Oblique Crosswind + Head-on Wind Long Run (t >= 15 s) ----
    print("\n  [7b] Strong Oblique Wind Long Run (80x64x56, U_inf=2.5 m/s @ 45deg, ABL=1, 45deg sash, t>=15s)")
    t0 = time.perf_counter()
    with CFD3DSolver(
        grid="fast",
        temp_in_c=20.0,
        temp_out_c=8.0,
        win_hinge_mode=HINGE_LEFT,
        win_angle_deg=45.0,
        wind_speed=2.5,
        wind_dir_deg=45.0,
        wind_use_abl=1,
        vent_enabled=0,
        smoke_enabled=1,
        smoke_temp_c=30.0,
        smoke_rate=0.08,
    ) as sim:
        max_spd_7b = 0.0
        max_div_7b = 0.0
        max_ke_7b = 0.0
        max_pcg_7b = 0
        next_log_t = 3.0
        while True:
            m = sim.step()
            max_spd_7b = max(max_spd_7b, m["max_speed"])
            max_div_7b = max(max_div_7b, m["max_div"])
            max_ke_7b = max(max_ke_7b, m["kinetic_energy"])
            max_pcg_7b = max(max_pcg_7b, m["pcg_iters"])
            if m["sim_time"] >= next_log_t or m["sim_time"] >= 15.0:
                print(
                    f"    t={m['sim_time']:5.2f}s (step {m['step_count']:4d}): "
                    f"max_u={m['max_speed']:.3f} m/s, KE={m['kinetic_energy']:.2f} J, "
                    f"Q_out={m['q_out']:.4f}, Q_in={m['q_in']:.4f}, "
                    f"div={m['max_div']:.2e}, pcg={m['pcg_iters']:2d}"
                )
                next_log_t += 3.0
            if m["sim_time"] >= 15.0:
                break

        fin_7b, fin_msg_7b = check_finite_fields(sim)
        wall_7b = time.perf_counter() - t0
        # With U_inf=2.5 m/s at z_ref=2m (ABL gives ~3.1 m/s at z=4.5m sky) + building corner speed-up (~1.5-2x),
        # physical bounded speed is < 8.0 m/s
        ok_7b = fin_7b and (max_spd_7b < 8.0) and (max_div_7b < 1e-4) and (max_ke_7b < 5000.0)
        line_7b = (
            f"7b Oblique Wind ({m['step_count']} steps, t={m['sim_time']:.2f}s, {wall_7b:.1f}s wall): "
            f"max_u={max_spd_7b:.3f} m/s, max_KE={max_ke_7b:.2f} J, "
            f"max_div={max_div_7b:.2e}, max_pcg={max_pcg_7b}, finite={fin_msg_7b} -> {'PASS' if ok_7b else 'FAIL'}"
        )
        print("  " + line_7b)
        details.append(line_7b)
        all_ok = all_ok and ok_7b

    # ---- 7c: Heavy Cold Smoke + Top-Hung Awning Sash (30 deg) ----
    print("\n  [7c] Heavy Cold Smoke + Top-Hung Awning Sash (30deg, T_smoke=8C, alpha_s=0.05, t>=8s)")
    z_emit = 1.40
    with CFD3DSolver(
        grid="fast",
        temp_in_c=20.0,
        temp_out_c=5.0,
        win_hinge_mode=HINGE_TOP,
        win_angle_deg=30.0,
        wind_speed=0.0,
        vent_enabled=0,
        smoke_enabled=1,
        smoke_x=1.80,
        smoke_y=1.50,
        smoke_z=z_emit,
        smoke_temp_c=8.0,
        smoke_alpha_s=0.05,
        smoke_rate=0.08,
    ) as sim:
        g = sim.geom_info()
        h = g["h"]
        nz, ny, nx = g["nz"], g["ny"], g["nx"]
        z_grid = (np.arange(nz) + 0.5) * h
        max_spd_7c = 0.0
        max_div_7c = 0.0
        while True:
            m = sim.step()
            max_spd_7c = max(max_spd_7c, m["max_speed"])
            max_div_7c = max(max_div_7c, m["max_div"])
            if m["sim_time"] >= 8.0:
                break

        sm = sim.field_view("smoke")
        zone = sim.mask_view("cell_zone")
        sm_in = np.where(zone == ZONE_INDOOR, sm, 0.0)
        tot_in = float(np.sum(sm_in))
        z_cm = float(np.sum(sm_in * z_grid[:, None, None]) / max(tot_in, 1e-15))
        fin_7c, fin_msg_7c = check_finite_fields(sim)
        ok_7c = fin_7c and (z_cm < z_emit - 0.15) and (max_spd_7c < 3.0) and (max_div_7c < 1e-4)
        line_7c = (
            f"7c Heavy Cold Smoke ({m['step_count']} steps, t={m['sim_time']:.2f}s): "
            f"z_cm={z_cm:.3f}m vs z_emit={z_emit:.3f}m (sank by {z_emit - z_cm:+.3f}m), "
            f"max_u={max_spd_7c:.3f} m/s, max_div={max_div_7c:.2e}, finite={fin_msg_7c} -> {'PASS' if ok_7c else 'FAIL'}"
        )
        print("  " + line_7c)
        details.append(line_7c)
        all_ok = all_ok and ok_7c

    summary = "All 3 long-horizon stability stress tests passed" if all_ok else "One or more stress tests FAILED!"
    print(f"-> {'PASS' if all_ok else 'FAIL'}: {summary}")
    return TestResult("7. Long-Horizon Stability Stress Tests", all_ok, summary, details)


# =============================================================================
# TEST 8: Rear Vent Bernoulli Orifice Check
# =============================================================================
def test_8_rear_vent_bernoulli() -> TestResult:
    print("\n" + "=" * 78)
    print("TEST 8: Rear Vent Bernoulli Orifice Check (Delta P = +3 Pa and -3 Pa)")
    print("=" * 78)

    temp_c = 20.0
    t_k = temp_c + 273.15
    rho0 = CFD_P_ATM / (CFD_R_AIR * t_k)
    dp = 3.0
    k_loss = 0.5
    u_torr = math.sqrt(2.0 * dp / rho0)
    u_loss = math.sqrt(2.0 * dp / (rho0 * (1.0 + k_loss)))

    details: List[str] = []

    # Positive pressurization: Delta P = +3 Pa
    with CFD3DSolver(
        grid="fast",
        temp_in_c=temp_c,
        temp_out_c=temp_c,
        win_hinge_mode=HINGE_LEFT,
        win_angle_deg=90.0,
        wind_speed=0.0,
        vent_enabled=1,
        delta_p_pa=dp,
        vent_k_loss=k_loss,
        smoke_enabled=0,
        smoke_rate=0.0,
    ) as sim:
        g = sim.geom_info()
        h = g["h"]
        vent_area = (g["j_v1"] - g["j_v0"]) * (g["k_v1"] - g["k_v0"]) * h * h
        max_mass_err = 0.0
        max_div_pos = 0.0
        max_spd_pos = 0.0
        while True:
            m = sim.step()
            max_mass_err = max(max_mass_err, abs(m["q_net"] - m["q_vent"]))
            max_div_pos = max(max_div_pos, m["max_div"])
            max_spd_pos = max(max_spd_pos, m["max_speed"])
            if m["sim_time"] >= 6.0:
                break

        u = sim.field_view("u")
        u_vent_faces = u[g["k_v0"]:g["k_v1"], g["j_v0"]:g["j_v1"], 0]
        u_vent_mean = float(np.mean(u_vent_faces))
        u_vent_max = float(np.max(u_vent_faces))
        u_win_faces = u[g["k_w0"]:g["k_w1"], g["j_w0"]:g["j_w1"], g["ix_wall"]]
        u_win_mean = float(np.mean(u_win_faces))
        fin_pos, _ = check_finite_fields(sim)

        ok_pos = (
            fin_pos
            and (max_mass_err < 1e-5)
            and (m["q_out"] > 0.5)
            and (m["q_in"] < 0.05)
            and (0.2 * u_loss < u_vent_mean <= u_loss * 1.05)
            and (u_win_mean <= u_torr * 1.05)
            and (max_spd_pos <= u_torr * 1.35)
            and (max_div_pos < 1e-4)
        )
        line_pos = (
            f"  [+3 Pa] t={m['sim_time']:.2f}s (steps={m['step_count']}): "
            f"Q_vent={m['q_vent']:.4f}, Q_net={m['q_net']:.4f} m3/s (max|Q_net-Q_vent|={max_mass_err:.2e}), "
            f"u_vent_mean={u_vent_mean:.3f} (max={u_vent_max:.3f}), u_win_mean={u_win_mean:.3f}, "
            f"max_u={max_spd_pos:.3f} vs u_Torr={u_torr:.3f} (u_loss={u_loss:.3f}) m/s, "
            f"vent_area={vent_area:.2f} m2, div={max_div_pos:.2e}"
        )
        print(line_pos)
        details.append(line_pos)

    # Negative pressurization (suction): Delta P = -3 Pa
    with CFD3DSolver(
        grid="fast",
        temp_in_c=temp_c,
        temp_out_c=temp_c,
        win_hinge_mode=HINGE_LEFT,
        win_angle_deg=90.0,
        wind_speed=0.0,
        vent_enabled=1,
        delta_p_pa=-dp,
        vent_k_loss=k_loss,
        smoke_enabled=0,
        smoke_rate=0.0,
    ) as sim:
        max_mass_err_neg = 0.0
        max_div_neg = 0.0
        max_spd_neg = 0.0
        while True:
            m = sim.step()
            max_mass_err_neg = max(max_mass_err_neg, abs(m["q_net"] - m["q_vent"]))
            max_div_neg = max(max_div_neg, m["max_div"])
            max_spd_neg = max(max_spd_neg, m["max_speed"])
            if m["sim_time"] >= 4.0:
                break
        fin_neg, _ = check_finite_fields(sim)
        ok_neg = (
            fin_neg
            and (max_mass_err_neg < 1e-5)
            and (m["q_in"] > 0.5)
            and (m["q_vent"] < -0.5)
            and (max_spd_neg <= u_torr * 1.45)
            and (max_div_neg < 1e-4)
        )
        line_neg = (
            f"  [-3 Pa] t={m['sim_time']:.2f}s (steps={m['step_count']}): "
            f"Q_vent={m['q_vent']:.4f}, Q_net={m['q_net']:.4f} m3/s (max|Q_net-Q_vent|={max_mass_err_neg:.2e}), "
            f"max_u={max_spd_neg:.3f} vs u_Torr={u_torr:.3f} m/s, div={max_div_neg:.2e}"
        )
        print(line_neg)
        details.append(line_neg)

    passed = ok_pos and ok_neg
    summary = (
        f"+3Pa: Q_vent={details[0].split('Q_vent=')[1].split(',')[0]}, |dQ|={max_mass_err:.2e}, max_u={max_spd_pos:.3f}/{u_torr:.3f} m/s | "
        f"-3Pa: |dQ|={max_mass_err_neg:.2e}, max_u={max_spd_neg:.3f}/{u_torr:.3f} m/s"
    )
    print(f"-> {'PASS' if passed else 'FAIL'}: {summary}")
    return TestResult("8. Rear Vent Bernoulli Orifice", passed, summary, details)


# =============================================================================
# BENCHMARK: Performance (ms/step, Mcells/sec)
# =============================================================================
def run_benchmarks() -> List[str]:
    print("\n" + "=" * 78)
    print("BENCHMARK: Solver Performance (ms/step, Mcells/sec)")
    print("=" * 78)
    lines: List[str] = []
    for gname in ("fast", "default"):
        with CFD3DSolver(grid=gname, preset="cold_day") as sim:
            g = sim.geom_info()
            ncells = g["nx"] * g["ny"] * g["nz"]
            # Warm up 10 steps
            sim.step_n(10)
            ms_list = []
            iters_list = []
            for _ in range(30):
                m = sim.step()
                ms_list.append(m["step_ms"])
                iters_list.append(m["pcg_iters"])
            avg_ms = float(np.mean(ms_list))
            med_ms = float(np.median(ms_list))
            mcells_s = (ncells * 1e-6) / (avg_ms * 1e-3)
            line = (
                f"  {gname:7s} ({g['nx']}x{g['ny']}x{g['nz']} = {ncells:,} cells): "
                f"avg={avg_ms:.2f} ms/step (med={med_ms:.2f} ms), "
                f"{mcells_s:.2f} Mcells/s, avg_pcg={np.mean(iters_list):.1f}"
            )
            print(line)
            lines.append(line)
    return lines


def main() -> int:
    t_all = time.perf_counter()
    results = [
        test_1_ctypes_abi(),
        test_2_quiescent_equilibrium(),
        test_3_closed_hydrostatic(),
        test_4_smoke_conservation(),
        test_5_sash_impermeability(),
        test_6_linden_buoyancy_exchange(),
        test_7_long_horizon_stability(),
        test_8_rear_vent_bernoulli(),
    ]
    bench = run_benchmarks()

    print("\n" + "=" * 78)
    print(f"V&V SUMMARY (Total wall time: {time.perf_counter() - t_all:.1f} s)")
    print("=" * 78)
    n_pass = 0
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        if r.passed:
            n_pass += 1
        print(f"[{tag}] {r.name}: {r.summary}")
    print("-" * 78)
    print(f"PASSED {n_pass}/{len(results)} tests")
    for b in bench:
        print(b)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
