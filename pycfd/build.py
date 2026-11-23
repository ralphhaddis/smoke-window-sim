"""Build helper for libcfd3d.dylib using Apple Clang C99."""

from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
CSRC_DIR = ROOT_DIR / "csrc"
LIB_PATH = ROOT_DIR / "libcfd3d.dylib"

SOURCES = [
    "grid_geom.c",
    "bc.c",
    "advect_les.c",
    "poisson_mg.c",
    "step.c",
    "slices.c",
]

HEADERS = [
    "cfd3d.h",
    "cfd3d_internal.h",
]


def needs_rebuild() -> bool:
    if not LIB_PATH.exists():
        return True
    lib_mtime = LIB_PATH.stat().st_mtime
    for name in SOURCES + HEADERS:
        src = CSRC_DIR / name
        if src.exists() and src.stat().st_mtime > lib_mtime:
            return True
    return False


def build_library(force: bool = False, verbose: bool = True) -> Path:
    if not force and not needs_rebuild():
        return LIB_PATH

    cmd = [
        "clang",
        "-std=c99",
        "-O3",
        "-mcpu=apple-m3",
        "-fno-math-errno",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fPIC",
        "-shared",
        *[str(CSRC_DIR / s) for s in SOURCES],
        "-o",
        str(LIB_PATH),
        "-lm",
    ]
    if verbose:
        print("[pycfd.build] Compiling:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"Failed to compile libcfd3d.dylib (exit {res.returncode})")
    return LIB_PATH


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    path = build_library(force=force_flag, verbose=True)
    print(f"[pycfd.build] Ready: {path}")
