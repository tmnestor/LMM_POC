#!/usr/bin/env python3
"""
vLLM / PyTorch / CUDA diagnostic script.

Run on the target production machine BEFORE installing vLLM to confirm
that the environment is compatible. Does not require vLLM to be installed.
Explicitly checks that Flash-Attention is NOT required (vLLM falls back to
xFormers / Triton / PyTorch-native attention backends when FA is absent).

Usage:
    python vllm_diagnostic.py
    python vllm_diagnostic.py --json     # machine-readable output
    python vllm_diagnostic.py --target 0.11.2   # check against a specific vLLM version
    python vllm_diagnostic.py --expected-gpus 4 # fail if fewer than N GPUs visible

No third-party dependencies required for the basic checks. PyTorch is
imported only if it is already installed; the script still runs without it.

Compatibility matrix is sourced from the actual PyPI wheel METADATA of each
vLLM release (run `unzip -p <wheel>.whl '*/METADATA' | grep -i requires-dist`
to verify). Update VLLM_MATRIX below when bumping vLLM.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Compatibility matrix
# ---------------------------------------------------------------------------
# Update this table when you bump vLLM. Source of truth is the actual
# Requires-Dist lines inside the wheel METADATA on PyPI:
#   pip download vllm==<ver> --no-deps --platform manylinux_2_28_x86_64 \
#       --python-version 3.12 --implementation cp --abi cp312 \
#       --only-binary=:all: -d /tmp/vllm-check
#   unzip -p /tmp/vllm-check/vllm-*.whl '*/METADATA' \
#       | grep -i -E '^(requires-dist:|requires-python)'
VLLM_MATRIX: dict[str, dict[str, Any]] = {
    "0.11.2": {
        # Verified from PyPI wheel METADATA:
        #   Requires-Dist: torch==2.9.0
        #   Requires-Dist: torchaudio==2.9.0
        #   Requires-Dist: torchvision==0.24.0
        #   Requires-Dist: xformers==0.0.33.post1 ; platform_system == "Linux"
        #                                           and platform_machine == "x86_64"
        #   Requires-Dist: flashinfer-python==0.5.2
        #   Requires-Python: <3.14,>=3.10
        "torch": "2.9.0",
        "torchvision": "0.24.0",
        "torchaudio": "2.9.0",
        "cuda": ["12.8", "12.6", "12.4"],  # torch 2.9.0 ships cu128/cu126/cu124
        "python": (3, 10, 3, 13),  # min_major, min_minor, max_major, max_minor
        "transformers_min": "4.46.0",
        "min_compute_capability": 7.0,
        "hard_deps": {
            "xformers": "0.0.33.post1",
            "flashinfer-python": "0.5.2",
        },
        "flash_attn_required": False,  # FA is lazily imported; never required
    },
    "0.8.0": {
        "torch": "2.6.0",
        "cuda": ["12.4", "12.1"],
        "python": (3, 9, 3, 12),
        "transformers_min": "4.48.0",
        "min_compute_capability": 7.0,
        "hard_deps": {},
        "flash_attn_required": False,
    },
    "0.7.3": {
        "torch": "2.5.1",
        "cuda": ["12.4", "12.1"],
        "python": (3, 9, 3, 12),
        "transformers_min": "4.48.0",
        "min_compute_capability": 7.0,
        "hard_deps": {},
        "flash_attn_required": False,
    },
    "0.6.6": {
        "torch": "2.5.1",
        "cuda": ["12.4", "12.1"],
        "python": (3, 9, 3, 12),
        "transformers_min": "4.45.0",
        "min_compute_capability": 7.0,
        "hard_deps": {},
        "flash_attn_required": False,
    },
}

DEFAULT_TARGET = "0.11.2"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    status: str  # "ok" | "warn" | "fail" | "info"
    detail: str


@dataclass
class Report:
    target_vllm: str
    requirements: dict[str, Any]
    system: dict[str, Any] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name=name, status=status, detail=detail))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, text=True, timeout=10
        )
        return out.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        return f"<error: {e}>"


def _parse_version(v: str) -> tuple[int, ...]:
    nums: list[int] = []
    for piece in v.split("+")[0].split("."):
        try:
            nums.append(int(piece))
        except ValueError:
            break
    return tuple(nums)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_python(report: Report) -> None:
    v = sys.version_info
    report.system["python"] = f"{v.major}.{v.minor}.{v.micro}"
    lo_maj, lo_min, hi_maj, hi_min = report.requirements["python"]
    ok = (v.major, v.minor) >= (lo_maj, lo_min) and (v.major, v.minor) <= (
        hi_maj,
        hi_min,
    )
    detail = (
        f"found {v.major}.{v.minor}.{v.micro}; need {lo_maj}.{lo_min}–{hi_maj}.{hi_min}"
    )
    report.add("python_version", "ok" if ok else "fail", detail)


def check_os(report: Report) -> None:
    report.system["platform"] = platform.platform()
    report.system["machine"] = platform.machine()
    is_linux = sys.platform.startswith("linux")
    detail = f"{platform.system()} {platform.release()} ({platform.machine()})"
    report.add(
        "operating_system",
        "ok" if is_linux else "warn",
        detail + ("" if is_linux else " — vLLM officially supports Linux only"),
    )


def check_glibc(report: Report) -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        libc_ver = platform.libc_ver()
        report.system["libc"] = f"{libc_ver[0]} {libc_ver[1]}"
        report.add("glibc", "info", f"{libc_ver[0]} {libc_ver[1]}")
    except Exception as e:
        report.add("glibc", "warn", f"could not detect: {e}")


def check_cpu_memory(report: Report) -> None:
    cores_logical = os.cpu_count()
    cores_affinity: int | None
    try:
        cores_affinity = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        cores_affinity = cores_logical
    report.system["cpu_cores_logical"] = cores_logical
    report.system["cpu_cores_available"] = cores_affinity

    mem_gb = None
    try:
        if sys.platform.startswith("linux"):
            with Path("/proc/meminfo").open() as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_gb = int(line.split()[1]) / (1024 * 1024)
                        break
    except Exception:
        pass
    report.system["ram_gb"] = round(mem_gb, 1) if mem_gb else None
    report.add(
        "cpu_memory",
        "info",
        f"{cores_affinity}/{cores_logical} cores, {report.system['ram_gb']} GB RAM",
    )


def check_nvidia_smi(report: Report) -> None:
    if not shutil.which("nvidia-smi"):
        report.add(
            "nvidia_smi", "fail", "nvidia-smi not found in PATH — no NVIDIA driver?"
        )
        return
    driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    cuda_runtime = _run(
        ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"]
    )
    gpus = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total",
            "--format=csv,noheader",
        ]
    )
    report.system["nvidia_driver"] = driver.splitlines()[0] if driver else None
    report.system["cuda_driver_max"] = (
        cuda_runtime.splitlines()[0] if cuda_runtime else None
    )
    report.system["gpus"] = gpus.splitlines() if gpus else []
    report.add(
        "nvidia_driver",
        "ok" if driver else "fail",
        f"driver={driver}, cuda(driver-reported)={cuda_runtime}",
    )
    for line in report.system["gpus"]:
        report.add("gpu", "info", line)
        # Compute capability check
        try:
            parts = [p.strip() for p in line.split(",")]
            cc = float(parts[1])
            need = report.requirements["min_compute_capability"]
            status = "ok" if cc >= need else "fail"
            report.add(
                f"gpu_compute_cap[{parts[0]}]",
                status,
                f"compute capability {cc} (need ≥ {need})",
            )
        except (IndexError, ValueError):
            pass


def check_cuda_toolkit(report: Report) -> None:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        report.add(
            "nvcc",
            "info",
            "nvcc not in PATH (only required for source builds; pip wheels are fine)",
        )
        return
    out = _run(["nvcc", "--version"])
    report.system["nvcc"] = out
    # Try to extract release X.Y
    cuda_ver = None
    for tok in out.split():
        if tok.startswith("V") and "." in tok:
            cuda_ver = tok.lstrip("V").split(",")[0]
            break
        if tok == "release":
            idx = out.split().index(tok)
            try:
                cuda_ver = out.split()[idx + 1].rstrip(",")
            except IndexError:
                pass
    report.system["cuda_toolkit"] = cuda_ver
    if cuda_ver:
        major_minor = ".".join(cuda_ver.split(".")[:2])
        ok = major_minor in report.requirements["cuda"]
        report.add(
            "cuda_toolkit",
            "ok" if ok else "warn",
            f"nvcc reports CUDA {cuda_ver}; vLLM wheels target {report.requirements['cuda']}",
        )
    else:
        report.add("cuda_toolkit", "warn", f"could not parse: {out}")


def check_torch(report: Report, expected_gpus: int | None) -> None:
    try:
        import torch  # type: ignore
    except ImportError:
        report.add(
            "torch",
            "info",
            f"torch not installed — vLLM will pull torch=={report.requirements['torch']}",
        )
        return

    report.system["torch"] = torch.__version__
    report.system["torch_cuda"] = torch.version.cuda
    report.system["torch_cudnn"] = (
        torch.backends.cudnn.version() if torch.cuda.is_available() else None
    )

    # torch version (exact pin — vLLM uses ==)
    want = report.requirements["torch"]
    have = torch.__version__.split("+")[0]
    status = "ok" if have == want else "fail"
    report.add(
        "torch_version",
        status,
        f"installed {torch.__version__}; vLLM {report.target_vllm} pins torch=={want}",
    )

    # torchvision / torchaudio (also exact pins for recent vLLM)
    for pkg_name, req_key in [
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
    ]:
        want_v = report.requirements.get(req_key)
        if not want_v:
            continue
        try:
            mod = __import__(pkg_name)
            have_v = mod.__version__.split("+")[0]
            report.system[pkg_name] = mod.__version__
            status = "ok" if have_v == want_v else "fail"
            report.add(
                f"{pkg_name}_version",
                status,
                f"installed {mod.__version__}; vLLM pins {pkg_name}=={want_v}",
            )
        except ImportError:
            report.add(
                f"{pkg_name}_version",
                "warn",
                f"not installed; vLLM pins {pkg_name}=={want_v}",
            )

    # C++ ABI flag — matters for prebuilt kernel wheels (flash-attn, xformers)
    try:
        cxx11 = torch._C._GLIBCXX_USE_CXX11_ABI
        report.system["torch_cxx11_abi"] = cxx11
        report.add(
            "torch_cxx11_abi",
            "info",
            f"_GLIBCXX_USE_CXX11_ABI = {cxx11} "
            f"(must match any out-of-tree CUDA kernel wheel)",
        )
    except AttributeError:
        pass

    if torch.cuda.is_available():
        dev_count = torch.cuda.device_count()
        report.system["torch_device_count"] = dev_count
        report.add(
            "torch_cuda_available",
            "ok",
            f"torch built with CUDA {torch.version.cuda}, "
            f"{dev_count} device(s) visible",
        )
        for i in range(dev_count):
            props = torch.cuda.get_device_properties(i)
            cc = float(f"{props.major}.{props.minor}")
            need = report.requirements["min_compute_capability"]
            status = "ok" if cc >= need else "fail"
            report.add(
                f"torch_gpu[{i}]",
                status,
                f"{props.name}, cc={cc}, {round(props.total_memory / 1024**3, 1)} GB",
            )
        # GPU count sanity check
        if expected_gpus is not None:
            gpu_status = "ok" if dev_count >= expected_gpus else "fail"
            report.add(
                "gpu_count_expected",
                gpu_status,
                f"expected ≥{expected_gpus} GPUs, torch sees {dev_count}",
            )
        # torch CUDA build vs vLLM's expected builds
        torch_cuda_mm = ".".join((torch.version.cuda or "").split(".")[:2])
        cuda_ok = torch_cuda_mm in report.requirements["cuda"]
        report.add(
            "torch_cuda_match",
            "ok" if cuda_ok else "warn",
            f"torch built for CUDA {torch_cuda_mm}; "
            f"vLLM wheels typically target {report.requirements['cuda']}",
        )
    else:
        report.add(
            "torch_cuda_available",
            "fail",
            "torch.cuda.is_available() == False — install a CUDA build of torch",
        )


def check_gpu_visibility(report: Report, expected_gpus: int | None) -> None:
    """Cross-check env vars and nvidia-smi -L with expected GPU count."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    report.system["CUDA_VISIBLE_DEVICES"] = cvd
    if cvd is None:
        report.add(
            "cuda_visible_devices",
            "info",
            "CUDA_VISIBLE_DEVICES not set (all GPUs visible by default)",
        )
    else:
        report.add("cuda_visible_devices", "info", f"CUDA_VISIBLE_DEVICES={cvd!r}")

    # nvidia-smi -L is the ground truth for how many physical GPUs exist
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "-L"])
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]
        report.system["nvidia_smi_gpu_count"] = len(lines)
        for ln in lines:
            report.add("nvidia_smi_gpu", "info", ln.strip())
        if expected_gpus is not None and lines:
            status = "ok" if len(lines) >= expected_gpus else "fail"
            report.add(
                "nvidia_smi_gpu_count",
                status,
                f"nvidia-smi -L reports {len(lines)} GPU(s); expected ≥{expected_gpus}",
            )


def check_flash_attention_absent(report: Report) -> None:
    """Confirm Flash-Attention is NOT installed and document the fallback."""
    try:
        import flash_attn  # type: ignore

        report.add(
            "flash_attention",
            "warn",
            f"flash_attn {flash_attn.__version__} IS installed — "
            "user requested it NOT be present. Uninstall with "
            "`pip uninstall flash-attn` if undesired.",
        )
    except ImportError:
        report.add(
            "flash_attention",
            "ok",
            "flash_attn not installed (as required). vLLM will use "
            "xformers / flashinfer / Triton / PyTorch SDPA. Set "
            "VLLM_ATTENTION_BACKEND=XFORMERS (or TORCH_SDPA) to be explicit.",
        )
    # Also check whether the current backend env var is pinned
    backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    if backend:
        status = (
            "ok"
            if backend.upper()
            in {"XFORMERS", "TORCH_SDPA", "FLASHINFER", "TRITON_ATTN_VLLM_V1"}
            else "warn"
        )
        report.add("attention_backend_env", status, f"VLLM_ATTENTION_BACKEND={backend}")
    else:
        report.add(
            "attention_backend_env",
            "info",
            "VLLM_ATTENTION_BACKEND unset — recommend XFORMERS on A10G "
            "when flash-attn is unavailable",
        )


def check_hard_deps(report: Report) -> None:
    """Verify the exact-pin transitive dependencies vLLM ships with."""
    hard = report.requirements.get("hard_deps") or {}
    if not hard:
        return
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return
    for pkg, want in hard.items():
        try:
            have = version(pkg)
            status = "ok" if have == want else "warn"
            report.add(
                f"hard_dep:{pkg}", status, f"installed {have}; vLLM pins {pkg}=={want}"
            )
        except PackageNotFoundError:
            report.add(
                f"hard_dep:{pkg}",
                "info",
                f"not installed; vLLM will pull {pkg}=={want}",
            )


def check_other_packages(report: Report) -> None:
    """Look for vLLM, transformers, xformers, triton, ray, etc."""
    packages = [
        "vllm",
        "transformers",
        "tokenizers",
        "xformers",
        "flashinfer-python",
        "triton",
        "ray",
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "sentencepiece",
        "outlines",
        "xgrammar",
        "flash-attn",
    ]
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return

    for pkg in packages:
        try:
            v = version(pkg)
            report.add(f"pkg:{pkg}", "info", v)
            report.system.setdefault("packages", {})[pkg] = v
        except PackageNotFoundError:
            report.add(f"pkg:{pkg}", "info", "not installed")


def check_disk_space(report: Report) -> None:
    try:
        usage = shutil.disk_usage(Path.home())
        free_gb = usage.free / 1024**3
        report.system["home_free_gb"] = round(free_gb, 1)
        status = "ok" if free_gb >= 50 else "warn"
        report.add(
            "disk_space",
            status,
            f"{round(free_gb, 1)} GB free in $HOME (≥50 GB recommended for model weights)",
        )
    except Exception as e:
        report.add("disk_space", "warn", f"could not check: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_report(target: str, expected_gpus: int | None = None) -> Report:
    if target not in VLLM_MATRIX:
        raise SystemExit(
            f"Unknown vLLM target {target!r}. Known: {sorted(VLLM_MATRIX)}"
        )
    report = Report(target_vllm=target, requirements=VLLM_MATRIX[target])
    check_python(report)
    check_os(report)
    check_glibc(report)
    check_cpu_memory(report)
    check_disk_space(report)
    check_nvidia_smi(report)
    check_cuda_toolkit(report)
    check_torch(report, expected_gpus)
    check_gpu_visibility(report, expected_gpus)
    check_flash_attention_absent(report)
    check_hard_deps(report)
    check_other_packages(report)
    return report


COLORS = {
    "ok": "\033[92m",  # green
    "warn": "\033[93m",  # yellow
    "fail": "\033[91m",  # red
    "info": "\033[94m",  # blue
}
RESET = "\033[0m"


def print_human(report: Report) -> int:
    use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    def tag(status: str) -> str:
        label = f"[{status.upper():4}]"
        if use_color:
            return f"{COLORS.get(status, '')}{label}{RESET}"
        return label

    print("=" * 72)
    print(f" vLLM environment diagnostic — target vLLM {report.target_vllm}")
    print("=" * 72)
    req = report.requirements
    print(f" Required torch:       {req['torch']}")
    if req.get("torchvision"):
        print(f" Required torchvision: {req['torchvision']}")
    if req.get("torchaudio"):
        print(f" Required torchaudio:  {req['torchaudio']}")
    print(f" Required CUDA:        {' or '.join(req['cuda'])}")
    print(
        f" Python range:         {req['python'][0]}.{req['python'][1]}"
        f"–{req['python'][2]}.{req['python'][3]}"
    )
    print(f" Min GPU compute:      {req['min_compute_capability']}")
    print(f" transformers ≥        {req['transformers_min']}")
    if req.get("hard_deps"):
        print(
            " Pinned hard deps:     "
            + ", ".join(f"{k}=={v}" for k, v in req["hard_deps"].items())
        )
    print()
    print(" --- System ---")
    for k, v in report.system.items():
        if k == "packages":
            continue
        print(f"   {k}: {v}")
    if "packages" in report.system:
        print(" --- Detected packages ---")
        for k, v in report.system["packages"].items():
            print(f"   {k}: {v}")
    print()
    print(" --- Checks ---")
    for c in report.checks:
        print(f" {tag(c.status)} {c.name}: {c.detail}")
    print()

    fails = [c for c in report.checks if c.status == "fail"]
    warns = [c for c in report.checks if c.status == "warn"]
    print(
        f" Summary: {len(fails)} fail, {len(warns)} warn, "
        f"{len([c for c in report.checks if c.status == 'ok'])} ok"
    )
    if fails:
        print(" RESULT: environment is NOT ready for vLLM.")
        return 2
    if warns:
        print(" RESULT: environment is usable but has warnings.")
        return 1
    print(" RESULT: environment looks good for vLLM.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="vLLM / PyTorch / CUDA diagnostic")
    p.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"target vLLM version (default: {DEFAULT_TARGET}). "
        f"Known: {sorted(VLLM_MATRIX)}",
    )
    p.add_argument(
        "--expected-gpus",
        type=int,
        default=None,
        help="fail if fewer than this many GPUs are visible "
        "(cross-checks torch.cuda.device_count and nvidia-smi -L)",
    )
    p.add_argument(
        "--json", action="store_true", help="emit JSON instead of human output"
    )
    args = p.parse_args()

    report = build_report(args.target, expected_gpus=args.expected_gpus)

    if args.json:
        payload = {
            "target_vllm": report.target_vllm,
            "requirements": report.requirements,
            "system": report.system,
            "checks": [asdict(c) for c in report.checks],
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0 if not any(c.status == "fail" for c in report.checks) else 2

    return print_human(report)


if __name__ == "__main__":
    raise SystemExit(main())
