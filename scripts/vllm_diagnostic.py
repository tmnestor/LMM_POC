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
    python vllm_diagnostic.py --target 0.8.0   # check against a specific vLLM version

No third-party dependencies required for the basic checks. PyTorch is
imported only if it is already installed; the script still runs without it.
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
from typing import Any

# ---------------------------------------------------------------------------
# Compatibility matrix
# ---------------------------------------------------------------------------
# Update this table when you bump vLLM. Source of truth is the
# requirements/*.txt files in the vLLM repo at the matching git tag.
VLLM_MATRIX: dict[str, dict[str, Any]] = {
    "0.8.0": {
        "torch": "2.6.0",
        "cuda": ["12.4", "12.1"],
        "python": (3, 9, 3, 12),  # min_major, min_minor, max_major, max_minor
        "transformers_min": "4.48.0",
        "min_compute_capability": 7.0,
    },
    "0.7.3": {
        "torch": "2.5.1",
        "cuda": ["12.4", "12.1"],
        "python": (3, 9, 3, 12),
        "transformers_min": "4.48.0",
        "min_compute_capability": 7.0,
    },
    "0.6.6": {
        "torch": "2.5.1",
        "cuda": ["12.4", "12.1"],
        "python": (3, 9, 3, 12),
        "transformers_min": "4.45.0",
        "min_compute_capability": 7.0,
    },
}

DEFAULT_TARGET = "0.8.0"


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
    try:
        cores_affinity = len(os.sched_getaffinity(0))  # Linux
    except AttributeError:
        cores_affinity = cores_logical
    report.system["cpu_cores_logical"] = cores_logical
    report.system["cpu_cores_available"] = cores_affinity

    mem_gb = None
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
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


def check_torch(report: Report) -> None:
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

    want = report.requirements["torch"]
    have = torch.__version__.split("+")[0]
    status = "ok" if have == want else "warn"
    report.add(
        "torch_version",
        status,
        f"installed {torch.__version__}; vLLM {report.target_vllm} pins torch=={want}",
    )

    if torch.cuda.is_available():
        report.add(
            "torch_cuda_available",
            "ok",
            f"torch built with CUDA {torch.version.cuda}, "
            f"{torch.cuda.device_count()} device(s) visible",
        )
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cc = float(f"{props.major}.{props.minor}")
            need = report.requirements["min_compute_capability"]
            status = "ok" if cc >= need else "fail"
            report.add(
                f"torch_gpu[{i}]",
                status,
                f"{props.name}, cc={cc}, {round(props.total_memory / 1024**3, 1)} GB",
            )
        # Check torch CUDA against required
        torch_cuda_mm = ".".join((torch.version.cuda or "").split(".")[:2])
        cuda_ok = torch_cuda_mm in report.requirements["cuda"]
        report.add(
            "torch_cuda_match",
            "ok" if cuda_ok else "warn",
            f"torch built for CUDA {torch_cuda_mm}; "
            f"vLLM expects one of {report.requirements['cuda']}",
        )
    else:
        report.add(
            "torch_cuda_available",
            "fail",
            "torch.cuda.is_available() == False — install a CUDA build of torch",
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
            "flash_attn not installed (as required). vLLM will fall back to "
            "xFormers / Triton / PyTorch SDPA. Set "
            "VLLM_ATTENTION_BACKEND=XFORMERS (or TORCH_SDPA) to be explicit.",
        )


def check_other_packages(report: Report) -> None:
    """Look for vLLM, transformers, xformers, triton, ray, etc."""
    packages = [
        "vllm",
        "transformers",
        "tokenizers",
        "xformers",
        "triton",
        "ray",
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "sentencepiece",
        "outlines",
        "xgrammar",
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
        usage = shutil.disk_usage(os.path.expanduser("~"))
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
def build_report(target: str) -> Report:
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
    check_torch(report)
    check_flash_attention_absent(report)
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
    print(f" Required torch:   {req['torch']}")
    print(f" Required CUDA:    {' or '.join(req['cuda'])}")
    print(
        f" Python range:     {req['python'][0]}.{req['python'][1]}"
        f"–{req['python'][2]}.{req['python'][3]}"
    )
    print(f" Min GPU compute:  {req['min_compute_capability']}")
    print(f" transformers ≥    {req['transformers_min']}")
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
        "--json", action="store_true", help="emit JSON instead of human output"
    )
    args = p.parse_args()

    report = build_report(args.target)

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
