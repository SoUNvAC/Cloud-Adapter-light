import argparse
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys

from phase30_jetson_runtime import clocks_are_locked, cuda_result, load_cudart


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit the frozen Jetson Orin Nano Phase 30 environment."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--required-device-substring", default="NVIDIA Jetson Orin Nano"
    )
    parser.add_argument("--required-power-mode-substring", default="MAXN")
    parser.add_argument("--min-memory-mib", type=float, default=7000.0)
    return parser.parse_args()


def read_device_model():
    path = Path("/proc/device-tree/model")
    if not path.is_file():
        return ""
    return path.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace")


def command_output(command):
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
    }


def memory_mib():
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024.0
    raise RuntimeError("MemTotal is missing from /proc/meminfo")


def read_l4t_release():
    path = Path("/etc/nv_tegra_release")
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def is_jetpack_6_1_or_later(l4t_release):
    match = re.search(r"#\s*R(\d+).*?REVISION:\s*(\d+)(?:\.(\d+))?", l4t_release)
    if match is None:
        return False
    version = tuple(int(value or 0) for value in match.groups())
    return version >= (36, 4, 0) and version < (37, 0, 0)


def main():
    args = parse_args()
    if args.min_memory_mib <= 0:
        raise ValueError("min-memory-mib must be positive")
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError("TensorRT Python bindings are unavailable") from error
    cudart = load_cudart()
    runtime_version = cuda_result(cudart.cudaRuntimeGetVersion(), "runtime version")
    device = read_device_model()
    architecture = platform.machine()
    total_memory = memory_mib()
    l4t_release = read_l4t_release()
    nvpmodel = command_output(["nvpmodel", "-q", "--verbose"])
    jetson_clocks = command_output(["jetson_clocks", "--show"])
    tegrastats_path = shutil.which("tegrastats")
    git = command_output(["git", "rev-parse", "HEAD"])
    clocks_output = jetson_clocks["stdout"] + "\n" + jetson_clocks["stderr"]
    gates = {
        "jetson_orin_nano": args.required_device_substring.lower() in device.lower(),
        "aarch64": architecture in ("aarch64", "arm64"),
        "memory_capacity": total_memory >= args.min_memory_mib,
        "jetpack_6_1_or_later": is_jetpack_6_1_or_later(l4t_release),
        "tensorrt_10": trt.__version__.startswith("10."),
        "cuda_12": 12000 <= int(runtime_version) < 13000,
        "nvpmodel_query": nvpmodel["returncode"] == 0,
        "required_power_mode": args.required_power_mode_substring.lower()
        in nvpmodel["stdout"].lower(),
        "jetson_clocks_query": jetson_clocks["returncode"] == 0,
        "clocks_locked": clocks_are_locked(clocks_output),
        "tegrastats_available": tegrastats_path is not None,
        "git_revision_available": git["returncode"] == 0,
        "pytorch_not_imported": "torch" not in sys.modules,
    }
    result = {
        "phase": 30,
        "device_model": device,
        "architecture": architecture,
        "memory_mib": total_memory,
        "l4t_release": l4t_release,
        "python": sys.version,
        "tensorrt_version": trt.__version__,
        "cuda_runtime_version": int(runtime_version),
        "nvpmodel": nvpmodel,
        "jetson_clocks": jetson_clocks,
        "tegrastats_path": tegrastats_path,
        "git": git,
        "gates": gates,
        "passed": all(gates.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
