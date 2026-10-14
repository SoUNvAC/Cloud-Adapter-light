"""Read-only hardware probe and non-installing dependency resolution for Phase111.

Writes only beneath the explicitly selected project-owned output directory.
Never imports models, opens imagery/labels, or installs packages.
"""
import argparse
import base64
import datetime
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REQUIREMENTS = ["gdown>=5.1.0", "huggingface-hub>=0.28.1", "platformdirs>=4.0.0",
                "rasterio>=1.3", "safetensors>=0.3.0",
                "segmentation-models-pytorch>=0.5.0", "timm>=0.9", "torch>=2.3.0",
                "tqdm>=4.0", "fastai>=2.7"]


def run(cmd, timeout=45):
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{cmd[0]} exit {result.returncode}: {result.stderr[-2000:]}")
    return result.stdout.strip()


def save(out, name, obj):
    (out / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    assert out.is_relative_to(Path("/home/scv/shared")), out
    out.mkdir(parents=True, exist_ok=False)
    ps = "Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,Size,FreeSpace,FileSystem | ConvertTo-Json -Compress"
    encoded = base64.b64encode(ps.encode("utf-16-le")).decode()
    volumes = json.loads(run(["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                              "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]))
    host = next(v for v in volumes if v["DeviceID"] == "I:")
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.startswith(("MemTotal:", "MemAvailable:"))}
    gpu = run(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,memory.total,memory.free",
               "--format=csv,noheader,nounits"])
    installed = {}
    for dist in md.distributions():
        name = dist.metadata.get("Name", "")
        if name:
            installed[name.lower().replace("_", "-")] = {
                "version": dist.version, "site_path": str(dist.locate_file("")),
                "metadata_path": str(getattr(dist, "_path", "unknown"))}
    save(out, "RESOURCE_PROBE.json", {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "physical_volume": host,
        "volume_mapping": "Conservative Phase110 Debian registry mapping to I; not WSL ext4 free",
        "ram": memory, "gpu_csv": gpu, "wsl_disk_free_bytes": shutil.disk_usage(out).free,
        "python": sys.version, "python_executable": sys.executable,
        "original_environment_installation": False})
    save(out, "EXISTING_PACKAGES.json", installed)
    if host["FreeSpace"] < 4000000000:
        save(out, "GATE_STATUS.json", {"status": "stop_host_reserve", "installation": False})
        return
    # PIP dry-run resolves latest compatible versions without modifying any installed package.
    # All temporary metadata is constrained to this new shared directory.
    temp = out / "tmp"
    temp.mkdir()
    env = dict(os.environ, TMPDIR=str(temp), PIP_DISABLE_PIP_VERSION_CHECK="1",
               PIP_NO_CACHE_DIR="1", PIP_CONFIG_FILE=os.devnull)
    cmd = [sys.executable, "-m", "pip", "--isolated", "install", "--dry-run",
           "--ignore-installed", "--only-binary=:all:", "--no-cache-dir",
           "--index-url", "https://pypi.org/simple", "--timeout", "20", "--retries", "0",
           "--report", str(out / "PIP_RESOLUTION.json"), *REQUIREMENTS]
    save(out, "RESOLUTION_REQUEST.json", {"command": cmd, "requirements": REQUIREMENTS,
         "installation": False, "existing_environment_read_only": True,
         "timeout_seconds": 240, "metadata_workspace_cap_bytes": 50000000})
    started = time.monotonic()
    reason = None
    with (out / "resolver.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
            if size > 50000000 or time.monotonic() - started > 240:
                reason = "metadata_workspace_cap" if size > 50000000 else "resolver_timeout"
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(1)
    save(out, "GATE_STATUS.json", {"status": reason or ("resolution_available_not_yet_priced"
         if proc.returncode == 0 else "resolution_failed"), "resolver_exit_code": proc.returncode,
         "elapsed_seconds": time.monotonic() - started, "installation": False,
         "model_or_band_download": False, "model_calls": 0})
    print((out / "GATE_STATUS.json").read_text())


if __name__ == "__main__":
    main()
