"""Install the complete pre-resolved runtime in an isolated shared venv only.

Sequential pinned wheels are SHA-checked, installed, then their owned temporary
copy is removed. All dependencies were resolved together; final pip check is
mandatory. The original conda environment is never an installation target.
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import venv
from phase111_resource_gate import run


def host_free():
    ps = "(Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='I:'\").FreeSpace"
    encoded = base64.b64encode(ps.encode("utf-16-le")).decode()
    return int(run(["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                    "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-directory", type=Path, required=True)
    args = ap.parse_args()
    gate = args.gate_directory.resolve()
    base = gate.parent
    assert base == Path("/home/scv/shared/phase111_ocm_runtime")
    price = json.loads((gate / "complete_wheel_pricing/COMPLETE_ENVIRONMENT_PRICE.json").read_text())
    resolution = json.loads((gate / "PIP_RESOLUTION.json").read_text())
    items = {x["metadata"]["name"].lower().replace("_", "-"):x for x in resolution["install"]}
    costs = {x["name"]:x for x in price["packages"]}
    free = host_free()
    required = price["estimated_total_additional_peak_with_safety_and_inputs"]
    decision = {"host_free_before_bytes":free,"estimated_additional_peak_bytes":required,
                "reserve_bytes":4000000000,"environment_4GB_cap_overridden_by_user":True,
                "status":"admitted" if free-required>=4000000000 else "stop_host_reserve_estimate"}
    (gate / "INSTALL_RESOURCE_GATE.json").write_text(json.dumps(decision,indent=2)+"\n")
    if decision["status"] != "admitted":
        print(json.dumps(decision),flush=True)
        return
    lock = base / "INSTALL_LOCK"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    runtime = base / "runtime_cuda128"
    temporary = base / "runtime_install_tmp"
    try:
        assert not runtime.exists(), "Existing runtime must not be silently replaced"
        temporary.mkdir(exist_ok=False)
        venv.EnvBuilder(with_pip=False,symlinks=True).create(runtime)
        python = runtime / "bin/python"
        env = dict(os.environ,TMPDIR=str(temporary),PIP_NO_CACHE_DIR="1",
                   PIP_CONFIG_FILE=os.devnull,PYTHONDONTWRITEBYTECODE="1")
        ledger = base / "INSTALL_LEDGER.jsonl"
        with ledger.open("x",encoding="utf-8") as log:
            for name in price["installation_order"]:
                item=items[name]; cost=costs[name]
                assert host_free()-cost["wheel_declared_length"]-cost["uncompressed_all_entries_bytes"]>=4256000000
                filename = urllib.parse.unquote(urllib.parse.urlsplit(item["download_info"]["url"]).path.rsplit("/",1)[-1])
                wheel = temporary / filename
                assert wheel.suffix==".whl" and not wheel.exists()
                digest=hashlib.sha256(); count=0; checked=time.monotonic()
                with urllib.request.urlopen(item["download_info"]["url"],timeout=60) as response,wheel.open("xb") as target:
                    while True:
                        chunk=response.read(4*1024*1024)
                        if not chunk: break
                        count+=len(chunk)
                        assert count<=cost["wheel_declared_length"]
                        target.write(chunk); digest.update(chunk)
                        if time.monotonic()-checked>=15:
                            assert host_free()>=4256000000,"physical reserve reached"
                            checked=time.monotonic()
                assert count==cost["wheel_declared_length"]
                assert digest.hexdigest()==item["download_info"]["archive_info"]["hashes"]["sha256"]
                command=[sys.executable,"-m","pip","--isolated","--python",str(python),
                         "install","--no-deps","--no-cache-dir","--no-compile",str(wheel)]
                result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=240)
                (temporary/(name+".install.log")).write_text(result.stdout+result.stderr)
                assert result.returncode==0,(name,result.stderr)
                # Only this invocation's successfully installed wheel is removed.
                wheel.unlink()
                row={"name":name,"version":item["metadata"]["version"],"downloaded_bytes":count,
                     "sha256":digest.hexdigest(),"installed_to":str(runtime),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
                log.write(json.dumps(row)+"\n");log.flush()
                print(json.dumps(row),flush=True)
        check=subprocess.run([sys.executable,"-m","pip","--isolated","--python",str(python),"check"],
                             env=env,capture_output=True,text=True,timeout=120)
        (base/"PIP_CHECK.txt").write_text(check.stdout+check.stderr)
        assert check.returncode==0,"Final full dependency check failed"
        code="import torch,torchvision,fastai,timm,segmentation_models_pytorch as smp,rasterio,safetensors,json; print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),torchvision=torchvision.__version__,fastai=fastai.__version__,timm=timm.__version__,smp=smp.__version__,rasterio=rasterio.__version__)))"
        imports=run([str(python),"-c",code],timeout=120)
        parsed=json.loads(imports)
        assert parsed["torch"]=="2.11.0+cu128" and parsed["cuda_available"]
        parsed.update(host_free_after_bytes=host_free(),dependency_check="passed",model_calls=0)
        (base/"RUNTIME_IMPORT_ACCEPTANCE.json").write_text(json.dumps(parsed,indent=2)+"\n")
        print(json.dumps(parsed),flush=True)
    finally:
        # Keep lock even on failure to require explicit diagnosis, never auto-restart.
        if (base/"RUNTIME_IMPORT_ACCEPTANCE.json").exists(): lock.unlink()


if __name__ == "__main__":
    main()
