"""Check the budget lower bound using only Torch and its direct GPU dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
from packaging.requirements import Requirement
from price_phase111_wheels import directory_price


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-directory", type=Path, required=True)
    args = ap.parse_args()
    base = args.gate_directory
    out = base / "wheel_pricing_gpu_core"
    out.mkdir(exist_ok=False)
    resolved = json.loads((base / "PIP_RESOLUTION.json").read_text(encoding="utf-8"))
    items = {x["metadata"]["name"].lower().replace("_", "-"): x for x in resolved["install"]}
    torch = items["torch"]
    direct = {Requirement(s).name.lower().replace("_", "-") for s in torch["metadata"]["requires_dist"]}
    gpu_names = {n for n in items if n in direct and (n.startswith("nvidia-") or n == "triton")}
    gpu_names.add("torch")
    old = json.loads((base / "EXISTING_PACKAGES.json").read_text(encoding="utf-8"))
    previous = json.loads((base / "wheel_pricing_continued/ENVIRONMENT_PRICE.json").read_text(encoding="utf-8"))
    rows = []
    for row in previous["priced_packages"]:
        if row["name"] not in gpu_names:
            continue
        paths = list(base.glob("wheel_pricing*/" + row["name"] + ".central-directory.bin"))
        assert paths and any(hashlib.sha256(p.read_bytes()).hexdigest() == row["central_directory_sha256"] for p in paths)
        assert items[row["name"]]["download_info"]["url"] == row["url"]
        assert old.get(row["name"], {}).get("version") != row["version"]
        rows.append(row)
    total = sum(row["mandatory_library_payload_bytes"] for row in rows)
    for name in sorted(gpu_names - {row["name"] for row in rows}):
        if total > 4000000000:
            break
        item = items[name]
        version = item["metadata"]["version"]
        assert old.get(name, {}).get("version") != version
        row = directory_price(item["download_info"]["url"], out, name)
        row.update(version=version, existing_version=old.get(name, {}).get("version", "not_installed"),
                   wheel_declared_sha256=item["download_info"]["archive_info"]["hashes"]["sha256"])
        rows.append(row)
        total += row["mandatory_library_payload_bytes"]
        print(json.dumps({"package": name, "core_new_library_lower_bound_bytes": total}), flush=True)
    result = {"scope": "Only torch and resolved direct nvidia/triton dependencies; excludes rasterio/hf-xet and other packages",
              "environment_cap_bytes": 4000000000,
              "core_new_library_lower_bound_bytes": total,
              "gate": "STOP_ENVIRONMENT_BUDGET" if total > 4000000000 else "incomplete_core_pricing",
              "estimate_type": "Uncompressed library logical bytes declared by official wheel central directories; not actual installation measurement",
              "priced_packages": rows, "torch_direct_dependency_names": sorted(direct),
              "excluded_costs": ["all non-core dependencies", "Python/runtime files", "installation caches", "unpack temporaries"],
              "model_download_or_forward": False, "installation": False}
    (out / "ENVIRONMENT_CORE_PRICE.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
