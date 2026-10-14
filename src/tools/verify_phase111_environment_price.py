"""Independently parse saved central directories using stdlib zipfile, no network."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-directory", type=Path, required=True)
    args = ap.parse_args()
    base = args.gate_directory
    report = json.loads((base / "wheel_pricing_gpu_core/ENVIRONMENT_CORE_PRICE.json").read_text(encoding="utf-8"))
    resolution = json.loads((base / "PIP_RESOLUTION.json").read_text(encoding="utf-8"))
    items = {x["metadata"]["name"].lower().replace("_", "-"): x for x in resolution["install"]}
    results = []
    for row in report["priced_packages"]:
        paths = list(base.glob("wheel_pricing*/" + row["name"] + ".central-directory.bin"))
        matching = [p for p in paths if hashlib.sha256(p.read_bytes()).hexdigest() == row["central_directory_sha256"]]
        assert len(matching) == 1
        cd = matching[0].read_bytes()
        # zipfile.infolist only needs the central directory and an EOCD. It
        # independently interprets entries; no library payload is read/extracted.
        eocd = struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, row["wheel_entries"],
                           row["wheel_entries"], len(cd), 0, 0)
        with zipfile.ZipFile(io.BytesIO(cd + eocd)) as archive:
            entries = archive.infolist()
        assert len(entries) == row["wheel_entries"]
        libs = [p for p in entries if ".so" in Path(p.filename).name and not p.is_dir()]
        assert sum(p.file_size for p in entries) == row["uncompressed_all_entries_bytes"]
        assert sum(p.file_size for p in libs) == row["mandatory_library_payload_bytes"]
        assert {p.filename: p.file_size for p in libs} == {p["path"]: p["uncompressed_bytes"] for p in row["library_entries"]}
        item = items[row["name"]]
        assert item["metadata"]["version"] == row["version"]
        assert item["download_info"]["url"] == row["url"]
        assert item["download_info"]["archive_info"]["hashes"]["sha256"] == row["wheel_declared_sha256"]
        results.append({"name": row["name"], "version": row["version"],
                        "library_bytes": sum(p.file_size for p in libs),
                        "central_directory_sha256": row["central_directory_sha256"]})
    total = sum(p["library_bytes"] for p in results)
    assert total == report["core_new_library_lower_bound_bytes"] > report["environment_cap_bytes"]
    result = {"status": "PASS", "method": "stdlib zipfile independent directory parser and pip report identity checks",
              "packages_verified": results, "core_library_bytes": total,
              "environment_cap_bytes": report["environment_cap_bytes"],
              "excess_lower_bound_bytes": total - report["environment_cap_bytes"],
              "weight_or_library_content_checksums_verified": False,
              "no_installation_or_network": True}
    (base / "CPU_ENVIRONMENT_PRICE_VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "core_library_bytes": total, "packages": len(results)}))


if __name__ == "__main__":
    main()
