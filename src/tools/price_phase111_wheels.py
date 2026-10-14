"""Price resolved mandatory new runtime libraries from wheel ZIP directories only.

No wheel installation/full downloads. A lower bound exceeding the environment
cap is sufficient to stop; small/existing packages need not be downloaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.parse
import urllib.request


CAP = 4000000000


def directory_price(url, out, name):
    assert urllib.parse.urlparse(url).hostname == "files.pythonhosted.org"
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
        length = int(r.headers["Content-Length"])

    def read_range(start, end):
        assert 0 <= start <= end < length and end - start + 1 <= 4000000
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            assert r.status == 206, "Server must support ranges; refusing full wheel"
            assert r.headers["Content-Range"].startswith(f"bytes {start}-{end}/")
            body = r.read(end - start + 2)
            assert len(body) == end - start + 1
            return body

    tail_start = max(0, length - 65557)
    tail = read_range(tail_start, length - 1)
    pos = tail.rfind(b"PK\x05\x06")
    assert pos >= 0
    eocd = struct.unpack_from("<4s4H2LH", tail, pos)
    count, cd_size, cd_offset = eocd[4:7]
    assert count != 65535 and cd_offset != 0xffffffff and cd_size != 0xffffffff, "ZIP64 needs explicit pricing support"
    assert cd_size <= 4000000
    if cd_offset >= tail_start and cd_offset + cd_size <= length:
        cd = tail[cd_offset - tail_start:cd_offset - tail_start + cd_size]
        transferred = len(tail)
    else:
        cd = read_range(cd_offset, cd_offset + cd_size - 1)
        transferred = len(tail) + len(cd)
    (out / (name + ".central-directory.bin")).write_bytes(cd)
    rows = []
    offset = 0
    while offset < len(cd):
        fields = struct.unpack_from("<4s6H3L5H2L", cd, offset)
        assert fields[0] == b"PK\x01\x02"
        compressed, uncompressed = fields[8:10]
        n, extra, comment = fields[10:13]
        filename = cd[offset + 46:offset + 46 + n].decode("utf-8")
        assert uncompressed != 0xffffffff and compressed != 0xffffffff
        rows.append({"path": filename, "uncompressed_bytes": uncompressed,
                     "compressed_bytes": compressed})
        offset += 46 + n + extra + comment
    assert len(rows) == count
    # Only shared library payload is counted. It must exist in the installed
    # runtime, regardless of pip cache/unpack strategy or reused small packages.
    libraries = [x for x in rows if ".so" in Path(x["path"]).name and not x["path"].endswith("/")]
    return {"name": name, "url": url, "wheel_declared_length": length,
            "wheel_entries": count, "uncompressed_all_entries_bytes": sum(x["uncompressed_bytes"] for x in rows),
            "mandatory_library_payload_bytes": sum(x["uncompressed_bytes"] for x in libraries),
            "library_entries": libraries, "central_directory_sha256": hashlib.sha256(cd).hexdigest(),
            "http_range_body_bytes": transferred,
            "content_checksum_validated": False,
            "estimate_type": "ZIP central-directory declared uncompressed logical bytes, not installed disk measurement"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-directory", type=Path, required=True)
    ap.add_argument("--continue-from", type=Path)
    args = ap.parse_args()
    base = args.gate_directory
    out = base / ("wheel_pricing_continued" if args.continue_from else "wheel_pricing")
    out.mkdir(exist_ok=False)
    resolution = json.loads((base / "PIP_RESOLUTION.json").read_text(encoding="utf-8"))
    existing = json.loads((base / "EXISTING_PACKAGES.json").read_text(encoding="utf-8"))
    items = {x["metadata"]["name"].lower().replace("_", "-"): x for x in resolution["install"]}
    rows = []
    if args.continue_from:
        previous = json.loads(args.continue_from.read_text(encoding="utf-8"))
        for row in previous["priced_packages"]:
            cd = args.continue_from.parent / (row["name"] + ".central-directory.bin")
            assert hashlib.sha256(cd.read_bytes()).hexdigest() == row["central_directory_sha256"]
            assert items[row["name"]]["download_info"]["url"] == row["url"]
            assert items[row["name"]]["metadata"]["version"] == row["version"]
            rows.append(row)
    total = sum(row["mandatory_library_payload_bytes"] for row in rows)
    priority = ["torch", "nvidia-cudnn-cu13", "nvidia-cublas", "nvidia-cusolver", "nvidia-cusparse", "triton"]
    order = priority + [name for name in items if name not in priority]
    for name in order:
        if total > CAP:
            break
        if any(row["name"] == name for row in rows):
            continue
        if name not in items:
            continue
        item = items[name]
        old = existing.get(name, {})
        version = item["metadata"]["version"]
        if old.get("version") == version:
            continue
        row = directory_price(item["download_info"]["url"], out, name)
        row.update(version=version, existing_version=old.get("version", "not_installed"),
                   wheel_declared_sha256=item["download_info"]["archive_info"]["hashes"]["sha256"])
        rows.append(row)
        total += row["mandatory_library_payload_bytes"]
        print(json.dumps({"package": name, "version": version,
                          "mandatory_library_payload_bytes": row["mandatory_library_payload_bytes"],
                          "cumulative_bytes": total}), flush=True)
        if total > CAP:
            break
    result = {"environment_cap_bytes_including_caches": CAP,
              "new_library_payload_lower_bound_bytes": total,
              "lower_bound_excludes": ["other resolved packages", "Python runtime/venv files", "pip caches", "unpack temporaries", "pyc files"],
              "gate": "STOP_ENVIRONMENT_BUDGET" if total > CAP else "incomplete_pricing_not_authorized_to_install",
              "priced_packages": rows, "other_versions_searched": False,
              "installation": False, "model_calls": 0,
              "metadata_http_body_bytes": sum(x["http_range_body_bytes"] for x in rows)}
    (out / "ENVIRONMENT_PRICE.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
