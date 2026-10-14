"""Complete environment payload and sequential-install peak estimate, no install."""
import argparse
import json
import time
import urllib.error
from pathlib import Path
from price_phase111_wheels import directory_price


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-directory", type=Path, required=True)
    args = ap.parse_args()
    base = args.gate_directory
    out = base / "complete_wheel_pricing"
    out.mkdir(exist_ok=True)
    resolved = json.loads((base / "PIP_RESOLUTION.json").read_text(encoding="utf-8"))
    previous_base = base.parent / "resource_gate_20261014_probe2"
    old_rows = {}
    for p in previous_base.glob("wheel_pricing*/ENVIRONMENT*PRICE.json"):
        for row in json.loads(p.read_text(encoding="utf-8"))["priced_packages"]:
            old_rows[row["url"]] = row
    rows = []
    for item in resolved["install"]:
        name = item["metadata"]["name"].lower().replace("_", "-")
        url = item["download_info"]["url"]
        dest = out / (name + ".price.json")
        if dest.exists():
            row = json.loads(dest.read_text(encoding="utf-8"))
            assert row["url"] == url
        elif url in old_rows:
            row = dict(old_rows[url], reused_directory_evidence=str(previous_base))
            dest.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        else:
            for attempt in range(3):
                try:
                    row = directory_price(url, out, name)
                    break
                except urllib.error.URLError as exc:
                    with (out / "metadata_network_retries.jsonl").open("a") as retry_log:
                        retry_log.write(json.dumps({"name":name,"attempt":attempt+1,"error":str(exc)})+"\n")
                    if attempt == 2:
                        raise
                    time.sleep(attempt+1)
            row.update(version=item["metadata"]["version"],
                       wheel_declared_sha256=item["download_info"]["archive_info"]["hashes"]["sha256"])
            dest.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        rows.append(row)
        print(json.dumps({"priced":len(rows),"of":len(resolved["install"]),"name":name,
                          "total_uncompressed":sum(x["uncompressed_all_entries_bytes"] for x in rows)}),flush=True)
    order = sorted(rows, key=lambda x:x["wheel_declared_length"], reverse=True)
    cumulative = 0
    peak = 0
    for row in order:
        cumulative += row["uncompressed_all_entries_bytes"]
        peak = max(peak, cumulative + row["wheel_declared_length"])
    result = {"resolved_packages":len(rows), "all_wheel_uncompressed_declared_bytes":cumulative,
              "sequential_largest_first_peak_bytes":peak,
              "peak_assumptions":"one wheel retained at a time; installed directly to isolated venv; no persistent wheel cache or bytecode compilation; wheel deleted after successful SHA check and install",
              "safety_margin_bytes":512000000,
              "selected_weight_and_B8A_declared_bytes":236199802,
              "estimated_total_additional_peak_with_safety_and_inputs":max(peak,cumulative+236199802)+512000000,
              "installation_order":[x["name"] for x in order], "packages":rows,
              "estimation_not_actual_installation":True,"installation":False,"model_calls":0}
    (out / "COMPLETE_ENVIRONMENT_PRICE.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
