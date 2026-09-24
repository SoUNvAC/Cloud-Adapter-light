import argparse
import csv
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


SOURCE_URL = "https://landsat.usgs.gov/node/7"
SCENE_RE = re.compile(r"^LC8\d{6}\d{7}[A-Z]{3}\d{2}$")


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_row = False
        self.in_cell = False
        self.cell = []
        self.row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.in_row = True
            self.row = []
        elif self.in_row and tag in ("td", "th"):
            self.in_cell = True
            self.cell = []

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if self.in_row and self.in_cell and tag in ("td", "th"):
            self.row.append(" ".join("".join(self.cell).split()))
            self.in_cell = False
        elif self.in_row and tag == "tr":
            self.rows.append(self.row)
            self.in_row = False


def parse_scene_metadata(html):
    parser = TableParser()
    parser.feed(html)
    records = {}
    for row in parser.rows:
        scene = next((cell for cell in row if SCENE_RE.fullmatch(cell)), None)
        if scene is None:
            continue
        flags = [cell.lower() for cell in row if cell.lower() in ("yes", "no")]
        if len(flags) != 1:
            raise RuntimeError(f"Expected one Shadows? value for {scene}: {row}")
        if scene in records:
            raise RuntimeError(f"Duplicate USGS scene row: {scene}")
        records[scene] = flags[0]
    if len(records) != 96:
        raise RuntimeError(f"Expected 96 USGS scenes, parsed {len(records)}")
    yes = sum(value == "yes" for value in records.values())
    if yes != 32:
        raise RuntimeError(f"Expected 32 Shadows?=yes scenes, parsed {yes}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument(
        "--output",
        default="research_plans/protocol_data/l8_biome_usgs_shadow_status.csv",
    )
    args = parser.parse_args()
    request = urllib.request.Request(
        args.url, headers={"User-Agent": "Cloud-Adapter protocol audit"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", errors="replace")
    records = parse_scene_metadata(html)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("scene", "usgs_shadows", "source_url")
        )
        writer.writeheader()
        for scene, shadows in sorted(records.items()):
            writer.writerow(
                {"scene": scene, "usgs_shadows": shadows, "source_url": args.url}
            )
    print(f"wrote {len(records)} scenes ({sum(v == 'yes' for v in records.values())} yes) to {output}")


if __name__ == "__main__":
    main()
