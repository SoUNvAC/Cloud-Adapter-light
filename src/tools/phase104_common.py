"""Phase104 input acceptance utilities. No model loading or evaluation."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / 'src/research_results/phase104_20261013'
BANDS = ('B02', 'B03', 'B04', 'B08', 'B11', 'B12')
EXPECTED_BYTES = 8344418736
CAP = 9000000000
MODEL_SHA = {'G6': '385b26b0b90b10281e24b0a3f73479ac2d9811c65ac345af75c1f35c0ebc9cd0',
             'G7': 'e0e701b1fa68fdb96eb93eab1c88687f6eef3574b7f455db0e8f42c441ea9513'}
COORD_SHA = 'c2f029a349096738935d9239585c67ca6dfc5a4c71017f5e8634af982b86ce68'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def write(path, obj, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8', newline='\n') as f: f.write(payload)
    else:
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(payload, encoding='utf-8', newline='\n'); temp.replace(path)

def now(): return datetime.now(timezone.utc).isoformat()

def csvwrite(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def coords(path):
    assert sha(path) == COORD_SHA
    with Path(path).open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    assert len(rows) == 9500 and len({r['point_id'] for r in rows}) == 9500
    return rows

def sample48(rows):
    return sorted((r['point_id'] for r in rows), key=lambda x: hashlib.sha256(('PHASE104_INPUT|' + x).encode()).hexdigest())[:48]

def window(row, col, height=5490, width=5490):
    y, x = row // 1020 * 1020, col // 1020 * 1020
    return dict(y=y, x=x, true_height=min(1020, height-y), true_width=min(1020, width-x),
                point_y=row-y, point_x=col-x, shape=[1020,1020], resize=[512,512])
