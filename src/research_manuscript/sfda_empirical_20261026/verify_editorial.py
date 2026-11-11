"""Check manuscript transcription against existing CSVs; no model/data evaluation."""
import csv
import hashlib
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULT = REPO / 'src/research_results/phase311_20261025'


def read_csv(name):
    with (RESULT / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def table_row(text, label, cells):
    matches = [[v.strip() for v in line.strip().strip('|').split('|')]
               for line in text.splitlines() if line.startswith('| ' + label + ' |')]
    matches = [row for row in matches if len(row) == cells]
    assert len(matches) == 1, (label, cells, matches)
    return matches[0]


metrics = {r['strategy']: r for r in read_csv('METRICS.csv')}
events = {r['strategy']: r for r in read_csv('DAMAGE_BENEFIT.csv')
          if r['product'] == 'ALL' and r['region'] == 'all'}
tables = (HERE / 'CORE_TABLES.md').read_text(encoding='utf-8')
main_columns = ['group_mean_mIoU_pct', 'pooled_mIoU_pct', 'Surface_IoU_pct',
                'Cloud_IoU_pct', 'Shadow_IoU_pct', 'macro_AP_pct', 'pooled_AP_pct',
                'macro_recall_pct', 'macro_FPR_pct']
for name in ['Source', 'Q90-PL']:
    row = table_row(tables, name, 10)
    assert row[1:] == [f'{float(metrics[name][key]):.3f}' for key in main_columns]
delta = table_row(tables, 'Difference (pp)', 10)
for cell, key in zip(delta[1:], main_columns):
    expected = f'{float(metrics["Q90-PL"][key]) - float(metrics["Source"][key]):+.3f}'
    assert cell.replace('−', '-') == expected, (cell, key, expected)

for name in ['Q90-PL', 'Q90-SP', 'Q90-CM']:
    event = events[name]
    d, benefit = int(event['original_D']), int(event['original_Cloud_correction'])
    repair, retain = int(event['D_to_Surface']), int(event['benefit_retained_Cloud'])
    assert d == 506798 and benefit == 784478
    assert sum(int(event[k]) for k in ['D_to_Surface', 'D_still_Cloud', 'D_to_Shadow']) == d
    assert sum(int(event[k]) for k in ['benefit_retained_Cloud', 'benefit_back_Shadow', 'benefit_to_Surface']) == benefit
    row = table_row(tables, name, 8)
    assert row[1] == f'{float(metrics[name][main_columns[0]]):.3f} / {float(metrics[name][main_columns[1]]):.3f}'
    assert row[2] == f'{repair:,} ({100 * repair / d:.3f}%)'
    assert row[3] == f'{int(event["D_still_Cloud"]):,} / {int(event["D_to_Shadow"]):,}'
    assert row[4] == f'{retain:,} ({100 * retain / benefit:.3f}%)'
    assert row[5] == f'{int(event["benefit_back_Shadow"]):,} / {int(event["benefit_to_Surface"]):,}'
    assert row[6] == f'{int(event["outside_D_new_Surface_to_Cloud"]):,} / {int(event["outside_D_new_Surface_to_Shadow"]):,}'

class_index = {'Surface': 0, 'Cloud': 1, 'Shadow': 2}
source_cm = json.loads(metrics['Source']['confusion'])
pl_cm = json.loads(metrics['Q90-PL']['confusion'])
flow_count = 0
for line in tables.splitlines():
    if not line.startswith('| ') or ' → ' not in line:
        continue
    row = [v.strip() for v in line.strip().strip('|').split('|')]
    if len(row) != 7:
        continue
    reference, prediction = row[0].split(' → ')
    if reference not in class_index:
        continue
    i, j = class_index[reference], class_index[prediction]
    old, new, removed, added = [int(v.replace(',', '')) for v in row[1:5]]
    assert old == source_cm[i][j] and new == pl_cm[i][j]
    assert new - old == added - removed == int(row[5].replace(',', '').replace('−', '-'))
    corrected, another = [int(v.strip().replace(',', '')) for v in row[6].split('/')]
    assert corrected + another == removed
    flow_count += 1
assert flow_count == 6
assert 40622307 + 33554491 + 1292360 == 75469158
assert f'{75469158 / 104448397 * 100:.5f}' == '72.25497'
assert f'{101857 / 104448397 * 100:.6f}' == '0.097519'
assert 506798 + 19553 == 526351
assert 955 + 288 + 16 == 1259
assert 2 * 1259 == 2518

seed_path = REPO / 'src/research_results/phase315_20261026/SEED_METRICS.csv'
with seed_path.open(encoding='utf-8-sig', newline='') as stream:
    seed_rows = {r['seed']: r for r in csv.DictReader(stream)}
seed_names = ['30201', '31501', '31502']
base = seed_rows['Source']
for name in seed_names:
    item = seed_rows[name]
    row = table_row(tables, name, 8)
    group = float(item['group_mean_mIoU_pct'])
    pooled = float(item['pooled_mIoU_pct'])
    recall = float(item['macro_Shadow_recall_pct'])
    assert row[1:3] == [f'{group:.3f}', f'{pooled:.3f}']
    assert row[3] == f'{group - float(base["group_mean_mIoU_pct"]):+.3f}'
    assert row[4] == f'{recall:.3f}'
    assert row[5].replace('−', '-') == f'{recall - float(base["macro_Shadow_recall_pct"]):+.3f}'
    cm = json.loads(item['confusion'])
    assert int(row[6].replace(',', '')) == source_cm[1][2] - cm[1][2]
    assert int(row[7].replace(',', '')) == cm[0][1] - source_cm[0][1]
mean_row = table_row(tables, 'Mean ± sample SD', 8)
for key, output_col, shift in [('group_mean_mIoU_pct', 1, False),
                               ('pooled_mIoU_pct', 2, False),
                               ('group_mean_mIoU_pct', 3, True),
                               ('macro_Shadow_recall_pct', 4, False),
                               ('macro_Shadow_recall_pct', 5, True)]:
    values = [float(seed_rows[name][key]) for name in seed_names]
    mean, sd = statistics.mean(values), statistics.stdev(values)
    if shift:
        mean -= float(base[key])
    expected = f'{mean:+.3f} ± {sd:.3f}' if shift else f'{mean:.3f} ± {sd:.3f}'
    assert mean_row[output_col].replace('−', '-') == expected, (key, expected)
    assert abs(statistics.mean(values) - float(seed_rows['mean3'][key])) < 1e-12
    assert abs(sd - float(seed_rows['sample_SD3_ddof1'][key])) < 1e-12

full_draft = (HERE / 'MANUSCRIPT_FULL_DRAFT.md').read_text(encoding='utf-8')
assert re.findall(r'^#### Table (\d+b?)\.', full_draft, re.M) == ['1', '1b', '2', '3', '4']
for heading in ['## 1. Introduction', '## 2. Related work', '## 3. Experimental protocol',
                '## 4. Results', '## 5. Discussion and limitations', '## 6. Conclusion', '## References']:
    assert full_draft.count(heading) == 1, heading
assert 'argument outline' not in full_draft and 'should establish' not in full_draft
refs = full_draft.split('## References\n', 1)[1].split('## Internal editorial note', 1)[0]
assert re.findall(r'^(\d+)\. ', refs, re.M) == [str(i) for i in range(1, 11)]

links = 0
for document in HERE.glob('*.md'):
    for match in re.finditer(r'\]\((D:/[^\)]+)\)', document.read_text(encoding='utf-8')):
        assert Path(match.group(1)).exists(), (document.name, match.group(1))
        links += 1

receipt = dict(
    status='passed', checked_at_utc=datetime.now(timezone.utc).isoformat(),
    table1_metrics_and_deltas='matched original CSV to displayed precision',
    table4_counts_rates_and_metrics='matched original CSV; event partitions checked',
    directed_error_rows=flow_count, coverage_and_budget_arithmetic='passed',
    PL_initialization_replication='3 rows and mean/sample SD matched Phase315 CSV; directed changes checked against confusion',
    full_draft_structure='6 sections, 4 canonical tables plus replication panel, 10 reference entries; writing-outline placeholders removed',
    existing_local_file_links=links,
    source_files={str(p.relative_to(REPO)).replace('\\', '/'):
                  hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [RESULT / 'METRICS.csv', RESULT / 'DAMAGE_BENEFIT.csv', seed_path]},
    scope='Editorial transcription/arithmetic/link checks only; no new fits, forwards, raw metric evaluation or remote SHA revalidation',
)
(HERE / 'EDITORIAL_QA.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(status=receipt['status'], error_flow_rows=flow_count, file_links=links)))
