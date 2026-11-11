"""Insert canonical result tables into the continuous manuscript; no data analysis."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
draft = (ROOT / 'MANUSCRIPT_DRAFT.md').read_text(encoding='utf-8')
tables = (ROOT / 'CORE_TABLES.md').read_text(encoding='utf-8')


def block(start, end):
    assert tables.count(start) == 1 and tables.count(end) == 1
    text = tables.split(start, 1)[1].split(end, 1)[0]
    return (start + text).strip()


inserts = [
    ('### 4.2 The acceptance statistic determines usable supervision',
     block('## Table 1.', '## Table 2.')),
    ('### 4.3 Directed events distinguish correction from error replacement',
     block('## Table 2.', '## Table 3.')),
    ('### 4.4 Surface repair does not establish a better overall adaptation outcome',
     block('## Table 3.', '## Table 4.')),
    ('## 5. Discussion and limitations',
     block('## Table 4.', '## Material assigned to the supplement')),
]
for marker, section in inserts:
    assert draft.count(marker) == 1
    section = re.sub(r'^#{2,3} Table ', '#### Table ', section, flags=re.M)
    draft = draft.replace(marker, section + '\n\n' + marker, 1)

draft = draft.replace('Internal empirical manuscript draft,', 'Assembled internal empirical manuscript draft,', 1)
draft = draft.replace('and inserted into the assembled full draft.', 'and reproduced inline in this reading version.', 1)
(ROOT / 'MANUSCRIPT_FULL_DRAFT.md').write_text(draft, encoding='utf-8')
print('Assembled MANUSCRIPT_FULL_DRAFT.md from manuscript text and four canonical tables plus replication panel.')
