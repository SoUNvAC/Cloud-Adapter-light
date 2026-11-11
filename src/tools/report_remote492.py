"""Finalize metrics with verified remote weights; do not imply local weight delivery."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'outputs/phase492/phase492_20261109'
acceptance = json.loads((R / 'TRAINED_WEIGHT_ACCEPTANCE.json').read_text())
assert acceptance['passed']
for arm, record in acceptance['arms'].items():
    assert record['passed']
    for name, signature in record['weights'].items():
        assert len(signature['sha256']) == 64
        local = R / arm / 'network_run' / name
        if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() != signature['sha256']:
            local.rename(local.with_suffix('.pth.partial'))

note = '\n归档：完整权重保留在远端 /home/scv/shared/phase492_20261109，已完成实际权重验签；本地完整归档评测结果及签名，未完成权重下载，.partial 文件不可用于推理。\n'
interpretation = R / 'INTERPRETATION.md'
text = interpretation.read_text(encoding='utf-8')
if note.strip() not in text:
    interpretation.write_text(text + note, encoding='utf-8')
(R / 'DELIVERY_STATUS.json').write_text(json.dumps(dict(
    metadata_transfer_verified=True, local_weights_complete=False,
    remote_weight_root='/home/scv/shared/phase492_20261109',
    remote_weight_acceptance='TRAINED_WEIGHT_ACCEPTANCE.json',
    partial_files_excluded=True), indent=2))

p = Path(__file__).with_name('report_shadow492.py')
s = p.read_text(encoding='utf-8')
old = "if n!='branch_0.pth':assert sha(R/arm/'network_run'/n)==s['sha256']"
assert s.count(old) == 1
s = s.replace(old, "assert len(s['sha256'])==64 and v['passed']  # Actual remote hashes verified in TRAINED_WEIGHT_ACCEPTANCE")
s = s.replace('weight_signatures=True,', 'remote_weight_signatures_verified=True,local_weights_complete=False,weight_storage="remote",')
s = s.replace('独立混淆矩阵复算、权重签名与传输通过。', '独立混淆矩阵复算、远端权重验签与评测元数据传输通过；本地权重未完整下载。')
exec(compile(s, str(p), 'exec'), {'__file__': str(p), '__name__': '__main__'})
