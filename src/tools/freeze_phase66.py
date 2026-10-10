"""Metadata-only freeze. No target pixel, prediction or metric reads."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

SPLIT_SHA = 'a431745e7a17bbd4a7b9e12ef3bf76767d4eb827911bd5cb5e632683c2177f7b'
RECOVERY_SHA = 'bcb91cf56d3b796c22bbf60891fb8d045dd8e6af80f5c42a00818fa72501f31f'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def code_digest(path):
    # Git autocrlf on Windows must not create false code/config drift on Linux.
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--recovery', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--groups', type=Path, required=True)
    p.add_argument('--minimum-support', type=int, choices=[2, 16], required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if digest(a.split) != SPLIT_SHA or digest(a.recovery) != RECOVERY_SHA:
        raise ValueError('Original freeze mismatch')
    old = json.loads(a.split.read_text(encoding='utf-8'))
    recovery = json.loads(a.recovery.read_text(encoding='utf-8'))
    assert old['status'] == 'stopped_insufficient_h1_support' and not old['training_authorized']
    selected = [r for r in old['proposed_rows'] if r['split'] == '65a_confirmation']
    assert len(selected) == len({r['group_id'] for r in selected}) == 64
    assert len({r['representative'] for r in selected}) == 64
    others = [r for r in old['proposed_rows'] if r['split'] != '65a_confirmation']
    assert not {r['group_id'] for r in selected} & {r['group_id'] for r in others}
    assert not {x for r in selected for x in r['products']} & {x for r in others for x in r['products']}
    for r in selected:
        assert r['representative'] in r['products']
    gd = json.loads(a.groups.read_text(encoding='utf-8'))
    eligible = {g['group_id']: g for g in gd['groups'] if g['eligible']}
    for r in old['proposed_rows']:
        assert set(r['products']) == set(eligible[r['group_id']]['shadow_valid_products'])
    semantic_groups_sha = hashlib.sha256(json.dumps(gd['groups'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    files = ['src/research_plans/PHASE66_20261010.md', 'src/tools/freeze_phase66.py',
             'src/tools/run_phase66.py', 'src/tools/phase66_statistics.py',
             'src/tools/test_phase66.py', 'src/tools/launch_phase66.sh',
             'src/tools/prepare_phase65a_explore.py', 'src/tools/phase65d_metrics.py',
             'src/tools/phase65d_recovery_math.py', 'src/cloud_adapter/datasets/phase65_catalogue.py',
             'src/configs/protocol/phase65a_explore_rgb.py',
             'src/configs/protocol/phase64_source_parent_rgb.py']
    code = {rel: code_digest(repo / rel) for rel in files}
    for folder in ['src/cloud_adapter', 'src/configs']:
        for path in sorted((repo / folder).rglob('*.py')):
            code[path.relative_to(repo).as_posix()] = code_digest(path)
    a.output.mkdir(parents=True, exist_ok=False)
    # Preserve the byte-for-byte original shallow models, standardization and thresholds.
    with (a.output / 'recovery_lock.json').open('xb') as f:
        f.write(a.recovery.read_bytes())
    write(a.output / 'phase66_lock.json', dict(
        phase='Phase66 frozen spectral candidate independent validation',
        created_utc=datetime.now(timezone.utc).isoformat(),
        status='frozen_before_unsealing_remote_provenance_audit_pending',
        authorization='user_20261010_phase66_once_independent_evaluation',
        original_split_sha256=SPLIT_SHA, recovery_lock_sha256=RECOVERY_SHA,
        source_checkpoint_sha256=recovery['source_checkpoint_sha256'],
        msre_checkpoint_sha256=recovery['msre_checkpoint_sha256'],
        prepared_fit_development_manifest_sha256=recovery['prepared_manifest_sha256'],
        fit_input_sha256=recovery['fit_input_sha256'], code_sha256=code,
        code_hash_encoding='SHA256 of UTF8 file bytes with CRLF normalized to LF, matching Git blobs',
        group_semantics_sha256=semantic_groups_sha,
        provenance_input_sha256=sorted(gd['input_sha256'].values()),
        official_files=json.loads((repo/'data/sentinel2_cloud_mask_catalogue_4172871/download_status.json').read_text())['verified_files'],
        rows=selected, public_shadow_supported_groups=sum(r['h1_supported'] for r in selected),
        primary_comparison='L3_minus_L2', minimum_ap_groups=a.minimum_support,
        bootstrap_seed=66010, bootstrap_resamples=10000, fpr_budget=.01,
        model_order=['Source', 'MsRE', 'L1', 'L2', 'L3'],
        candidate='L3', spectral_ablation='L2',
        fit_sampling='153 equal groups; up to 8192 valid pixels/group without replacement; seed 65081 + first8hex SHA256(product)',
        evaluation_sampling='exactly one original representative for each of the unchanged 64 groups; all valid pixels; no replacement or additions',
        forbidden=['65b previously seen', '65c', 'old sealed test', 'ALCD',
                   'refitting', 'threshold scanning for deployment', 'checkpoint selection'],
        old_h1_failure_unchanged=True, phase65c_authorized=False,
        independent_pixels_read=False, independent_predictions_read=False))
    print(json.dumps(dict(output=str(a.output), phase66_lock_sha256=digest(a.output/'phase66_lock.json'),
                          groups=64, public_shadow_groups=sum(r['h1_supported'] for r in selected))))


if __name__ == '__main__':
    main()
