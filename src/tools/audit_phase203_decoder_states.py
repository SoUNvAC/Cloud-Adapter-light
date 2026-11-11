"""Read-only tensor audit; no torch, model construction, inference or training."""
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/phase203/phase203_20261018/state_audit'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensors(path):
    raw = path.read_bytes()
    length = struct.unpack('<Q', raw[:8])[0]
    assert 0 < length < len(raw) - 8
    header = json.loads(raw[8:8 + length])
    start = 8 + length
    result = {}
    sizes = {'F32': 4, 'I64': 8}
    ranges = []
    for name, spec in header.items():
        if name == '__metadata__':
            continue
        lo, hi = spec['data_offsets']
        assert 0 <= lo <= hi <= len(raw) - start
        assert hi - lo == math.prod(spec['shape']) * sizes[spec['dtype']]
        result[name] = (spec, raw[start + lo:start + hi])
        ranges.append((lo, hi))
    ranges.sort()
    assert ranges[0][0] == 0 and ranges[-1][1] == len(raw) - start
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    return result


def digest(items):
    value = hashlib.sha256()
    for name in sorted(items):
        value.update(name.encode())
        value.update(items[name][1])
    return value.hexdigest()


def main():
    white_path = ROOT / 'src/research_results/phase122_20261015/C100_PARAMETER_WHITELIST.json'
    paths = {
        'initial': ROOT / 'outputs/phaseCal/remote_backup/phase122_20261015/train_C/decoder_initial.safetensors',
        '12201': ROOT / 'outputs/phase122/phase122_20261015/C_decoder_final.safetensors',
        '12301': ROOT / 'outputs/phase123/phase123_20261016/C_decoder_final.safetensors',
    }
    receipts = {
        '12201': ROOT / 'outputs/phase122/phase122_20261015/C100_TRAINING_RECEIPT.json',
        '12301': ROOT / 'outputs/phase123/phase123_20261016/C123_TRAINING_RECEIPT.json',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    lock = {'files': [{'path': str(p), 'sha256': sha(p)}
                      for p in [white_path, *paths.values(), *receipts.values(), Path(__file__)]]}
    (OUT / 'INPUT_CODE_LOCK.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    white = read_json(white_path)
    initial = tensors(paths['initial'])
    assert set(initial) == set(white['decoder_state_names'])
    assert digest(initial) == white['initial_decoder_sha256']
    parameters = set(white['trainable_names'])
    buffers = set(initial) - parameters
    suffixes = ('.running_mean', '.running_var', '.num_batches_tracked')
    assert all(name.endswith(suffixes) for name in buffers), 'Unclassified decoder buffer'
    bn_modules = {name.rsplit('.', 1)[0] for name in buffers}
    graph = dict(white['graph_signature']['modules'])
    assert all('BatchNorm' in graph[name] for name in bn_modules)
    rows, summary = [], {}
    for seed, path in paths.items():
        if seed == 'initial':
            continue
        receipt = read_json(receipts[seed])
        assert sha(path) == receipt['final_model_sha256']
        assert receipt['initial_decoder_sha256'] == digest(initial)
        assert receipt['encoder_unchanged'] is True
        final = tensors(path)
        assert set(final) == set(initial)
        assert digest(final) == receipt['final_decoder_sha256']
        for name in sorted(initial):
            spec, before = initial[name]
            new_spec, after = final[name]
            assert spec['shape'] == new_spec['shape'] and spec['dtype'] == new_spec['dtype']
            row = {'seed': seed, 'name': name,
                   'group': 'learned_parameter' if name in parameters else 'BN_buffer',
                   'numel': math.prod(spec['shape']), 'changed': before != after,
                   'before_sha256': hashlib.sha256(before).hexdigest(),
                   'after_sha256': hashlib.sha256(after).hexdigest(),
                   'mean_absolute_change': '', 'before_counter': '', 'after_counter': ''}
            if name in buffers:
                fmt = '<' + ('f' if spec['dtype'] == 'F32' else 'q') * row['numel']
                a, b = struct.unpack(fmt, before), struct.unpack(fmt, after)
                assert all(math.isfinite(x) for x in [*a, *b])
                row['mean_absolute_change'] = sum(abs(y - x) for x, y in zip(a, b)) / len(a)
                if name.endswith('.running_var'):
                    assert min(a) >= 0 and min(b) >= 0
                if name.endswith('.num_batches_tracked'):
                    assert len(a) == len(b) == 1
                    row['before_counter'], row['after_counter'] = a[0], b[0]
            rows.append(row)
        current = [r for r in rows if r['seed'] == seed]
        counters = [r for r in current if r['name'].endswith('.num_batches_tracked')]
        summary[seed] = {
            'decoder_BN_modules': len(bn_modules),
            'parameter_tensors': len(parameters),
            'changed_parameter_tensors': sum(r['changed'] for r in current if r['group'] == 'learned_parameter'),
            'BN_buffer_tensors': len(buffers),
            'changed_BN_buffer_tensors': sum(r['changed'] for r in current if r['group'] == 'BN_buffer'),
            'counter_deltas': sorted({r['after_counter'] - r['before_counter'] for r in counters}),
            'recorded_encoder_unchanged': receipt['encoder_unchanged'],
            'encoder_bytes_independently_audited_here': False,
            'decoder_digest_verified': True,
        }
    with (OUT / 'TENSOR_CHANGES.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    outputs = {p.name: sha(p) for p in OUT.iterdir() if p.name != 'SHA256SUMS.json'}
    (OUT / 'SHA256SUMS.json').write_text(json.dumps(outputs, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
