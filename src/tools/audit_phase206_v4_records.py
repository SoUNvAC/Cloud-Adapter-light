"""Cross-check existing V4 endpoint metadata; no checkpoint loading or inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/phase205/phase205_20261019/delivery'
OUT = ROOT / 'outputs/phase206/phase206_20261019/metadata_audit'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    white_path = BASE / 'gate/SMP_WHITELIST.json'
    endpoint_path = BASE / 'ENDPOINT_CPU_STATE_ACCEPTANCE.json'
    seal_path = BASE / 'TRAINING_SEAL.json'
    receipt_paths = [BASE / f'train_{seed}/TRAINING_RECEIPT.json' for seed in (12201, 12301)]
    white, endpoint, seal = read(white_path), read(endpoint_path), read(seal_path)
    params = set(white['trainable'])
    buffers = set(white['decoder_state']) - params
    assert len(params) == len(white['trainable']) == 32
    bn_names = {name.rsplit('.', 1)[0] for name in buffers}
    decoder_bn = [row for row in white['BN'] if row['name'] in bn_names]
    assert len(buffers) == 30 and len(decoder_bn) == len(bn_names) == 10
    assert all(name.endswith(('.running_mean', '.running_var', '.num_batches_tracked')) for name in buffers)
    assert all(row['affine'] and row['track_running_stats'] for row in decoder_bn)
    assert endpoint['status'] == 'passed'
    results = []
    for path in receipt_paths:
        receipt = read(path)
        old = next(row for row in endpoint['rows'] if row['seed'] == receipt['seed'])
        assert set(receipt['trainable_names']) == params
        assert receipt['steps'] == 1000 and receipt['encoder_unchanged']
        assert old['initial_matches_original_R4'] and old['only_decoder_head_saved']
        assert old['encoder_unchanged_execution_assertion']
        assert old['initial_sha256'] == receipt['initial_sha256']
        assert old['final_sha256'] == receipt['final_sha256'] == seal[str(receipt['seed'])]
        changed = set(old['changed_keys'])
        assert changed <= set(white['decoder_state'])
        results.append(dict(seed=receipt['seed'], parameter_tensors=len(params),
                            changed_parameter_tensors=len(params & changed),
                            BN_modules=len(decoder_bn), BN_buffer_tensors=len(buffers),
                            changed_BN_buffer_tensors=len(buffers & changed),
                            initial_recorded_sha256=receipt['initial_sha256'],
                            final_recorded_sha256=receipt['final_sha256']))
    assert results[0]['initial_recorded_sha256'] == results[1]['initial_recorded_sha256']
    OUT.mkdir(parents=True, exist_ok=True)
    lock = [{'path': str(p), 'sha256': sha(p)}
            for p in [white_path, endpoint_path, seal_path, *receipt_paths, Path(__file__)]]
    report = dict(status='records_consistent', rows=results, input_code_sha256=lock,
                  checkpoint_bytes_read=False, encoder_independently_loaded=False,
                  image_pixels_read=False, neural_calls=0, training=0, fits=0,
                  engineering_note='The source BN list includes encoder modules. Decoder BN is selected by exact buffer ownership; initial whole-list count assertion was corrected before producing this report.',
                  limit='Metadata consistency only. Executor must verify real model bytes before state swaps.')
    (OUT / 'RECORD_AUDIT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'input_code_sha256'}, indent=2))


if __name__ == '__main__':
    main()
