"""Detached sequential 40k runs; stop on failure and keep status receipts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    code = Path(__file__).parent
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')}

    def status(state, **kwargs):
        value = dict(state=state, controller_pid=os.getpid(),
                     utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kwargs)
        tmp = root/'QUEUE_STATUS.tmp'
        tmp.write_text(json.dumps(value, indent=2))
        tmp.replace(root/'QUEUE_STATUS.json')

    try:
        for variant in ('baseline','fused'):
            assert json.loads((root/'preflight'/variant/'COMPLETE.json').read_text())['steps'] == 10
        a, b = [root/'preflight'/v for v in ('baseline','fused')]
        assert (a/'INPUT_TRACE.jsonl').read_bytes() == (b/'INPUT_TRACE.jsonl').read_bytes()
        for field in ('frozen_sha','common_sha'):
            assert json.loads((a/'INITIAL.json').read_text())[field] == json.loads((b/'INITIAL.json').read_text())[field]
        for variant in ('baseline','fused'):
            assert hashes == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')}
            with (root/(variant+'.log')).open('x') as log:
                child = subprocess.Popen([sys.executable,'-u','-m','phase429.train',variant,
                    '--work-dir',str(root/variant)], stdout=log, stderr=subprocess.STDOUT)
                status('running', variant=variant, child_pid=child.pid)
                result = child.wait()
            assert result == 0, (variant,result)
            assert json.loads((root/variant/'COMPLETE.json').read_text())['steps'] == 40000
        a, b = [root/v for v in ('baseline','fused')]
        assert (a/'INPUT_TRACE.jsonl').read_bytes() == (b/'INPUT_TRACE.jsonl').read_bytes()
        assert (a/'INPUT_LOCK.json').read_bytes() == (b/'INPUT_LOCK.json').read_bytes()
        status('complete', steps_per_variant=40000, sampled_input_traces_equal=True)
    except Exception:
        status('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
