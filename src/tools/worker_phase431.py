"""Wait for Phase430, validate zero-init layer, then fresh 40k and target evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

R = Path('/home/scv/shared/phase431_20261102')


def status(state, **values):
    (R/'STATUS.json').write_text(json.dumps(dict(state=state,pid=os.getpid(),**values),indent=2))


def run(args, logfile):
    with (R/logfile).open('x') as log:
        p = subprocess.Popen([sys.executable,'-u',*args],stdout=log,stderr=subprocess.STDOUT)
        status('running', stage=logfile, child_pid=p.pid)
        assert p.wait() == 0, logfile


def main():
    prior = Path('/home/scv/shared/phase430_20261102/EXIT_CODE')
    status('queued', waiting_for='Phase430 training and evaluation')
    while not prior.exists() and os.environ.get("PHASE431_PARALLEL") != "1":
        time.sleep(30)
    smi = '/usr/lib/wsl/lib/nvidia-smi'
    while os.environ.get('PHASE431_PARALLEL') != '1' and subprocess.check_output([smi,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        status('queued', waiting_for='GPU idle')
        time.sleep(30)
    import torch
    from phase431.network import OutputFeatureResidual
    torch.set_num_threads(4)
    layer = OutputFeatureResidual()
    assert sum(p.numel() for p in layer.parameters()) == 26144
    for i in range(4):
        x = torch.randn(2,384,4,6)
        assert torch.equal(layer(x,i), x)
    (R/'LAYER_CHECK.json').write_text(json.dumps(dict(parameters=26144,zero_init_identity=True)))
    run(['-m','phase431.preflight','fused','--work-dir',str(R/'preflight/fused')], 'preflight.log')
    # Formal process starts fresh: no smoke checkpoint is loaded.
    run(['-m','phase431.train','fused','--work-dir',str(R/'fused')], 'train.log')
    run([str(R/'evaluate_phase431.py')], 'catalogue_eval.log')
    status('complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        status('failed', error=traceback.format_exc())
        raise
