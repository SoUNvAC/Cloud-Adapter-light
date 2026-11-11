"""Wait for Phase430, validate zero-init layer, then fresh 40k and target evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

R = Path('/home/scv/shared/phase432_20261102')


def status(state, **values):
    (R/'STATUS.json').write_text(json.dumps(dict(state=state,pid=os.getpid(),**values),indent=2))


def run(args, logfile):
    with (R/logfile).open('x') as log:
        p = subprocess.Popen([sys.executable,'-u',*args],stdout=log,stderr=subprocess.STDOUT)
        status('running', stage=logfile, child_pid=p.pid)
        assert p.wait() == 0, logfile


def main():
    import torch
    from phase432.network import OutputFeatureResidual, SpatialPrior
    import torch.nn.functional as F
    torch.set_num_threads(4)
    layer = OutputFeatureResidual()
    assert sum(p.numel() for p in layer.parameters()) == 26144
    for i in range(4):
        x = torch.randn(2,384,4,6)
        assert torch.equal(layer(x,i), x)
    (R/'LAYER_CHECK.json').write_text(json.dumps(dict(parameters=26144,zero_init_identity=True)))
    spatial = SpatialPrior().eval()
    x = torch.randn(2,3,64,96)
    values = [spatial.stem(x)]
    for block in spatial.down:
        values.append(block(values[-1]))
    expected = sum(F.adaptive_avg_pool2d(v, values[-1].shape[-2:]) for v in values)
    torch.testing.assert_close(spatial(x)[0], expected)
    assert spatial.fusion.weight.numel() == 20480
    (R/'FUSION_CHECK.json').write_text(json.dumps(dict(sum_initialization_equivalent=True,parameters=20480)))
    run(['-m','phase432.preflight' ,'fused','--work-dir',str(R/'preflight/fused')], 'preflight.log')
    # Formal process starts fresh: no smoke checkpoint is loaded.
    run(['-m','phase432.train','fused','--work-dir',str(R/'fused')], 'train.log')
    run([str(R/'evaluate_phase432.py')], 'catalogue_eval.log')
    status('complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        status('failed', error=traceback.format_exc())
        raise
