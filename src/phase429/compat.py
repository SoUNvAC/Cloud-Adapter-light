"""Apply one compile-configuration guard to a private official DINOv3 checkout.

The eager forward mathematics are unchanged. Record the git diff in run locks.
"""
import argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('repo')
    args=parser.parse_args()
    path=Path(args.repo)/'dinov3/layers/block.py'
    source=path.read_text()
    statement='torch._dynamo.config.accumulated_cache_size_limit = 1024'
    guarded="if hasattr(torch._dynamo.config, 'accumulated_cache_size_limit'):\n    "+statement
    if guarded in source:
        print('Guard already present');return
    assert source.count(statement)==1
    path.write_text(source.replace(statement,guarded))
    print('Guarded optional torch.compile cache setting; eager model operations unchanged')

if __name__=='__main__':
    main()
