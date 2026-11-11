"""Download the original .pth from Meta's gated HF repository after approval.

Authenticate on the remote machine with `huggingface-cli login` or `hf auth login`.
Credentials are read by huggingface_hub; never passed or printed by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--directory',default='/home/scv/shared/checkpoints/dinov3-splus')
    args=parser.parse_args()
    repo='facebook/dinov3-vits16plus-pretrain-lvd1689m'
    filename='dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
    path=Path(hf_hub_download(repo_id=repo,filename=filename,local_dir=args.directory))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha.startswith('4057cbaa'), 'Official checkpoint hash does not match'
    receipt=dict(repo=repo,file=filename,path=str(path),bytes=path.stat().st_size,sha256=sha)
    Path(args.directory,'DOWNLOAD.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    main()
