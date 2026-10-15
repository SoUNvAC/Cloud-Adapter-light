"""Package only shared Phase120 small receipts and unlabelled point scores."""
import json,zipfile,sys
from pathlib import Path
from phase120_common import *

def main():
    setup();assert (ROOT/'EXIT_CODE').read_text().strip()=='0'
    c=load(ROOT/'CONTROLLER_STATUS.json');assert c['calls_completed']<=8000 and c['elapsed_seconds']<=14400
    files=[]
    for p in ROOT.rglob('*.json'):
        if 'controlled_caches' in p.parts:continue
        files.append(p)
    files += list((ROOT/'aggregate').glob('*.csv'))
    # No dense scores, raw logits, labels, model optimizer or work_dirs paths.
    assert all(p.is_relative_to(ROOT) for p in files)
    rel={str(p.relative_to(ROOT)):sha(p) for p in files}
    write(ROOT/'DELIVERY_MANIFEST.json',{'files':rel,'source':'shared only','no_work_dirs_transfer':True})
    zpath=ROOT/'phase120_delivery.zip'
    with zipfile.ZipFile(zpath,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/'DELIVERY_MANIFEST.json']:z.write(p,p.relative_to(ROOT).as_posix())
    write(ROOT/'DELIVERY_RECEIPT.json',{'bytes':zpath.stat().st_size,'sha256':sha(zpath),'files':len(files)+1,
        'decoder_checkpoints':{p:{'bytes':(ROOT/(p+'_decoder_final.safetensors')).stat().st_size,
            'sha256':sha(ROOT/(p+'_decoder_final.safetensors'))} for p in ('C','D')}})
    print(load(ROOT/'DELIVERY_RECEIPT.json'))
if __name__=='__main__':main()
