"""Auditable in-memory seed/budget adapter; never modifies Phase122 sources."""
import hashlib,sys
import phase123_common as context
def main():
    kind=sys.argv.pop(1);assert kind in ('train','infer')
    cfg=context.setup();record=cfg['accepted_worker_transforms'][kind]
    source=(context.REPO/record['source']).read_text(encoding='utf-8')
    for old,new in record['replacements']:
        assert old in source,(kind,old)
        source=source.replace(old,new)
    assert hashlib.sha256(source.encode('utf-8')).hexdigest()==record['transformed_sha256']
    sys.modules['phase122_common']=context
    exec(compile(source,str(context.REPO/record['source']), 'exec'),{'__name__':'__main__','__file__':str(context.REPO/record['source'])})
if __name__=='__main__':main()
