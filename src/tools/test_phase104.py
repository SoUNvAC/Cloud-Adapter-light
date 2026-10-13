"""Synthetic download fault/budget checks; no research data or network."""
import base64
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import download_phase104 as d

class Response(io.BytesIO):
    status=206
    def __init__(self,payload,start,total,status=206):
        super().__init__(payload);self.status=status
        self.headers={'Content-Range':f'bytes {start}-{total-1}/{total}','Content-Length':str(len(payload))}

class DownloadTests(unittest.TestCase):
    def run_case(self, *, partial=b'',bad_md5=False,cap=100,status=206):
        tmp=tempfile.TemporaryDirectory(dir=d.REPO/'outputs');self.addCleanup(tmp.cleanup)
        parent=Path(tmp.name);out=parent/'lock';out.mkdir();root=parent/'data'
        payload=b'0123456789abcdef'
        obj=dict(product_id='test',band='B02',name='test/B02.jp2',generation='123',bytes=len(payload),
                 md5_base64=base64.b64encode(hashlib.md5(payload).digest()).decode(),path='test/B02.jp2')
        if bad_md5:obj['md5_base64']='invalid'
        d.write(out/'object_manifest.json',dict(objects=[obj]));d.write(out/'task_lock.json',dict(object_manifest_sha256=d.sha(out/'object_manifest.json')))
        if partial:
            p=root/'imagery/test/B02.jp2.partial';p.parent.mkdir(parents=True);p.write_bytes(partial)
        def request(req,**kw):
            self.assertIn('generation=123',req.full_url)
            start=int(req.headers['Range'].split('=')[1].split('-')[0]);self.assertEqual(start,len(partial))
            return Response(payload[start:],start,len(payload),status)
        with patch.object(d,'RESULT',out),patch.object(d,'EXPECTED_BYTES',len(payload)),patch.object(d,'CAP',cap),patch.object(d.urllib.request,'urlopen',request):
            # One-object fixture reaches its intentional final96-count assertion.
            try:d.download(root)
            except AssertionError as e:
                self.assertFalse(str(e).startswith('Existing file checksum mismatch'))
        return root,payload

    def test_resume_and_hash(self):
        root,payload=self.run_case(partial=b'01234')
        self.assertEqual((root/'imagery/test/B02.jp2').read_bytes(),payload)
        self.assertEqual(d.read(root/'transfer_budget.json')['actual_received_bytes'],len(payload)-5)

    def test_md5_failure_preserves_partial(self):
        root,payload=self.run_case(bad_md5=True)
        self.assertFalse((root/'imagery/test/B02.jp2').exists())
        self.assertEqual((root/'imagery/test/B02.jp2.partial').read_bytes(),payload)

    def test_cap_prevents_overread(self):
        root,_=self.run_case(cap=5)
        self.assertEqual(d.read(root/'download_status.json')['status'],'stopped')
        self.assertFalse((root/'imagery/test/B02.jp2').exists())

    def test_ignored_range_not_appended(self):
        root,_=self.run_case(partial=b'01234',status=200)
        self.assertEqual((root/'imagery/test/B02.jp2.partial').read_bytes(),b'01234')

if __name__=='__main__':unittest.main()
