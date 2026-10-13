"""Inspect inherited Phase89 preprocessor/interpolation configuration, no network build."""
import ast
import inspect
import os
import argparse
from phase104_common import *

def main(root):
    import mmcv,mmseg,torch,cv2,numpy as np
    from mmcv.transforms import Resize
    from mmseg.models.segmentors.base import BaseSegmentor
    from mmseg.models.decode_heads.decode_head import BaseDecodeHead
    from mmengine.config import Config
    os.environ['PHASE65_PREPARED_MANIFEST']='metadata_only_not_opened'
    os.environ['PHASE65_SOURCE_CHECKPOINT']='metadata_only_not_opened'
    config=REPO/'src/configs/protocol/phase65a_explore_rgb.py'
    c=Config.fromfile(str(config),import_custom_modules=False)
    locked=read(RESULT/'input_operator_lock.json')['runtime_chain']
    a=hashlib.sha256(inspect.getsource(Resize).encode()).hexdigest()
    b=hashlib.sha256(inspect.getsource(BaseSegmentor.postprocess_result).encode()).hexdigest()
    assert a==locked['Resize_class_source_sha256'] and b==locked['postprocess_result_source_sha256']
    assert mmcv.__version__==locked['mmcv'] and mmseg.__version__==locked['mmseg']
    files={}
    def inherited(p):
        p=p.resolve();assert p.is_relative_to(REPO)
        files[str(p.relative_to(REPO))]=sha(p)
        for node in ast.parse(p.read_text(encoding='utf-8-sig')).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_base_' for t in node.targets):
                value=ast.literal_eval(node.value)
                for name in ([value] if isinstance(value,str) else value):inherited(p.parent/name)
    inherited(config)
    align=c.model.decode_head.get('align_corners',inspect.signature(BaseDecodeHead.__init__).parameters['align_corners'].default)
    pre=dict(c.model.data_preprocessor)
    assert pre['bgr_to_rgb'] is True
    write(root/'runtime_chain_acceptance.json',dict(status='passed',mmcv=mmcv.__version__,mmseg=mmseg.__version__,
          torch=torch.__version__,cv2=cv2.__version__,numpy=np.__version__,
          Resize_source_sha256=a,postprocess_source_sha256=b,original_data_preprocessor=pre,
          decode_head_align_corners=align,config_inheritance_sha256=files,
          network_built=False,model_loaded=False,forwards=0),True)
    print('Original Phase89 runtime chain accepted:',pre,'align_corners=',align)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();main(a.root)
