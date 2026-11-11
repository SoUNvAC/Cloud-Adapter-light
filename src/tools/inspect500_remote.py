import inspect
from mmdet.models.layers import MSDeformAttnPixelDecoder
from mmseg.models.decode_heads import Mask2FormerHead
print(inspect.getsource(MSDeformAttnPixelDecoder.forward))
print(inspect.getsource(Mask2FormerHead.forward))
