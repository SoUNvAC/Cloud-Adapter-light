import inspect,torch,json
from pathlib import Path
from mmengine.runner import Runner
from mmengine.runner.loops import IterBasedTrainLoop
r=Path('/home/scv/shared/phase497_20261110')
c=torch.load(r/'texture/iter_28000.pth',map_location='cpu')
print('keys',list(c));print('meta',{k:v for k,v in c['meta'].items() if k!='config'})
print(inspect.getsource(IterBasedTrainLoop.run))
print(inspect.getsource(Runner.resume)[-6500:])
