"""Read only trusted SHA-verified torch.save head ZIP tensor storage, without torch."""
import pickle,zipfile,io,collections
import numpy as np
def read(path):
 with zipfile.ZipFile(path) as z:
  prefix=next(n[:-len('data.pkl')] for n in z.namelist() if n.endswith('/data.pkl'));cache={}
  def rebuild(storage,offset,size,stride,*unused):
   a=storage;size=tuple(size);stride=tuple(stride);assert offset>=0 and all(s>=0 for s in size) and all(s>=0 for s in stride);last=offset+sum((s-1)*t for s,t in zip(size,stride)) if all(size) else offset;assert not all(size) or last<len(a);v=np.lib.stride_tricks.as_strided(a[offset:],shape=size,strides=tuple(t*a.dtype.itemsize for t in stride),writeable=False);return v.copy()
  class Restricted(pickle.Unpickler):
   def find_class(self,module,name):
    if module=='collections' and name=='OrderedDict':return collections.OrderedDict
    if module=='torch._utils' and name in ['_rebuild_tensor_v2','_rebuild_tensor']:return rebuild
    if module=='torch' and name in ['FloatStorage','LongStorage','DoubleStorage','HalfStorage','IntStorage','ByteStorage','BoolStorage']:return {'FloatStorage':'<f4','LongStorage':'<i8','DoubleStorage':'<f8','HalfStorage':'<f2','IntStorage':'<i4','ByteStorage':'u1','BoolStorage':'?'}[name]
    raise ValueError((module,name))
   def persistent_load(self,pid):
    kind,dtype,key,location,count=pid;assert kind=='storage'
    if key not in cache:cache[key]=np.frombuffer(z.read(prefix+'data/'+str(key)),dtype=np.dtype(dtype));assert len(cache[key])==count
    return cache[key]
  return Restricted(io.BytesIO(z.read(prefix+'data.pkl'))).load()
