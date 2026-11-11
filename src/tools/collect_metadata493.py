"""Use a longer transport timeout; download metadata only, never checkpoints."""
from pathlib import Path
import inspect,runpy
import collect_phase241 as transport
source=inspect.getsource(transport.ssh).replace('timeout=90','timeout=300').replace("'ServerAliveCountMax=2'","'ServerAliveCountMax=12'")
exec(compile(source,'transport_timeout_override','exec'),transport.__dict__)
runpy.run_path(str(Path(__file__).with_name('collect_all493.py')),run_name='__main__')
