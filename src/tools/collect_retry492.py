"""Longer transport timeout only; original experiment files stay unchanged."""
from pathlib import Path
import inspect,runpy
import collect_phase241 as transport
# Reuse the original SSH command/return-code checks with a larger transfer timeout.
code=inspect.getsource(transport.ssh).replace('timeout=90','timeout=300')
exec(compile(code,'transport_timeout_override','exec'),transport.__dict__)
runpy.run_path(str(Path(__file__).with_name('collect_all492.py')),run_name='__main__')
