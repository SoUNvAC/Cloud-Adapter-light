"""Diagnosed pre-data synthetic failure; preserve first run, execute real audit once."""
import audit_phase321 as a
import traceback,subprocess
assert (a.O/'FAILURE.json').exists() and not (a.O/'INPUT_LOCK.json').exists()
assert "assert v.geom_type=='Polygon'" in (a.O/'FAILURE.json').read_text()
old=subprocess.check_output(['git','show','4b0b188478aa32630b19509321f0e6231edb7228:src/tools/audit_phase321.py'],cwd=a.R)
p=a.O/'INITIAL_AUDIT_CODE.py'
assert not p.exists();p.write_bytes(old)
a.O=a.O/'recovery01'
try:a.main()
except Exception:
 if a.O.exists():a.put('FAILURE.json',{'traceback':traceback.format_exc(),'elapsed':a.time.monotonic()-a.START});a.put('EXIT_CODE.json',{'exit_code':1})
 raise
