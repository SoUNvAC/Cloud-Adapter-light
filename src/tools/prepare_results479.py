from pathlib import Path
T=Path(__file__).parent
s=(T/'compare_shadow478.py').read_text(encoding='utf-8').replace('478','479')
where="groups=sorted({v['group'] for v in data['477best']['per_image']})"
addition="""for arm in ['A','B']:
 d=R/arm/'network_run';step=read(d/'FINAL_RESULTS.json')['choices']['best_shadow']
 data[f'479{arm}off']=read(d/f'OFF_INDEPENDENT_{step}.json');pairs.append((f'479{arm}best',f'479{arm}off'))
for step in [1000,3000,6000]:pairs.append((f'479B_{step}',f'479A_{step}'))
"""
assert where in s;s=s.replace(where,addition+where);(T/'compare_shadow479.py').write_bytes(s.encode())
s=(T/'status_shadow478.py').read_text(encoding='utf-8').replace('478','479');(T/'status_shadow479.py').write_bytes(s.encode())
s=(T/'finish_shadow478.sh').read_text(encoding='utf-8').replace('478','479').replace('PY=/home', 'test -f A/network_run/ABLATION_OFF.json\ntest -f B/network_run/ABLATION_OFF.json\nPY=/home').replace('$PY compare_shadow479.py','$PY diagnose_envelope479.py\n$PY compare_shadow479.py');(T/'finish_shadow479.sh').write_bytes(s.encode())
s=(T/'diagnose_material476.py').read_text(encoding='utf-8').replace('476','479').replace('phase479_20261107','phase479_20261108').replace('material','envelope').replace('chroma_mode','morphology_mode').replace('CONTRAST_DIAG','ENVELOPE_DIAG')
s='\n'.join(line for line in s.splitlines() if not line.startswith("result['kernel_abs_mean']") and not line.startswith("result['temperature']"))+'\n'
(T/'diagnose_envelope479.py').write_bytes(s.encode())
