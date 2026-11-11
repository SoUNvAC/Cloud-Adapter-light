from pathlib import Path
T=Path(__file__).parent;s=(T/'boundary_shadow480.py').read_text(encoding='utf-8')
a=s.index(" roots=");b=s.index('\n rows=',a)
s=s[:a]+" roots={'off':R/'ablation_off'/f'step_{step}','480':R/'independent'/f'step_{step}'}"+s[b:]
s=s.replace(" for name,cm in cms.items():", " off=json.loads((R/'ABLATION_OFF.json').read_text());assert cms['off'].tolist()==off['results'][0]['confusion']\n for name,cm in cms.items():")
s=s.replace('delta480minus450','delta_on_minus_off').replace("totals['450']","totals['off']").replace('BOUNDARY_DIAG','EFFECT_BOUNDARY_DIAG').replace('BOUNDARY_PER_IMAGE','EFFECT_BOUNDARY_PER_IMAGE')
(T/'boundary_effect480.py').write_bytes(s.encode())
p=T/'finish_shadow480.sh';s=p.read_text(encoding='utf-8');s=s.replace('$PY compare_shadow480.py','$PY boundary_effect480.py A\n$PY boundary_effect480.py B\n$PY compare_shadow480.py');p.write_bytes(s.encode())
