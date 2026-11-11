"""Metadata-only qualification update; never opens imagery, scores or label payloads."""
from pathlib import Path
import csv,json,datetime,re,hashlib,collections
B=Path(__file__).resolve().parents[2];O=B/'outputs/phase248/phase248_20261031';O.mkdir(parents=True,exist_ok=True);evidence={}
def read(rel):
 p=B/rel;raw=p.read_bytes();evidence[rel]=hashlib.sha256(raw).hexdigest()
 return list(csv.DictReader(raw.decode('utf-8-sig').splitlines())) if p.suffix=='.csv' else json.loads(raw.decode('utf-8-sig'))
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 old=read('outputs/phase243/phase243_20261031/CANDIDATES.csv');history=read('outputs/phase102/phase102_20261013/inputs/history_positions_only.json')['records'];irish=read('outputs/phase245/phase245_20261031/IRISH_ALL_SCENES.csv');groups=read('outputs/phase245/phase245_20261031/L7_CANDIDATE_LOCK.json')['groups'];extra=read('outputs/phase244/phase244_20261031/CANDIDATES.csv');geom=read('outputs/phase245/phase245_20261031/GEOMETRY_REFINEMENT.json')['results'];seen=read('outputs/phase247/phase247_20261031/DATA_SUPPORT_ACCEPTANCE.json');read('outputs/phase247/phase247_20261031/INFERENCE_COMPLETE.json')
 seen_ids={v['scene'] for v in seen['scenes']};groupof={s:g for g in groups for s in g['members']};geobyid={v['id']:v for v in geom};byid=collections.defaultdict(list)
 for v in history:
  for sid in re.findall(r'LC8\d{13}LGN\d{2}',v.get('product','')):byid[sid].append(v)
 rows=[]
 def base(pool,sid):
  if sid.startswith('LC8'):
   path,row=sid[3:6],sid[6:9];date=(datetime.date(int(sid[9:13]),1,1)+datetime.timedelta(days=int(sid[13:16])-1)).isoformat()
  else:path=row=date='unknown'
  h=byid.get(sid,[]);bounds=[v.get('wgs84_bounds') for v in h if v.get('wgs84_bounds')]
  return {'pool':pool,'product_id':sid,'WRS_path':path,'WRS_row':row,'date':date,'geographic_bounds_WGS84':json.dumps(bounds[0]) if bounds else 'unknown; WRS identity only','geography_basis':'historical registered bounds; not a new grid audit' if bounds else 'no exact per-scene footprint recovered in this table','history_use':'unknown','historical_spatial_correlation':'unknown','Shadow_coverage':'unknown','unmarked_Shadow_may_be_Surface':'unknown','input_conditions':'unknown','inventory_state':'unknown','group_id':'','qualification':'待核实','reason':''}
 for v in old:
  if v['pool'] not in ['L8','SPARCS']:continue
  r=base(v['pool'],v['id']);r['inventory_state']='existing shared raw/prepared pool; previous metadata inventory';r['input_conditions']='existing L8 RGB B4/B3/B2; prepared patch pipeline; identity-specific grid/Fill audit still required' if v['pool']=='L8' else 'existing 1000px official photo/mask -> 512 model input; target study already used all80'
  if v['status']=='research_exposed':
   r.update(history_use='existing study '+v['role']+'; training/selection/evaluation or repeated analysis',qualification='排除',reason='已参与本条研究开发，不能作为未暴露确认景',historical_spatial_correlation='exact scene identity already registered')
  else:r.update(history_use='not in current42 L8 split; no proof of absence from all unrecorded uses',reason='官方Shadows=no，需独立补足云影参考及历史/地理核验；不是现成云影确认景')
  r['Shadow_coverage']='official Shadows='+v['annotation'] if v['pool']=='L8' else 'official native Shadow+Shadow-over-water mapped to Shadow; published reference not independent truth'
  r['unmarked_Shadow_may_be_Surface']='yes; completeness unresolved' if v['pool']=='L8' and v['annotation']=='no' else 'published annotations; omissions cannot be excluded'
  rows.append(r)
 for v in irish:
  sid=v['id'];r=base('L7_Irish',sid);r.update(WRS_path=v['path'],WRS_row=v['row'],date=v['date'],Shadow_coverage='official Shadows='+v['official_shadow_status'],input_conditions='ETM+ B3/B2/B1, 30m; third-sensor extrapolation, not L8 replication; original same-version imagery/mask required',inventory_state='metadata/link only; no scene payload inventoried before this preparation',history_use='no registered current Source/C train/selection/calibration use found; metadata screening only')
  p=B/f'outputs/phase245/phase245_20261031/{sid}_stac.json'
  if p.exists():
   st=read(str(p.relative_to(B)));items=st.get('items',[])
   if len(items)==1:r['geographic_bounds_WGS84']=json.dumps(items[0]['bbox']);r['geography_basis']='same-date/WRS C2 footprint proxy; not original L1/mask grid'
  g=groupof.get(sid)
  if g:
   r['group_id']=g['group_id'];r['historical_spatial_correlation']=g['reason']+'; phase245 transitive related grouping, 2km rule'
  if sid in seen_ids:
   r.update(qualification='仅作探索',history_use='Phase246 label/input inspection and Phase247 Source/C external evaluation; now exposed',inventory_state='three 2015 originals downloaded/prepared; 2019 counterparts retained',reason='已暴露，不再是确认候选；原191仍未确认，仅64/128限定支持可评价',unmarked_Shadow_may_be_Surface='cannot exclude; 191 ignored, full semantics unresolved')
  elif v['official_shadow_status']!='yes':r.update(reason='Shadows=no，不能充当可靠无云影负样本',unmarked_Shadow_may_be_Surface='yes; not qualified')
  elif g and not g['metadata_spatial_eligible']:r.update(qualification='排除',reason='相关组中有历史几何相交提示；整组按既定保守规则排除',unmarked_Shadow_may_be_Surface='unknown')
  elif g:r.update(reason='元数据候选，尚缺同版本输入/标签语义/有效支持核验；未按模型性能筛选',unmarked_Shadow_may_be_Surface='unknown; annotation audit pending')
  else:r['reason']='related group/identity pending'
  rows.append(r)
 for v in extra:
  sid=v['id'];r=base(v['pool'],sid);r.update(WRS_path=v['path'],WRS_row=v['row'],date=v['date'],geographic_bounds_WGS84=v['bbox'] or 'unknown',geography_basis='same-acquisition C2 STAC proxy',history_use='metadata/point-position screening only; no Source/C performance evaluation found in scoped records',inventory_state='PixBox label/position zip and metadata locally; imagery not accepted' if v['pool']=='PixBox_L8' else 'official description/metadata only; no accepted raster inventory',Shadow_coverage='point-reference Shadow support/counts and grid pending' if v['pool']=='PixBox_L8' else 'declared cloud/shadow description; original mapping/support unverified',input_conditions='L8 B4/B3/B2 compatibility pending original imagery and positions' if v['pool']=='PixBox_L8' else '30m RGB declared, original product/crop/normalization unverified')
  g=geobyid.get(sid);status=g['status'] if g else v['status'];r['historical_spatial_correlation']=status
  if status in ['geometry_flag','historical_spatial_flag']:r.update(qualification='排除',reason='历史空间相关提示，不能当独立确认')
  else:r['reason']='元数据层面保留，原始输入/标签/点位支持未验收；尚非现成确认景'
  rows.append(r)
 assert len({(r['pool'],r['product_id']) for r in rows})==len(rows)
 with (O/'LANDSAT_ELIGIBILITY.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 potential=[g for g in groups if g['metadata_spatial_eligible'] and not any(s in seen_ids for s in g['members'])]
 summary={'rows':len(rows),'by_pool':dict(collections.Counter(r['pool'] for r in rows)),'by_qualification':dict(collections.Counter(r['qualification'] for r in rows)),'confirmation_ready':0,'L8_existing_not_current42_but_Shadow_no':54,'SPARCS_existing_all_study_used':80,'L7_unexposed_metadata_groups':len(potential),'L7_unexposed_metadata_members':sum(len(g['members']) for g in potential),'L7_prepared_now_exposed':sorted(seen_ids),'new_model_calls':0,'candidate_performance_reads':0,'candidate_image_or_label_payload_reads':0,'limits':'scoped registered history only; upstream VFM pretraining/unrecorded manual use unknown'}
 assert len(potential)==21
 save('ELIGIBILITY_SUMMARY.json',summary);save('ELIGIBILITY_EVIDENCE_SHA.json',evidence)
 lines=['# Landsat独立场景资格表（更新至Phase247后）','','**当前现有库存中，确认就绪的独立云影场景仍为0。** 资格表不是新的性能实验，没有读取候选影像/标签像元或模型性能。','','|池|当前资格|','|---|---|','|L8 Biome|现有96景中42景已用于本条研究；余54景官方Shadows=no，不能直接作云影确认或可靠无影负类。历史空间独立性也未逐景补齐。|','|SPARCS|80景全部进入既有60/10/10，未发现额外库存场景。|','|L7 Irish|本次准备并评价的3景已暴露，仅作探索。另23产品/21相关组仍为未暴露元数据候选，但尚未下载验收，不是现成合格样本。|','|PixBox-L8|一个无登记空间相交候选LC81970182015080LGN00；原输入、点位格网和Shadow支持未验收。|','|WHU Cloud|124/033、2018-04-15元数据候选保留；原发布输入、裁块及标签口径未验收。126/035等身份/空间缺口继续保留。|','','每个ID的日期、WRS、已恢复地理范围、历史角色、标签覆盖、输入条件、库存状态和资格见LANDSAT_ELIGIBILITY.csv。unknown字段不补造；日期由产品ID或原表读取，C2足迹只是同采集位置代理。','','现成数据的缺口是：**同条件、未暴露、Shadow覆盖明确、原输入/标签格网已经验收的独立Landsat场景**。不能通过增加seed、重复使用L7三景、把Shadows=no当负样本来填补。优先保持L8条件；若用L7，须明确是跨传感器外推。','','没有按“是否出现失配”筛选候选。元数据筛查不证明基础VFM未预训练见过这些区域，也不排除未记录的人工访问。']
 (O/'LANDSAT_ELIGIBILITY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
