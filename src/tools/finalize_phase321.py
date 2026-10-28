"""Format already-computed remote results after byte-level local backup acceptance."""
from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parents[2];L=R/'outputs/phase321/phase321_20261028';B=L/'remote_backup';S=R/'src/research_results/phase321_20261028'
def main():
 receipt=json.loads((L/'BACKUP_RECEIPT.json').read_text());assert receipt['status']=='completed'
 for n,e in receipt['files'].items():
  b=(B/n).read_bytes();assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
 v=json.loads((B/'recovery01/RESULT.json').read_text());a=json.loads((B/'ACCEPTANCE.json').read_text());assert a['status']=='passed'
 S.mkdir(exist_ok=True)
 for n in ['COUNTS.csv','TARGET_OVERLAP.csv','MATCH_PAIRS.csv','SOURCE_SENSITIVITY.csv','RESULT.json','INPUT_LOCK.json','INPUT_FINAL_SHA.json','SYNTHETIC_CHECKS.json']:
  shutil.copyfile(B/'recovery01'/n,S/n)
 shutil.copyfile(B/'ACCEPTANCE.json',S/'ACCEPTANCE.json');shutil.copyfile(B/'FAILURE.json',S/'INITIAL_FAILURE.json');shutil.copyfile(L/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
 lines=['# Phase321：Source来源重叠与零训练敏感性分析','','## 目标与范围','','一次性核对Source官方train8490、选模val535与Catalogue513、ALCD37、PixBox29。V4降为参考，本轮未审OCM1751及完整V4来源，不为R4原指标签发外部独立性证明。Source的DINO预训练覆盖与历史目标开发暴露仍未知。官方train名单是保守的可采样全集，未按4000步重建实际采样清单。','','## 改动与事先规则','','同tile同UTC日列同景候选，精确时刻另列；同tile异日列地理重复；原多边形在同一目标UTM投影相交严格>1平方米列空间重叠，无缓冲。ALCD使用含ignore区域的完整影像网格，PixBox使用整幅母影像网格，所以空间排除相对实际标注/采样点保守。同景候选不自动证明裁剪像元泄漏。三种flag可同时成立，不能把列数相加当总数。','','scene_clean排除同景候选和未知；strict_clean再排除同tile异日或空间重叠。train/val分别统计，敏感性另给两者联合排除。5%为预设同景候选占比小/大的分界。','','## 核对结果','','|目标|Source角色|产品数|同景候选|同tile异日|空间重叠|未知|scene_clean|strict_clean|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in v['counts']:lines.append(f"|{r['dataset']}|{r['split']}|{r['products']}|{r['same_scene_candidates']}|{r['same_tile_different_day']}|{r['spatial_overlap']}|{r['unknown']}|{r['scene_clean']}|{r['strict_clean']}|")
 lines+=['','以上分母为完整目标池；具体已评价样本见TARGET_OVERLAP.csv的evaluated列，不能将完整池重叠比例套到已评价子集。每个命中对、面积及两侧面积比例均见MATCH_PAIRS.csv。','','train+val联合排除（按产品去重）：','']
 for r in a['combined_counts']:lines.append(f"- {r['dataset']}：同景候选{r['same_scene_candidate']}，同tile异日{r['same_tile_different_day']}，空间重叠{r['spatial_overlap']}，strict_clean {r['strict_clean']}/{r['products']}；已评价{r['evaluated_products']}产品中排除{r['evaluated_excluded']}。")
 lines+=['','## 同一批预测的敏感性','','|目标|排除Source角色|子集|保留相关组|组均AP %|组均mIoU %|池化mIoU %|组均FPR %|','|---|---|---|---:|---:|---:|---:|---:|']
 def fmt(x):return 'NA' if x is None else f'{x:.6f}'
 for r in v['sensitivity']:lines.append(f"|{r['dataset']}|{r['roles']}|{r['policy']}|{r['retained_groups']}|{fmt(r['macro_AP_pct'])}|{fmt(r['macro_mIoU_pct'])}|{fmt(r['pooled_mIoU_pct'])}|{fmt(r['macro_FPR_pct'])}|")
 lines+=['','Catalogue复用Phase317 Source32逐景混淆和统计，全集与原指标在1e-6个百分点内复现；PixBox复用Phase112 Source16产品/10相关组的统计，部分成员被排除时整组剔除。缺少充分统计的池化AP/FPR不重建，不将组均AP当池化AP。ALCD没有已冻结Source预测，性能NA。','','## 网络、止损、失败与备份','','本地GitHub SSH443推送成功；远端沿用已知GitHub TLS不稳定的既有路线，以逐SHA验证bundle执行真实git pull，未改origin。首次合成自检因GDAL已返回MultiPolygon而断言失败，尚未读取Source名单或计算真实匹配；原记录保留，独立recovery01只修几何类型处理。规则、面积阈值及池未变，实际匹配仅一次。',f"独立恢复实际CPU墙钟{v['elapsed_seconds']:.3f}秒；峰值RSS {v['resources']['maxrss_bytes']}字节。CPU墙钟上限1800秒、进程地址空间8GiB、宿主空闲4GiB、双机新增200MB。零训练/模型加载/前向/像元读取/下载。所有原输入、执行代码及全部回传成员SHA通过；详见INPUT_FINAL_SHA、ACCEPTANCE及BACKUP_RECEIPT。",'','## 如何使用','','仅可将无命中的已评价子集描述为“相对已审Source train/val未检出来源重叠的历史目标评价”；这不恢复盲测身份，也不认证骨干预训练独立。若某子集存在同景候选，完整集与去除后结果并列，来源范围写明。V4完整训练集合未审，R4 Catalogue82.14及既有PixBox结果保留为历史参考，不能用Source无重叠给V4背书。OCM hard-negative若与目标重叠，首先影响误报改善的解释，本轮对此保持未知。Omni教师版本选择不得以本轮Source审计替代其训练覆盖核查。','','本轮结算，不启动下一phase。']
 (S/'phase321.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 # Git normalizes text to LF. Normalize publication only; preserve raw backup bytes.
 for p in S.iterdir():
  if p.is_file() and p.name!='DELIVERY_SHA256.json':p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 pins={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in S.iterdir() if p.is_file() and p.name!='DELIVERY_SHA256.json'}
 (S/'DELIVERY_SHA256.json').write_bytes((json.dumps(pins,indent=2)+'\n').encode());print(str(S/'phase321.md'))
if __name__=='__main__':main()
