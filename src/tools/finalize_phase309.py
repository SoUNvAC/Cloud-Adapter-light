"""Publish complete verified documentary evidence; local work is file copying and SHA only."""
import shutil
from phase309_common import *
def main():
    receipt=load(LOCAL/'BACKUP_RECEIPT.json');assert receipt['all_members_SHA_verified'];root=LOCAL/'remote_backup'
    for n,v in load(root/'BACKUP_FILES.json').items():assert sha(root/n)==v['sha256'],n
    S.mkdir(parents=True,exist_ok=True)
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.csv','.json','.md','.log'):shutil.copyfile(p,S/p.name)
    shutil.copytree(root/'evidence',S/'evidence',dirs_exist_ok=True)
    shutil.copyfile(LOCAL/'BACKUP_RECEIPT.json',S/'BACKUP_RECEIPT.json')
    r=load(root/'RESOURCE.json');n=load(root/'NETWORK_EVIDENCE.json')
    notes=['# Phase309 实验记录','',
        '- 目标：闭环Phase308主参考语义、备用mask及空间窗口来源三个缺口，原历史文件只读。',
        '- 改动：src/tools新增隔离PDF工具、9页全文/渲染、固定三产品shapefile几何/DBF/元数据、语义证据表、主备关系及问题判定；不重算历史指标。',
        '- 环境：gzs cloud-lite-pt210启动独立CPU文档工具，pypdf5.9.0与pypdfium2 4.30.0仅装本轮toolenv；生产包清单前后一致。本机仅编辑/文件SHA及备份。',
        '- 止损线：CPU≤8GiB、累计执行≤1小时、新持久含备份≤500MB、宿主空闲内存/磁盘≥4GB；拟合/训练/神经前向0；证据网页≤10份、下载≤20MB。',
        f"- 资源：{r['status']}，累计{r['total_seconds']:.3f}秒，峰值CPU进程树{r['peak_CPU_tree_bytes']}字节，最低宿主空闲{r['minimum_available_bytes']}字节。",
        f"- 网络：代码本地push后远端pull；PDF工具轮包{n['wheel_bytes']}字节，一手资料请求{n['documents_attempted']}份、响应正文{n['document_bytes']}字节；实际失败见NETWORK_EVIDENCE/FAILURES，未盲重试。",
        '- 结果：README实际读通；类顺序/阴影缺标/薄云偏好、alt双标用途和1152→1024→1022窗口链已澄清。固定3组均MAIN且无alt；雪冰逐像元判据及逐景最终仿射未获直接证据，四角/中心残差NA。',
        '- 限制：65像元/1300m每侧是文档链条件推导，非已核验仿射；参考一致性不是真值准确率，保持所有历史数值/参考不变，不自动加溯源轮次或训练。',
        f"- 备份：{receipt['files']}成员，归档{receipt['bytes']}字节，SHA256={receipt['sha256']}；包括隔离工具/轮包、全部原文页证据、三产品空间资料和完整锁，逐文件SHA通过。",
        '- 检查：本轮按路线无需定时任务，交付后即结束，不启动下一phase。']
    (S/'phase309.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    write(S/'SUMMARY_SHA256.json',dict(status=r['status'],files={p.relative_to(S).as_posix():sha(p) for p in S.rglob('*') if p.is_file() and p.name!='SUMMARY_SHA256.json'},code={p.relative_to(R).as_posix():sha(p) for p in (R/'src/tools').glob('*phase309*.py')}))
    print(r['status'])
if __name__=='__main__':main()
