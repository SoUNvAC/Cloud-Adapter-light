"""Once-only delivery sealing; original audit outputs/code lock are preserved."""
from phase316_common import *
def main():
    assert load(ROOT/'STATUS.json')['status']=='completed' and not (ROOT/'DELIVERY_LOCK.json').exists()
    assert load(ROOT/'INDEPENDENT_ACCEPTANCE.json')['status']=='passed'
    code=ROOT/'delivery_code';code.mkdir()
    for p in (R/'src/tools').glob('*phase316*.py'):shutil.copyfile(p,code/p.name)
    a=load(ROOT/'AUDIT_COMPLETE.json');stages=load(ROOT/'STAGES.json')
    # Keep original rendered output and all scientific audit files byte-exact.
    for name in ['RESULT.md','phase316.md']:shutil.copyfile(ROOT/name,ROOT/('PRE_DELIVERY_'+name))
    with (ROOT/'phase316.md').open('a',encoding='utf-8') as f:
        f.write(f"\n- 实测资源：核心审计{a['cpu_seconds']:.3f}秒、峰值RSS{a['resources']['peak_RSS_bytes']}字节；三个CPU阶段合计{sum(s['seconds'] for s in stages):.3f}秒墙钟。独立复核及报告均退出0，RAM/磁盘均满足4GiB空闲线。\n")
        f.write('- 同步补记：GitHub SSH22多次握手关闭，HTTPS无现有凭据（禁用交互提示），SSH443首次关闭；诊断SSH22曾认证成功，Windows OpenSSH直连官方SSH443最终推送成功。一次冗余同SHA跳板同步因banner超时失败，未触发任何重复分析。以上均为Git同步，不计作新资料HTTP请求；完整账见DELIVERY_NETWORK.json。\n')
        f.write('- 交付代码：原RUN_LOCK及code/快照保持；delivery_code/另存最终备份/核验脚本，备份预算为摘要/代码/仓库对象等额外保留100MB，上限仍300MB，未重复审计或更改数据结论。\n')
    write(ROOT/'DELIVERY_NETWORK.json',dict(official_metadata_requests=0,official_metadata_response_bytes=0,code_sync_failures=['gzs firstgitpull GnuTLS handshake; succeeded retry','local SSH22 pushes connectionclosed multiple times','HTTPS push no existing credentials; interactiveprompt disabled; no login requested','firstSSH443 push via defaultssh connectionclosed','explicitWindowsSSH22 push connectionaborted duringbanner','ProxyJump attempt connectionclosed','redundant sameSHA explicitjump attempt bannerexchange timeout'],successful_route='local WindowsOpenSSH -> ssh.github.com:443, HostKeyAlias github.com, StrictHostKeyChecking yes; normal gzs ff-onlypull',analysis_retries=0,external_data_downloads=0))
    original=load(ROOT/'RUN_LOCK.json');originals={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.md','.py','.log']}
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());assert 4*size+100000000<=300000000
    write(ROOT/'DELIVERY_LOCK.json',dict(status='sealed',original_run_lock_sha256=sha(ROOT/'RUN_LOCK.json'),original_run_git=original['git'],original_audit_code=original['code'],files=originals,new_science_reads=0,metadata_requests=0,extra_persistence_reserve_bytes=100000000,raw_bytes_before_lock=size))
    print('Phase316 sealed; original audit retained, final backup code pinned; no rerun')
if __name__=='__main__':main()
