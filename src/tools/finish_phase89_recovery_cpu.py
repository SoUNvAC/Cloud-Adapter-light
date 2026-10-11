"""Finish existing outputs only; substitute independent CPU AP, never infer."""
import ast,inspect
import run_phase89 as p
from phase89_cpu_ap import self_test

class CPUOnlyDependency(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        if node.module=='sklearn.metrics':node.module='phase89_cpu_ap'
        return node
    def visit_keyword(self,node):
        if node.arg=='sklearn_AP_matches':node.arg='independent_score_bin_AP_matches'
        if node.arg=='pooled_sklearn_AP':node.arg='pooled_independent_score_bin_AP'
        return self.generic_visit(node)

def transformed(source,filename,namespace):
    tree=CPUOnlyDependency().visit(ast.parse(source));ast.fix_missing_locations(tree)
    exec(compile(tree,filename,'exec'),namespace)

def main():
    p.ROOT=p.SHARED/'phase89_recovered_20261011';root=p.ROOT;d=root/'delivery'
    p.locked();plan=p.read(root/'recovery_lock.json')
    assert not (root/'COMPLETED.json').exists() and not (d/'CPU_VERIFIED.json').exists()
    for name,count in [('Source',270),('MsRE',324)]:
        index=p.read(root/('scores_'+name)/'index.json')
        assert index['forward_count']==count and index['unique_block_count']==324
        assert index['before_state_sha256']==index['after_state_sha256']
    p.read(d/'phase89_report.json');p.read(root/'calibrated/index.json')
    p.write(d/'CPU_AP_DEPENDENCY_ADDENDUM.json',dict(created_utc=p.now(),tests=self_test(),
        reason='Existing conda has no sklearn; preserve all inference outputs and frozen production metrics.',
        algorithm='Non-interpolated AP: positive mass in each tied score bin times cumulative precision at that threshold.',
        production_AP_unchanged=True,no_forward_no_fit_no_tuning=True,
        helper_sha256=p.digest(__file__),oracle_sha256=p.digest(p.REPO/'src/tools/phase89_cpu_ap.py')))
    # Reuse all frozen CPU grid/calibration/confusion checks verbatim. Only the
    # absent dependency import and provenance field names change in the AST.
    ns=dict(p.__dict__);transformed(inspect.getsource(p.verify),str(p.REPO/'src/tools/run_phase89.py')+':CPU_only_dependency',ns)
    ns['verify']()
    finalfile=p.REPO/'src/tools/finalize_phase89.py';final={'__name__':'phase89_cpu_finalizer','__file__':str(finalfile)}
    transformed(finalfile.read_text(encoding='utf-8'),str(finalfile)+':CPU_only_dependency',final)
    original_write=p.write;original_summary=final['summarize']
    def honest_write(path,value):
        if path.name=='COMPLETED.json':
            value.pop('network_forwards',None)
            value.update(unique_network_block_model_pairs=648,new_recovery_forwards=594,reused_blocks=54,
                authorized_damaged_block_repeats=112,historical_forward_total=None,recovery_lock_sha256=p.digest(root/'recovery_lock.json'),
                cpu_AP='independent_score_bin_formula; sklearn unavailable')
        if path.name=='input_and_runtime_index.json':value.update(recovery=plan,source_attempt='run02 disk failure; authorized recovery preserving old files')
        original_write(path,value)
    def honest_summary(path):
        original_summary(path)
        f=path/'README.md';text=f.read_text(encoding='utf-8').replace('sklearn AP','独立分数分组AP')
        text+='\n\n恢复说明：复用54块，授权补算112块损坏输出并继续482块无输出任务，本次新增594次前向。历史总前向次数因日志损坏无法确定。复用块通过恢复时结构与输入身份检查，但没有故障前输出SHA封存。准备数据硬链接复用，故障原目录保留；无训练、拟合、调阈值。\n'
        text+='\nCPU复核说明：指定conda缺少sklearn，未安装或修改环境。使用独立按分数分组的非插值AP公式，已通过3个手算例及60个暴力阈值例；逐组和池化AP与冻结生产实现比较。没有声称实际运行sklearn。全部概率、阈值、mask和生产指标计算保持原样。\n'
        f.write_text(text,encoding='utf-8')
    p.write=honest_write;final['summarize']=honest_summary;final['main']()

if __name__=='__main__':main()
