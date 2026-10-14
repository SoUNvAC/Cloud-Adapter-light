"""Before construction, anchor the planned inputs to historical official file SHAs."""
from phase118_common import REPO,ROOT,load,sha,write


def main():
    assert (ROOT/'PREPARE_EXIT_CODE').read_text().strip()=='0'
    assert not (ROOT/'RUN_PID').exists()
    asset_lock=load(REPO/'src/research_results/phase118_20261014/NATIVE_ASSET_SHA_LOCK.json')
    plan=load(ROOT/'INPUT_PATCH_PLAN.json')
    checked=0
    for row in plan['rows']:
        for binding in row['input_bindings']:
            if binding['band'] not in ('B04','B03','B8A'):continue
            key=row['product_id']+'/'+binding['band']
            assert binding['sha256']==asset_lock['files'][key]['sha256']
            assert binding['native_grid']==row['product']['grid20' if binding['band']=='B8A' else 'grid10']
            checked+=1
    assert checked==96 and len(plan['rows'])==32
    assert plan['planned_calls_including_11_construction_and_8_repeats']<=20000
    write(ROOT/'CPU_HISTORICAL_INPUT_SHA_ACCEPTANCE.json',{'status':'passed','binding_checks':checked,
          'historical_native_assets':48,'plan_sha256':sha(ROOT/'INPUT_PATCH_PLAN.json'),
          'official_historical_asset_lock_sha256':sha(REPO/'src/research_results/phase118_20261014/NATIVE_ASSET_SHA_LOCK.json'),
          'model_calls':0})
    print('Historical48 native asset hashes /96 planned bindings accepted before models')


if __name__=='__main__':main()
