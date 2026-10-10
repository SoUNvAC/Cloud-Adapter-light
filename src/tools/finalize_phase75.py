"""Verify transferred metadata and write the local ready/pending delivery."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from phase75_metadata import select,parse_xml,cube_chain,mirror_digest,LIMIT
import csv


def main(root,repo):
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    hashes=read(root/'artifact_hashes.json')
    for name,h in hashes.items():assert sha(root/name)==h,name
    corrected=root/'reparse01'
    report=read(corrected/'metadata_audit.json');selection=read(root/'selected_products.json');lock=read(root/'input_lock.json')
    repaired=read(corrected/'input_lock.json')
    for name,h in read(corrected/'artifact_hashes.json').items():assert sha(corrected/name)==h
    assert report['reparse_input_lock_sha256']==sha(corrected/'input_lock.json')
    assert repaired['original_artifact_hashes_sha256']==sha(root/'artifact_hashes.json')
    assert repaired['new_http_requests']==0 and report['additional_HTTP_bytes']==0
    original_revision=(root/'initial_run_logs/GIT_COMMIT').read_text().strip()
    assert len(original_revision)==40 and all(c in '0123456789abcdef' for c in original_revision)
    assert report['input_lock_sha256']==sha(root/'input_lock.json')
    assert lock['selection_sha256']==sha(root/'selected_products.json')
    for name,h in lock['code_sha256'].items():
        b=subprocess.check_output(['git','show',original_revision+':src/tools/'+name],cwd=repo)
        assert hashlib.sha256(b).hexdigest()==h,name
    for name,h in repaired['code_sha256'].items():
        b=subprocess.check_output(['git','show','HEAD:src/tools/'+name],cwd=repo)
        assert hashlib.sha256(b).hexdigest()==h,name
    protocol=subprocess.check_output(['git','show','HEAD:src/research_plans/PHASE75_GEOMETRY_INTAKE_20261010.md'],cwd=repo)
    assert hashlib.sha256(protocol).hexdigest()==lock['protocol_sha256']
    data=repo/'data/sentinel2_cloud_mask_catalogue_4172871'
    split=repo/'outputs/phase65/work_dirs/phase65a/split_lock.json'
    assert sha(split)==lock['original_split_sha256']
    assert sha(data/'README.pdf')==lock['official_pdf_sha256'] and sha(data/'classification_tags.csv')==lock['official_tags_sha256']
    tags={r['scene']:r for r in csv.DictReader((data/'classification_tags.csv').open())}
    assert select(read(split),tags)==selection
    requests=read(root/'request_log.json');total=sum(r['received_bytes'] for r in requests)
    assert total==report['public_archive_response_bytes']<=LIMIT
    for req in requests:
        assert '.jp2' not in req['url'] and req['url'].startswith('https://storage.googleapis.com/')
        if 'verified_sha256' in req:
            path=root/req['destination'];md5,digest=mirror_digest(path)
            assert md5==req['server_md5_base64'] and digest==req['verified_sha256']
            assert path.stat().st_size==req['expected_size']
    for row in report['rows']:
        assert row['product'] in [x['product'] for x in selection['selected']]
        assert cube_chain(data,row['product'])==row['cube_chain']
        assert row['cube_mapping_status'] in ['pending','mismatch'] and not row['direction_geometry_ready']
        if row.get('xml_object_metadata'):
            folder=root/row['product'];objects=read(folder/'object_listing.json')['items']
            parsed=parse_xml((folder/'MTD_MSIL1C.xml').read_bytes(),(folder/'MTD_TL.xml').read_bytes(),
                row['product'],row['identity']['archive_prefix'],objects)
            for key,value in parsed.items():assert row[key]==value,(row['product'],key)
    assert report['products']==len(selection['selected'])<=3
    for key in ['solar_metadata_verified','source_grid_verified']:assert report[key]==sum(r[key] for r in report['rows'])
    assert report['new_original_rasters_downloaded']==report['new_predictions']==report['new_fits']==0
    result=dict(status='verified',remote_artifacts_sha_verified=len(hashes),
        fixed_three_product_selection_reproduced=True,public_http_response_bytes=total,
        XML_lengths_MD5_SHA_and_reparse_verified=True,cube_header_and_shapefile_chain_reproduced_locally=True,
        deployed_code_and_protocol_match_git_blobs=True,no_work_dirs_sync=True)
    with (root/'LOCAL_VERIFIED.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    lines=['# Phase75：方向几何输入可行性审计','','## 已核验','',
        f"按固定哈希规则从153个fit代表中选择公开Shadow比例>0的前三景（合格{selection['eligible_products']}景）。只获取精确原L1C产品XML、所属tile XML及目录清单，新增公开HTTP响应体共{total:,} bytes，上限20,000,000 bytes。0栅格下载、0网络预测、0拟合、0方法优劣评价。名单先于请求保存。",'',
        '| Tile | 太阳方位角° | 太阳天顶角° | 原始CRS | 太阳元数据 | 原始格网 | cube映射 | 方向几何ready |',
        '|---|---:|---:|---|---|---|---|---|']
    for r in report['rows']:
        s=r.get('solar',{});g=r.get('source_grid',{})
        lines.append(f"| {r['product'].split('_')[5]} | {s.get('azimuth','missing')} | {s.get('zenith','missing')} | {g.get('crs_code','missing')} | {r['solar_metadata_verified']} | {r['source_grid_verified']} | {r['cube_mapping_status']} | false |")
    lines+=['', '角度来自精确原始tile的Mean_Sun_Angle，单位为度；平均tile角不代表逐像元角。角度网格及步长存在性保存在逐景报告和原XML中，未插值到cube。太阳方位按节点当地北方向顺时针定义，天顶角相对椭球法线；未把当地北方向直接视为UTM格网北或图像行方向。[Copernicus角度定义](https://sentiwiki.copernicus.eu/web/s2-processing)',
        '', '完整身份链在逐景metadata_audit.json：Catalogue完整产品ID与档案SAFE产品名/产品XML URI精确一致；Granule标识与XML波段引用和档案对象相连，另核对tile、处理基线。产品名的采集起始时间与tile SENSING_TIME分别保留，它们对应不同层级，不能只凭同日期同tile匹配。末尾时间戳是产品区分/生成标识，不用它替代观测时间。',
        '', '原始格网按XML记录10/20/60m的尺寸、ULX/ULY/XDIM/YDIM和CRS；B04/B03/B02/B08为原生10m，B11/B12为20m。这里核验的是元数据及引用链，没有下载JP2来核验栅格内容。量化因子、nodata/饱和值和offset是否存在均保存在逐景radiometry，不为缺失字段补猜测值。',
        '', '## 未核验','',
        '三景NPY头均为1022×1022×13、float32，按官方20m声明宽高20,440m；shapefile范围均为23,040m，对应1152×1152。两者不能直接作为同一个完整范围。shapefile CRS与原始tile CRS的匹配也不能消除裁切偏移、行列方向和重采样格网的不确定性。',
        '', '官方README第1/4/6/9页说明20m、非20m双线性重采样、1152带边框窗口、64像元边框、原计划1024以及最终去边缘后的1022。由这些信息推测对称向内裁65个20m像元是一个候选，但本轮没有像素一致性证据，因此不指定cube affine。每景候选位置与缺项保存在cube_chain中。',
        '', '所以太阳角/原始格网可用，不等于方向几何实验已就绪。cube映射仍pending，direction_geometry_ready=false。输入链未闭合不是几何方法实验失败；Phase74停止结论、旧独立评价失败结论均不改写。',
        '', '## 下一步最小需求','',
        '另立像素验证协议后，仅需先获取这3个精确产品的B11原生20m与B04原生10m：B11检查裁切/格网偏移与数值对应；B04检查双线性插值、像元中心约定及裁切/重采样顺序。先冻结候选、分布采样位置、数值容差和歧义拒绝规则，再用影像自身比较；不用标签或模型结果配准。此步骤本轮没有执行，也未下载DEM或投影云影。',
        '', '交付：selected_products.json固定名单；每产品原MTD_MSIL1C.xml/MTD_TL.xml与目录清单；**reparse01/metadata_audit.json和.csv为最终逐景审计**；request_log.json请求URL/UTC/长度/MD5/generation/SHA；minimum_pixel_validation_plan.json计划；输入/代码锁及校验证据。初次metadata_audit报告及initial_run_logs保留：初版解析器错误地要求长granule ID等于紧凑目录名，已改为核对产品granuleIdentifier=tile TILE_ID，并核对IMAGE_FILE与同目录MTD_TL.xml的显式引用；重新解析没有新增下载。只从shared传回，没有同步work_dirs。','']
    (root/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);a=p.parse_args();main(a.root,a.repo)
