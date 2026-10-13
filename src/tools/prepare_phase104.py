"""Freeze exact 96 original objects and future question before image access."""
import argparse
import xml.etree.ElementTree as ET
from phase104_common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--storage',choices=['remote','local'],required=True);a=ap.parse_args()
    assert not RESULT.exists(), 'Preserve previous attempt'
    members=read(REPO/'src/research_results/phase103_20261013/member_grid_metadata.json')
    metadata=read(REPO/'src/research_results/phase102_20261013/inputs/candidate_products_metadata.json')
    metadata={r['product_id']:r for r in metadata if r['source']=='PixBox5036991'}
    acq=REPO/'outputs/phase102/phase102_20261013/acquired'
    listings=[]
    for p in sorted(acq.glob('gcs_pixbox_*.json')):
        listing=read(p);items=listing['items'];products={i['name'].split('.SAFE/')[0].split('/')[-1] for i in items}
        assert len(products)==1
        listings.append((next(iter(products)),p,items))
    byproduct={pid:(p,items) for pid,p,items in listings}
    objects=[]; products=[]
    for m in members:
        pid=m['product_id'];p,items=byproduct[pid];idx=p.stem.split('_')[-1]
        xml=acq/('pixbox_'+idx+'_product.xml');root=ET.fromstring(xml.read_bytes())
        values=lambda tag:[e.text for e in root.iter() if e.tag.split('}')[-1]==tag]
        assert values('PRODUCT_URI')==[pid+'.SAFE']
        assert float(values('QUANTIFICATION_VALUE')[0])==10000
        assert not any('OFFSET' in e.tag.split('}')[-1] for e in root.iter())
        meta=metadata[pid]
        assert float(meta['quantification'])==10000 and not meta['offset_elements']
        special={k:int(v) for k,v in meta['special_values'].items()}
        assert special=={'NODATA':0,'SATURATED':65535}
        for band in BANDS:
            hits=[i for i in items if '/IMG_DATA/' in i['name'] and i['name'].endswith('_'+band+'.jp2')]
            assert len(hits)==1,(pid,band,len(hits))
            o=hits[0];expected=meta['band_object_metadata'][band]
            assert int(o['size'])==expected['bytes'] and o['md5Hash']==expected['md5_base64'] and o['generation']==expected['generation']
            objects.append(dict(product_id=pid,band=band,name=o['name'],generation=o['generation'],bytes=int(o['size']),
                                md5_base64=o['md5Hash'],path=pid+'/'+band+'.jp2'))
        products.append(dict(**m,quantification=10000,offset=0,special_values=special,
                             product_xml_sha256=sha(xml),gcs_listing_sha256=sha(p)))
    assert len(objects)==96 and sum(o['bytes'] for o in objects)==EXPECTED_BYTES
    rows=coords(REPO/'data/pixbox_s2_cmix_5036991/phase103_20261013/coordinate_metadata.csv')
    assert {r['product_id'] for r in rows}=={m['product_id'] for m in members}
    original=read(REPO/'src/research_results/phase93_20261012/two_model_lock.json')
    assert all(original['models'][n]['model_sha256']==h for n,h in MODEL_SHA.items())
    RESULT.mkdir(parents=True)
    (RESULT/'.gitattributes').write_text('* -text whitespace=cr-at-eol\n',encoding='ascii')
    write(RESULT/'object_manifest.json',dict(products=products,objects=objects,count=96,bytes=EXPECTED_BYTES),True)
    chain=['src/tools/run_phase89.py','src/tools/recover_phase89.py','src/tools/phase89_math.py',
           'src/configs/protocol/phase65a_explore_rgb.py','src/configs/protocol/phase64_source_parent_rgb.py',
           'src/configs/protocol/phase22_clean_v8_l1c.py','src/cloud_adapter/datasets/phase65_catalogue.py']
    chain+=['src/configs/light/cloud_adapter_dinov2_s_mask2former_lite_q25_d2_p2_l1c.py']
    operator=dict(native10_to20='aligned 2x2 arithmetic mean; all four contributors valid',
        native20='unchanged B11/B12',DN_to_TOA='exact Phase89: native DN float32 / XML quantification; float64 aligned2x2 TOA mean then float32; offset zero',
        invalid='nonfinite OR XML NODATA=0 OR SATURATED=65535; all six bands common support',
        TOA_clip=False,RGB='B04/B03/B02 TOA float32 *255; reverse to BGR; original bgr_to_rgb preprocessor',
        window=dict(size=1020,stride=1020,origin='original upper left',padding='right/bottom zero AND invalid',
                    selection='all 9500 fixed coordinates, regardless of reference label'),
        resize='mmcv.transforms.Resize(scale=(512,512)), default bilinear cv2; PackSegInputs',
        output='original MMSeg postprocess_result restores1020 with model align_corners, crop true edge; no blend',
        runtime_chain=dict(mmcv='2.1.0',mmseg='1.2.2',
            Resize_class_source_sha256='ce4708c3f06cda614741471cc21d84fe6d5d4212628da4c9e57ca392b793323e',
            postprocess_result_source_sha256='cf8f6e27eab74d4b3ad1347f9e8d50da7c252ce177f803fa142c8969c460fdb7'),
        original_chain_sha256={p:sha(REPO/p) for p in chain},
        point='exact containing20m cell; no neighboring prediction average',windows_materialized=False,
        preparation='fixed lazy window index + validated original raster sources; no duplicated dense float caches')
    write(RESULT/'input_operator_lock.json',operator,True)
    lock=dict(created_utc=now(),status='frozen_before_image_download_or_read',storage=a.storage,
        object_manifest_sha256=sha(RESULT/'object_manifest.json'),operator_sha256=sha(RESULT/'input_operator_lock.json'),
        coordinate_sha256=COORD_SHA,reference_lock_sha256=sha(REPO/'src/research_results/phase103_20261013/prelabel_lock.json'),
        protocol_sha256=sha(REPO/'src/research_plans/PHASE104_PIXBOX_INPUT_ACCEPTANCE_20261013.md'),
        original153_models=MODEL_SHA,model_path='/home/scv/shared/phase93_20261012/delivery/{name}.pkl',
        future_primary='G7_153 minus G6_153 group-equal sampled-point AP; conditional Source-score increment',
        future_bootstrap=dict(paired_related_groups=10000,seed=105013,confidence=.95,lower_strictly_above=0,width_at_most_pp=20),
        minimum_support=dict(groups=8,positive_per_group=5,negative_per_group=20),
        keep_AP_undefined_tiles=['19GBQ','48NUG'],no_calibration=True,no_extra_products=True,
        point_reference_boundary='point labels local Phase103 only; images/no-label inputs on one machine',
        CPU_sample_rule='first48 SHA256(PHASE104_INPUT|point_id) from all9500',CPU_point_ids=sample48(rows),
        cumulative_actual_transfer_cap_bytes=CAP,expected_payload_bytes=EXPECTED_BYTES,
        new_fits=0,network_forwards=0,performance_evaluations=0,
        next_phase='Separate Phase105 protocol and authorization; no automatic evaluation',
        limitations=['selective sampled conservative reference; not dense truth','unknown base pretraining independence',
                     'no full-scene FPR/tail safety/dense mIoU claim','does not reopen failed branches'],
        code_sha256={str(p.relative_to(REPO)).replace('\\','/'):sha(p) for p in sorted((REPO/'src/tools').glob('*phase104*.py'))})
    write(RESULT/'task_lock.json',lock,True)
    print('Frozen 16 products / 96 objects /',EXPECTED_BYTES,'bytes; no image accessed')

if __name__=='__main__':main()
