"""Freeze original metadata categories before joining Phase300 student scores."""
import hashlib
import json
from pathlib import Path
import zipfile

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO/'outputs/phase300/phase300_20261020'


def main():
    path = ROOT/'SUBGROUP_MAPPING_LOCK.json'
    assert not path.exists(), 'Subgroup mapping is immutable'
    archive = REPO/'data/pixbox_s2_cmix_5036991/phase103_20261013/original_PixBox-S2-CMIX.zip'
    member = 'PixBox-S2-CMIX/pixbox_sentinel2_cmix_20180425_definitions.txt'
    with zipfile.ZipFile(archive) as source: content = source.read(member)
    surface = ['None of the classes','Forest Winter','Forest','Urban snow covered','Mine','Coast line',
      'Water on ice','Ship canal','Lagoon','Spatially mixed cloud/snow-ice','Road','Burnt area','Wind turbine',
      'Wadden mud/silty sand','Vessel, offshore platform','Spatially mixed water/cloud','Spatially mixed land/cloud',
      'Wetland','Spatially mixed land/snow-ice','Spatially mixed water/snow-ice','Open ocean','Coastal','River',
      'Lake','Water','Spatially mixed land/water','Inter-tidal area','Dry/Salt lake','Desert, sandy soil',
      'Agriculture','Urban','Mountain','Land','None','Land Snow-ice','Ravine']
    water = ['None of the classes','None','Bright turbid water (blue or brown)','Cocolithophorides',
      'Floating Cyanobacteria bloom','Floating vegetation','Dark water','Wave breaking','Algae','Sediment',
      'Ship trail on water','Water front']
    shadow = ['None of the classes','None','Topographic shadow (incl tree & buildings)','Cloud shadow','Shadow above cloud']
    auxiliary = ['Evergreen Needleleaf Forest','Barren/Desert','Permanent Snow/Ice','Crop/Natural Veg, Mosaic',
      'Urban','Cropland','Permenant Wetland','Grassland','Water Bodies','Savanna','Open Shrubland','Closed Shrubland',
      'Mixed Deciduous Forest','Deciduous Broadleaf Forest','Deciduous Needleleaf Forest','Evergreen Broadleaf Forest',
      'Woody Savanna','Tundra']
    fields = {'SURFACE_ID':surface,'WATER BODY CHARACTERISTICS_ID':water,'SHADOW_ID':shadow,'SURFACE_TYPE_ID':auxiliary}
    lock = dict(source_archive=str(archive.relative_to(REPO)),source_member=member,
       source_definition_sha256=hashlib.sha256(content).hexdigest(),
       fields={field:{str(i):name for i,name in enumerate(names)} for field,names in fields.items()},
       source_type=dict(SURFACE_ID='manual',SHADOW_ID='manual',
                        SURFACE_TYPE_ID='auxiliary',**{'WATER BODY CHARACTERISTICS_ID':'manual where applicable'}),
       explicit_bright_dark={'bright_turbid_water':{'field':'WATER BODY CHARACTERISTICS_ID','code':2},
                             'dark_water':{'field':'WATER BODY CHARACTERISTICS_ID','code':6}},
       other_bright_surface_candidates={'field':'SURFACE_ID','codes':[3,27,28,34],
                    'limitation':'Original snow/salt/desert classes, not measured brightness and not assumed uniformly bright'},
       all_original_codes_retained=True,empty_subtypes_retained=True,student_results_consulted=False,
       overlapping_dimensions_not_additive=True,
       not_collected_fields=['CLOUD_TYPE_ID','CLOUD HEIGHT_ID','OVERSATURATION_ID','SHALLOWNESS_ID'])
    path.write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(original_codes=sum(len(v) for v in fields.values()),definition_sha256=lock['source_definition_sha256'])))


if __name__=='__main__':main()
