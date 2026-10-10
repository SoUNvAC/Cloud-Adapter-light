import hashlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import phase75_metadata as m

PRODUCT='S2B_MSIL1C_20180205T073059_N0206_R049_T37LDL_20180205T111142'
PREFIX='tiles/37/L/DL/'+PRODUCT+'.SAFE/'
GRANULE='L1C_T37LDL_A004794_20180205T073650'


def fixture():
    refs='';info='';objects=[]
    for band in m.BANDS:
        f='GRANULE/'+GRANULE+'/IMG_DATA/T37LDL_20180205T073650_'+band
        refs+='<IMAGE_FILE>'+f+'</IMAGE_FILE>'
        info+='<Spectral_Information physicalBand="B'+str(int(band[1:]))+'"><RESOLUTION>'+('20' if band in ['B11','B12'] else '10')+'</RESOLUTION></Spectral_Information>'
        objects.append(dict(name=PREFIX+f+'.jp2'))
    product=('<root><PRODUCT_URI>'+PRODUCT+'.SAFE</PRODUCT_URI><PROCESSING_BASELINE>02.06</PROCESSING_BASELINE>'
        '<DATATAKE_SENSING_START>2018-02-05T07:30:59.024Z</DATATAKE_SENSING_START><Granule granuleIdentifier="'+GRANULE+'">'+refs+'</Granule>'+info+
        '<QUANTIFICATION_VALUE>10000</QUANTIFICATION_VALUE><Special_Values><SPECIAL_VALUE_TEXT>NODATA</SPECIAL_VALUE_TEXT><SPECIAL_VALUE_INDEX>0</SPECIAL_VALUE_INDEX></Special_Values></root>').encode()
    grid=''
    for res,size in [(10,10980),(20,5490),(60,1830)]:
        grid+=f'<Size resolution="{res}"><NROWS>{size}</NROWS><NCOLS>{size}</NCOLS></Size><Geoposition resolution="{res}"><ULX>399960</ULX><ULY>9100020</ULY><XDIM>{res}</XDIM><YDIM>{-res}</YDIM></Geoposition>'
    tile=('<root><TILE_ID>S2B_OPER_MSI_L1C_TL_SGS__20180205T111142_A004794_T37LDL_N02.06</TILE_ID><SENSING_TIME>2018-02-05T07:36:50.888Z</SENSING_TIME>'
        '<HORIZONTAL_CS_CODE>EPSG:32737</HORIZONTAL_CS_CODE><Mean_Sun_Angle><ZENITH_ANGLE unit="deg">30</ZENITH_ANGLE><AZIMUTH_ANGLE unit="deg">110</AZIMUTH_ANGLE></Mean_Sun_Angle>'
        '<Sun_Angles_Grid><Zenith><ROW_STEP unit="m">5000</ROW_STEP><COL_STEP unit="m">5000</COL_STEP><Values_List><VALUES>30 31</VALUES><VALUES>32 33</VALUES></Values_List></Zenith></Sun_Angles_Grid>'+grid+'</root>').encode()
    return product,tile,objects


class Tests(unittest.TestCase):
    def test_fixed_selection(self):
        rows=[dict(split='fit',representative=str(i),group_id=str(i)) for i in range(153)]
        tags={str(i):dict(shadow_percent='1' if i<5 else '0') for i in range(153)}
        a=m.select(dict(proposed_rows=rows),tags)
        expected=sorted(map(str,range(5)),key=lambda x:hashlib.sha256(('phase75:'+x).encode()).hexdigest())[:3]
        self.assertEqual([r['product'] for r in a['selected']],expected)
        tags['0']['shadow_percent']='0';tags['1']['shadow_percent']='0';tags['2']['shadow_percent']='0'
        self.assertEqual(len(m.select(dict(proposed_rows=rows),tags)['selected']),2)

    def test_xml_identity_grids_and_angles(self):
        a,b,objects=fixture();r=m.parse_xml(a,b,PRODUCT,PREFIX,objects)
        self.assertTrue(r['solar_metadata_verified']);self.assertTrue(r['source_grid_verified'])
        self.assertEqual(r['source_grid']['bands']['B02']['native_resolution_m'],'10')
        self.assertEqual(r['source_grid']['bands']['B12']['native_resolution_m'],'20')
        self.assertTrue(r['solar']['grid_present']);self.assertEqual(r['radiometry']['special_values']['NODATA'],0)
        self.assertEqual(r['radiometry']['reflectance_offsets'],[])
        with self.assertRaises(ValueError):m.parse_xml(a.replace(b'N0206',b'N0500'),b,PRODUCT,PREFIX,objects)

    def test_missing_angle_not_guessed(self):
        a,b,o=fixture();b=b.replace(b'<ZENITH_ANGLE unit="deg">30</ZENITH_ANGLE>',b'')
        self.assertFalse(m.parse_xml(a,b,PRODUCT,PREFIX,o)['solar_metadata_verified'])
        self.assertFalse(m.parse_xml(a,b,PRODUCT,PREFIX,o[:-1])['source_grid_verified'])

    def test_http_budget_and_no_rasters(self):
        class Response(io.BytesIO):
            status=200;headers={'Content-Length':'8'};url='https://storage.googleapis.com/test'
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as tmp:
            root=Path(tmp);client=m.BudgetClient(root)
            with patch.object(m,'LIMIT',5),patch.object(m.urllib.request,'urlopen',return_value=Response(b'12345678')):
                with self.assertRaises(ValueError):client.get('https://storage.googleapis.com/test',root/'x.json')
                self.assertEqual(client.used,0)
            with self.assertRaises(ValueError):client.get('https://storage.googleapis.com/x.jp2',root/'x.jp2')
            with self.assertRaises(ValueError):client.get('https://example.com/test',root/'x.xml')

if __name__=='__main__':unittest.main()
