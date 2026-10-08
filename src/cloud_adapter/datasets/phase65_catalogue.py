"""Prepared Catalogue fit/development only, for authorized exploratory Phase65a."""
import json
from pathlib import Path
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS, TRANSFORMS


@TRANSFORMS.register_module()
class LoadPhase65RGB(BaseTransform):
    def transform(self, results):
        rgb = np.load(results['img_path'], allow_pickle=False)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
            raise ValueError('Invalid prepared RGB')
        results['img'] = rgb[..., ::-1].copy()  # mmseg preprocessor converts BGR to RGB
        results['img_shape'] = rgb.shape[:2]
        results['ori_shape'] = rgb.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadPhase65Mask(BaseTransform):
    def transform(self, results):
        mask = np.load(results['seg_map_path'], allow_pickle=False)
        if not np.isin(mask, [0, 1, 2, 255]).all():
            raise ValueError('Unexpected parent labels')
        results['gt_seg_map'] = mask
        results.setdefault('seg_fields', []).append('gt_seg_map')
        return results


@DATASETS.register_module()
class Phase65CatalogueDataset(BaseSegDataset):
    METAINFO = dict(classes=('surface_visible', 'cloud', 'shadow'),
                    palette=[[80,130,70], [245,245,245], [70,70,70]])

    def __init__(self, prepared_manifest, cohort, **kwargs):
        if cohort not in {'fit', 'fit_representatives', 'development_val'}:
            raise ValueError('Confirmation and sealed cohorts are forbidden')
        self.prepared_manifest = Path(prepared_manifest)
        self.cohort = cohort
        super().__init__(reduce_zero_label=False, **kwargs)

    def load_data_list(self):
        d = json.loads(self.prepared_manifest.read_text())
        if d['scope'] != 'exploratory_fit_development_only':
            raise ValueError('Unrecognized exploratory manifest')
        rows = [r for r in d['rows'] if r['split'] == ('fit' if self.cohort == 'fit_representatives' else self.cohort)
                and (self.cohort != 'fit_representatives' or r['is_representative'])]
        return [dict(img_path=str(self.prepared_manifest.parent/r['rgb_path']),
                     seg_map_path=str(self.prepared_manifest.parent/r['mask_path']),
                     label_map=None, reduce_zero_label=False, seg_fields=[])
                for r in rows]


@DATASETS.register_module()
class Phase65EvidenceDataset(BaseSegDataset):
    """Explicit new 65b authorization, separate from the original 65a loader."""
    METAINFO = Phase65CatalogueDataset.METAINFO

    def __init__(self, prepared_manifest, cohort, **kwargs):
        if cohort != 'phase65b_evidence':
            raise ValueError('Only newly authorized 65b evidence representatives')
        self.prepared_manifest = Path(prepared_manifest)
        super().__init__(reduce_zero_label=False, **kwargs)

    def load_data_list(self):
        d=json.loads(self.prepared_manifest.read_text())
        if d['scope'] != 'exploratory_phase65b_evidence_only' or len(d['rows']) != 64:
            raise ValueError('Unrecognized newly authorized evidence manifest')
        if any(r['original_split'] != '65b_confirmation' for r in d['rows']):
            raise ValueError('Forbidden cohort')
        return [dict(img_path=str(self.prepared_manifest.parent/r['rgb_path']),
                     seg_map_path=str(self.prepared_manifest.parent/r['mask_path']),
                     label_map=None,reduce_zero_label=False,seg_fields=[]) for r in d['rows']]
