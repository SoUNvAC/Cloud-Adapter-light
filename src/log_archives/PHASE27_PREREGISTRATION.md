# Phase 27 preregistration: Landsat 8 cross-sensor generalization

Phase 27 measures zero-shot transfer from Sentinel-2 CloudSEN12 to the official
USGS Landsat 8 Collection 2 cloud-truth validation set. It starts only after one
Phase 25 candidate passes. No Landsat image or label may be used for checkpoint,
threshold, preprocessing, or tiling selection.

Official source: <https://www.usgs.gov/data/landsat-8-collection-2-cloud-truth-mask-validation-set>

The release contains 48 randomly selected Landsat 8 Collection 2 scenes with
manual GeoTIFF truth. Truth values are 0 fill, 128 clear, 192 thin cloud, and
255 opaque cloud. The data release is CC0.

## Manifest and audit

Place the release below `data/landsat8_c2_cca` or set
`LANDSAT8_C2_CCA_ROOT`. Create `manifest.json` with this structure:

```json
{
  "dataset": "USGS Landsat 8 Collection 2 cloud truth mask validation set",
  "source": "https://doi.org/10.5066/P9FI4A0Y",
  "sciencebase_item_id": "61015b2fd34ef8d7055d6395",
  "license": "CC0-1.0",
  "scenes": [
    {
      "scene_id": "LC08_...",
      "b2": "relative/path/to/B2.TIF",
      "b3": "relative/path/to/B3.TIF",
      "b4": "relative/path/to/B4.TIF",
      "truth": "relative/path/to/manual_truth.TIF",
      "qa_pixel": "relative/path/to/QA_PIXEL.TIF",
      "mtl": "relative/path/to/MTL.txt"
    }
  ]
}
```

The audit requires exactly 48 unique scenes, pixel-identical raster geometry,
only the four documented truth values, all three non-fill truth classes, all
required MTL reflectance fields, and SHA-256 for every referenced file.

## Frozen inference protocol

- Select the median-validation-mIoU Phase 25 seed. CloudSEN12 test remains
  sealed.
- Convert Landsat B4/B3/B2 DN to sun-angle-corrected Level-1 TOA reflectance
  using each scene's MTL coefficients; clip to [0,1] and scale to [0,255]. No
  percentile stretch or Landsat-dependent color tuning is allowed.
- Use 512 x 512 inference tiles with a 384 x 384 scored core and 64-pixel halo.
  Outside-scene context is zero. Every valid truth pixel is scored exactly once.
- Binary mapping: thick and thin cloud are cloud; clear and cloud shadow are
  non-cloud. Also report harmonized clear/thin/opaque metrics and the predicted
  shadow rate.
- Compare against the product's CFMask QA_PIXEL baseline using fixed bits 1
  (dilated cloud), 2 (cirrus), and 3 (cloud).
- Treat each of the 48 scenes as an independent statistical unit. Report pooled
  metrics, scene-macro metrics, and paired 10,000-replicate bootstrap 95% CIs
  with seed 20260918.

Run:

```bash
python -m pip install -r requirements-external.txt
bash tools/run_phase27_landsat_c2_4090d.sh
```

## Hard gates and stop-loss

All gates must pass:

1. Pooled cloud IoU >= 50.0 and cloud F1 >= 65.0.
2. Harmonized thin-cloud recall >= 40.0.
3. Scene-macro cloud-IoU bootstrap lower 95% bound >= 40.0.
4. Mean paired scene-IoU difference from CFMask >= -10.0 percentage points.
5. All 48 scenes are evaluated exactly once and no non-finite logits occur.

If any gate fails, zero-shot cross-sensor transfer is closed. Phase 28 switches
to a sensor-input adapter trained only on the disjoint 38-Cloud training scenes;
the 48 USGS scenes and all Phase 27 thresholds remain unchanged. Phase 27 is
appended to `EXPERIMENT_LOG.md` only after the measured run completes.
