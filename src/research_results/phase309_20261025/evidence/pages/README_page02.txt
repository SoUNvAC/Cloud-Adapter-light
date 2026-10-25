L0001: 2. Dataset Description
L0002: 
L0003: 
L0004: We will now describe the structure of the overall dataset, and then go into more detail for each specific component.
L0005: 
L0006: 
L0007: 
L0008:    DATASET/
L0009:    │   README.pdf
L0010:    │   classification_tags.csv
L0011:    │
L0012:    └───subscenes/
L0013:    │   │   <scene1>.npy
L0014:    │   │   <scene2>.npy     ---->  1022x1022x13, float32, TOA reflectance
L0015:    │   │   <scene3>.npy
L0016:    │   │   ....
L0017:    └───masks/
L0018:    │   │   <scene1>.npy
L0019:    │   │   <scene2>.npy     ---->  1022x1022x3, bool, one-hot encoded classes
L0020:    │   │   <scene3>.npy
L0021:    │   │   ...
L0022:    └───shapefiles/
L0023:    │   └───<scene1>/
L0024:    │   │   │   <scene1>.cpg  \
L0025:    │   │   │   <scene1>.dbf   \
L0026:    │   │   │   <scene1>.prj    }-> Esri shapefile for mask/subscene outline
L0027:    │   │   │   <scene1>.shp   /
L0028:    │   │   │   <scene1>.shx  /
L0029:    │   └───<scene2>/
L0030:    │   │   │   ...
L0031:    │   │   ...
L0032:    └───thumbnails/
L0033:    │   │   <scene1>.png
L0034:    │   │   <scene2>.png     ----> 512x512x3, uint8, downsampled image for quickly
L0035:    │   │   <scene3>.png                             inspecting subscene
L0036:    │   │   ...
L0037:    └───alt_masks/
L0038:    │   │   ALT_<scene1>.npy
L0039:    │   │   ALT_<scene2>.npy
L0040:    │   │   ALT_<scene3>.npy
L0041:    │   │   ....
L0042: 
L0043: 
L0044: 
L0045: 
L0046: classification_tags.csv
L0047: 
L0048: 
L0049: The classification_tags.csv file contains several parameters that should be helpful for users. All the categorical tags are non-exclusive,
L0050: and simply mean a given subscene contained some of that characteristic. For example, snow-covered mountain tops with forests in the
L0051: lower valleys, would be tagged with forest/jungle, snow/ice and hills/mountains. Similarly multiple cloud types can exist within the same
L0052: image. All tags were determined by visual inspection of both the image and, for SURFACE TYPE, other high-resolution imagery of the
L0053: area (e.g. BingMaps and Google Earth). Objective classifications of this kind are not possible, and all our classifications here are made
L0054: using subjective judgement. We describe each column (grouped into broad categories) as follows:
L0055: 
L0056: 
L0057: 
L0058: GENERAL INFORMATION:
L0059: 
L0060:      scene: string, Sentinel-2 product ID.
L0061:      difficulty: int, Subjective measure from 1->5 for difficulty of annotation.
L0062:           1. Near perfect
L0063:           2. Very good
L0064:           3. Mostly good
L0065:           4. Possibly some small errors
L0066:           5. Possibly large errors
L0067:      annotator: string, Which of the two annotators, A or B, created the mask used in masks/ (note that alt_masks/ may contain the
L0068:      other annotator's mask, if both labelled that subscene).
L0069:      shadows_marked: boolean, if 0 then no shadow markings were possible. If 1, shadows were marked where present.
