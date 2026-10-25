L0001: Sentinel-2 Cloud Mask Catalogue
L0002: 
L0003: 
L0004: 
L0005: 
L0006: 
L0007: 
L0008: 
L0009: 
L0010: 
L0011: 
L0012: 
L0013: 
L0014: 
L0015: 
L0016: 
L0017: 
L0018: 
L0019: 
L0020: 
L0021: 
L0022: 1. General information
L0023: 
L0024: 
L0025: This dataset comprises cloud masks for 513 1022-by-1022 pixel subscenes, at 20m resolution, sampled random from the 2018 Level-1C
L0026: Sentinel-2 archive. The design of this dataset follows from some observations about cloud masking: (i) performance over an entire
L0027: product is highly correlated, thus subscenes provide more value per-pixel than full scenes, (ii) current cloud masking datasets often
L0028: focus on specific regions, or hand-select the products used, which introduces a bias into the dataset that is not representative of the real-
L0029: world data, (iii) cloud mask performance appears to be highly correlated to surface type and cloud structure, so testing should include
L0030: analysis of failure modes in relation to these variables.
L0031: 
L0032: The data was annotated semi-automatically, using the IRIS toolkit, which allows users to dynamically train a Random Forest
L0033: (implemented using LightGBM), speeding up annotations by iteratively improving it's predictions, but preserving the annotator's ability to
L0034: make final manual changes when needed. This hybrid approach allowed us to process many more masks than would have been
L0035: possible manually, which we felt was vital in creating a large enough dataset to approximate the statistics of the whole Sentinel-2
L0036: archive.
L0037: 
L0038: 
L0039: In addition to the pixel-wise, 3 class (CLEAR, CLOUD, CLOUD_SHADOW) segmentation masks, we also provide users with binary
L0040: classification "tags" for each subscene that can be used in testing to determine performance in specific circumstances. These include:
L0041: 
L0042:      SURFACE TYPE: 11 categories
L0043:      CLOUD TYPE: 7 categories
L0044:      RELATIVE CLOUD HEIGHT: low, high
L0045:      CLOUD THICKNESS: thin, thick
L0046:      CLOUD EXTENT: isolated, extended
L0047: 
L0048: Wherever practical, cloud shadows were also annotated, however this was sometimes not possible due to high-relief terrain, or large
L0049: ambiguities. In total, 424 were marked with shadows (if present), and 89 have shadows that were not annotatable due to very
L0050: ambiguous shadow boundaries, or terrain that cast significant shadows. If users wish to train an algorithm specifically for cloud shadow
L0051: masks, we advise them to remove those 89 images for which shadow was not possible, however, bear in mind that this will
L0052: systematically reduce the difficulty of the shadow class compared to real-world use, as these contain the most difficult shadow examples.
L0053: 
L0054: In addition to the 20m sampled subscenes and masks, we also provide users with shapefiles that define the boundary of the mask on
L0055: the original Sentinel-2 scene. If users wish to retrieve the L1C bands at their original resolutions, they can use these to do so.
