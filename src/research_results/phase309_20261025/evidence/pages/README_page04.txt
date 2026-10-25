L0001: subscenes
L0002: 
L0003: 
L0004: Each subscene is a 1022-by-1022-by-13 numpy array, with Top of Atmosphere reflectance values held as float32 numbers. These are
L0005: taken directly from the Sentinel-2 L1C product that they share their name with, cropped randomly (but resampled until an area without
L0006: any no-data values is found). Bands that are not at 20m are resampled to 20m using bilinear interpolation. Users who wish to use the
L0007: bands at their original resolutions will need to download the L1C products and use the shapefiles provided to extract the area of the
L0008: mask.
L0009: 
L0010: The order of Sentinel-2 bands in the numpy array's 3rd dimension is in numerical order, with band 8A coming between bands 8 and 9. All
L0011: values are found by dividing the original L1C integer values by 10'000, to retrieve the reflectances, as advised in the Product
L0012: Specification for Sentinel-2 L1C. Please note that many values go above 1, because apparent reflectance can be greater than 1 if a
L0013: surface is at an angle which receives more light than is possible on a surface normal to the viewing angle.
L0014: 
L0015: 
L0016: 
L0017: masks
L0018: 
L0019: 
L0020: Each mask are a 1022-by-1022-by-3 numpy array, with boolean one-hot encoding (each pixel has exactly one True value across the last
L0021: dimension). The class order in the last dimension is: CLEAR, CLOUD, CLOUD_SHADOW. Even if shadow markings were not possible
L0022: for a given subscene, the third channel is still included. The masks are named using the corresponding product IDs of the scenes.
L0023: 
L0024: 
L0025: 
L0026: shapefiles
L0027: 
L0028: 
L0029: An Esri shapefile for each scene which describes the polygon of the extracted subscene. These can be used to extract the original band
L0030: data from the Sentinel-2 L1C scenes if users need them.
L0031: 
L0032: 
L0033: 
L0034: thumbnails
L0035: 
L0036: 
L0037: A set of downsampled .png images showing the subscenes, plus a small area around them. Not to be used for any processing, they are
L0038: included to give a simple way to browse the data.
L0039: 
L0040: 
L0041: 
L0042: alt_masks
L0043: 
L0044: 
L0045: For scenes which were annotated by both annotators (10 in CALIBRATION and 50 in VALIDATION—see Appendix B for more details
L0046: about this process) the other annotators mask is provided for completeness. This can be used to verify the statistical tests we performed
L0047: using them, or as backup masks should you spot one that you would prefer to use for any reason. The selection was random, so they
L0048: are just as valid as those found in the mask folder.
