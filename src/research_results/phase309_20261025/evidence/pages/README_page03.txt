L0001:      clear_percent: float, 0 -> 100 for percentage of pixels marked CLEAR
L0002:      cloud_percent: float, 0 -> 100 for percentage of pixels marked CLOUD
L0003:      shadow_percent: float, 0 -> 100 for percentage of pixels marked CLOUD_SHADOW
L0004:      dataset: string, referring to whether subscene was part of CALIBRATION, MAIN, or VALIDATION labelling stages (described in
L0005:      Section "Annotation Strategy")
L0006: 
L0007: 
L0008: 
L0009: SURFACE TYPES:
L0010: 
L0011:      forest/jungle: boolean, dense tree cover
L0012:      snow/ice: boolean, any snow or ice covering either land or water
L0013:      agricultural: boolean, farmlands or pastures
L0014:      urban/developed: boolean, human structures, conurbations, villages, industrial sites. Not including solitary roads.
L0015:      coastal: boolean, coastlines, specifically. (Not just water near to land, but the division between land and water itself)
L0016:      hills/mountains: boolean, noticeable hilly terrain
L0017:      desert/barren: boolean, completely arid, or very sparsely vegetated, drylands.
L0018:      shrublands/plains: boolean, somewhat sparsely vegetated areas, or non-forested areas without obvious signs or agriculture.
L0019:      wetland/bog/marsh: boolean, areas with large amounts of standing water (although )
L0020:      open_water: boolean, extended bodies of water, such as large lakes that continue beyond the image bounds, or the sea.
L0021:      enclosed_water: boolean, lakes, large rivers or contained areas of sea-water (e.g. fjords)
L0022: 
L0023: 
L0024: 
L0025: CLOUD THICKNESS:
L0026: 
L0027:      thin: boolean, any cloud where some surface colour/signal can be seen through it (not including at the edges of thick clouds, to
L0028:      avoid all thick tags being accompanied by thin ones)
L0029:      thick: boolean, any cloud where no surface can be seen through it (except possibly at its edges).
L0030: 
L0031: 
L0032: 
L0033: RELATIVE CLOUD HEIGHT:
L0034: 
L0035:      low: boolean, any cloud which has no obvious sign of high-altitude (high reflectance in the Cirrus (B10) band)
L0036:      high: boolean, any cloud with noticable reflectance in the Cirrus (B10) band
L0037: 
L0038: 
L0039: 
L0040: CLOUD EXTENT:
L0041: 
L0042: 
L0043:      isolated: boolean, clouds which are separated from one another, such that there are no areas of continuous cloud across the
L0044:      majority of the subscene.
L0045:      extended: boolean, clouds which span continuously across a majority of a subscene.
L0046: 
L0047: 
L0048: 
L0049: CLOUD TYPE:
L0050: 
L0051:      cumulus: boolean, rounded, clumpy and separate clouds.
L0052:      cumulonimbus: boolean, clouds with very large, steep, vertical profiles, reaching up very high.
L0053:      altocumulus/stratocumulus: boolean, most general cloud type, denoting any clouds that form periodic cells. Could be high or low,
L0054:      and thick or thin.
L0055:      cirrus: boolean, thin, extended clouds that have high reflectance in the Cirrus (B10) band. Sometimes almost invisible or very faint
L0056:      in any other band.
L0057:      haze/fog: boolean, low-lying, textureless, thin cloud. Often distinguishable because it looks similar to cirrus, but with no reflectance
L0058:      in B10.
L0059:      ice_clouds: boolean, any cloud that has a noticeable icy signature, seen as reddish-orange in the false-colour B1 (red), B11
L0060:      (green), B12 (blue)
L0061:      contrails: boolean, exhaust trails left by planes.
