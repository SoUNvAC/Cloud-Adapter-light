L0001: APPENDIX B: Annotation Strategy
L0002: 
L0003: 
L0004: To annotate the images, we first randomly sampled images from the Sentinel-2 2018 archive. By "random", we mean that every product
L0005: in the 2018 archive had an equal chance of being selected (rather than randomising the location or some other variable). For each
L0006: selected product, we then extracted random areas of 1152-by-1152 pixels at 20m (note the difference between this and the final 1022-
L0007: by-1022 subscenes, explained in the next paragraph). This was completed using a virtual machine using the CREODIAS system, which
L0008: allowed us to quickly query and access the Sentinel-2 data without needing to download it.
L0009: 
L0010: Once our subscenes were extracted, we began annotating some test images, and configured IRIS. Several different views were defined,
L0011: using different band combinations, which could be quickly switched between. It also provided a BingMaps view of the location, which
L0012: was invaluable when subscenes were highly ambiguous, and gave context for where we were actually looking. During annotation, we
L0013: padded the subscenes by 64 pixels on each side (resulting in the 1152-by-1152 image used in IRIS), so that there was no disadvantage
L0014: to annotations close to the edge of the final 1022-by-1022 window. We had originally intended to export 1024-by-1024 pixel masks,
L0015: however we noticed too late a bug which caused all edge pixels of the mask to be unfilled, hence the final 1022-by-1022 window size.
L0016: 
L0017: The Random Forest (RF) could also be configured dynamically. As a general rule, we began annotating a subscene with a simple, small
L0018: RF that just used the reflectance values. If the model was seen to be struggling, then edge filters could be added as input, and the RF
L0019: size increased in terms of number of trees, branches and leaves. A suppression filter could also be activated, which smoothed the mask
L0020: by removing cloudy/shadowy pixels which were not in the vicinity of others. We always checked carefully over the image to check that
L0021: the RF had done a reasonable job, and often found that significant numbers of iterations and manual corrections were required.
L0022: 
L0023: We annotated with high recall in mind. If we felt totally split between annotating an area as cloudy or clear, we defaulted to marking as
L0024: cloudy. There is no perfect solution to this, as cloud masking is an inherently ambiguous task, but we felt most users of cloud masks
L0025: desired high recall, and thus our cloud masks should be cautious.
L0026: 
L0027: Our annotation style was formalised using 10 hand-picked images (which contained difficult or instructive examples of the kinds of
L0028: decisions that would need to be made). These 10 images constitute the CALIBRATION set, on which we refined our annotations in an
L0029: attempt to get as high an agreement between us as possible, within a session of a few hours.
L0030: 
L0031: Once CALIBRATION was completed, we began annotating subscenes individually, served to us randomly by IRIS. This comprised the
L0032: scenes with the MAIN dataset tag, resulting in 453 annotated subscenes. After this, we then decided to both annotate another 50
L0033: subscenes in parallel (without consulting one another like we had in CALIBRATION). This VALIDATION set can then be used to measure
L0034: the level of inter-annotator agreement, and thus gives users a rough guide of human-level performance.
L0035: 
L0036: Once all masks were complete, we then went through every image again and added the classification tags. These were entered into a
L0037: spreadsheet using a Google Form, in which the product ID was entered, along with checkboxes for each of the tags. Some logical
L0038: checks were then carried out to reduce human errors in this process (e.g. no image can contain a CLOUD TYPE, but not a CLOUD
L0039: THICKNESS, RELATIVE CLOUD HEIGHT and CLOUD EXTENT tag). Undoubtedly, there will remain some errors in the classification
L0040: tags, and we encourage users to double-check them if they are especially important in their application.
L0041: 
L0042: 
L0043: 
L0044: 
L0045: Contributions & Acknowledgements
L0046: 
L0047: 
L0048: The data were collected, annotated, checked, formatted and published by Alistair Francis and John Mrziglod.
L0049: 
L0050: Support and advice was provided by Prof. Jan-Peter Muller and Dr. Panagiotis Sidiropoulos, for which we are grateful.
L0051: 
L0052: We would like to extend our thanks to Dr. Pierre-Philippe Mathieu and the rest of the team at ESA PhiLab, who provided the environment
L0053: in which this project was conceived, and continued to give technical support throughout.
L0054: 
L0055: Finally, we thank the ESA Network of Resources for sponsoring this project by providing ICT resources.
