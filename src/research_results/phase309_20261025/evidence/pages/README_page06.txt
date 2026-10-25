L0001: 4. Errors and Uncertainties
L0002: 
L0003: 
L0004: Unfortunately, cloud masking is an inherently ambiguous task. When framed as a binary segmentation task, an annotator must decide
L0005: how to distinguish between cloud and clear, and differences between their definition and another annotator's is inevitable. We have
L0006: sought, using our CALIBRATION process, to make sure that both annotators were as similar as possible, and that our decisions were as
L0007: consistent as possible. However, errors still exist throughout the dataset. Some of these can be estimated, whilst others are known but
L0008: cannot necessarily be accounted for. In this section, we first outline some of the known issues that are not easily quantified, then later,
L0009: we use some statistical metrics to quantify the level of inter-annotator agreement.
L0010: 
L0011: 
L0012: Known issues
L0013: 
L0014: Mask boundaries
L0015: 
L0016: As discussed in Appendix B, we discovered a small bug that affected the border pixels of each mask, making them all clear. To resolve
L0017: this issue, we have cropped all masks and subscenes to be 1022 pixels across in both dimensions.
L0018: 
L0019: 
L0020: 
L0021: 
L0022: Cloud pixels surrounding shadows
L0023: 
L0024: We noticed in many scenes that, at the boundary between cloud shadow and clear pixels, a thin line of cloud pixels could sometimes
L0025: erroneously appear in the Random Forest prediction. We corrected these errors manually when they were spotted during annotation,
L0026: however, because it affected very small areas, some were missed. It is likely that this occurred because the edge of cloud shadows
L0027: resembled the darker, shaded areas of clouds to the Random Forest. We considered attempting to use a post-processing technique to
L0028: remove these artifacts, however we determined that it would likely cause more errors than it would solve. Overall, these are most often a
L0029: few isolated pixels at the edges of some shadows, and so should not effect an algorithm's training (or test accuracy) too dramatically.
L0030: 
L0031: 
L0032: 
L0033: Large, thin cloud boundaries
L0034: 
L0035: If thin clouds (often cirrus) appeared over a large section of the image, it was often incredibly difficult to tell exactly where the boundary
L0036: between clear and cloud lay. Ultimately this is a subjective exercise, and we approached it with the understanding that, if in doubt, mark
L0037: it as cloud. This is because most end-users of cloud masks desire high cloud recall, though we appreciate it may not suit users who
L0038: prefer high precision.
L0039: 
L0040: 
L0041: 
L0042: High-reflectance icy subscenes
L0043: 
L0044: Some subscenes, often over the Antarctic, were essentially completely white in all bands, except the SWIR bands, indicating that the
L0045: entire subscene was dominated by ice or snow. It was often difficult to tell if this was either clear, high-altitude ice shelves, or smooth icy
L0046: clouds. We therefore usually marked these with high difficulty (4 or 5), as we were never completely certain that we hadn't marked the
L0047: entire scene incorrectly.
L0048: 
L0049: 
L0050: 
L0051: Classification tag ambiguities
L0052: 
L0053: The classification tag scheme was developed to be a quick-and-easy way to understand how an algorithm performs in a given context.
L0054: They are based on visual cues (described in Section 2), rather than physical measurements. As a result, the categories used will not be
L0055: suitable to every user, and some human errors will more than likely still exist. In particular, CLOUD TYPE should be seen as much more
L0056: of a subjective classification, that delineates clouds by certain visual characteristics, but without any formal meteorological rigour. We
L0057: recommend that if users have specific needs of their own they should develop their own classification scheme to compliment the one
L0058: provided here.
