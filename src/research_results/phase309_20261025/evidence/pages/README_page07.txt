L0001: Inter-annotator agreement
L0002: 
L0003: 
L0004: Now we consider the consistency of annotations between the two annotators. This does not give us a measure of actual accuracy, but
L0005: rather relative consistency between annotations. We present statistics here for both CALIBRATION (10 subscenes) and VALIDATION
L0006: (50 subscenes). See Appendix B for details on how these two stages were carried out.
L0007: 
L0008: 
L0009: F1-score here refers to the harmonic mean of recall and precision, and is therefore symmetrical with respect to the annotators (the recall
L0010: of one annotator is the precision of the other, and vice versa). Accuracy simply refers to the percentage of pixels that are in agreement
L0011: between the two annotators.
L0012: 
L0013: CALIBRATION DATASET:
L0014: 
L0015: 
L0016: 
L0017:     CLEAR F1-SCORE:             94.478%   |------------------------------------->  │
L0018:     CLOUD F1-SCORE:             93.149%   |------------------------------------->  │
L0019:     CLOUD_SHADOW F1-SCORE:      85.024%   |---------------------------------->     │
L0020:     TOTAL PIXEL-WISE ACCURACY:  93.739%   |------------------------------------->  │
L0021:                                           |────────────────────────────────────────|
L0022:                                           0                   50                  100
L0023: 
L0024: 
L0025: 
L0026: VALIDATION DATASET:
L0027: 
L0028: 
L0029: 
L0030:     CLEAR F1-SCORE:             94.574%   |------------------------------------->  │
L0031:     CLOUD F1-SCORE:             95.969%   |--------------------------------------> │
L0032:     CLOUD_SHADOW F1-SCORE:      77.947%   |------------------------------->        │
L0033:     TOTAL PIXEL-WISE ACCURACY:  94.978%   |------------------------------------->  │
L0034:                                           |────────────────────────────────────────|
L0035:                                           0                   50                  100
L0036: 
L0037: 
L0038: 
L0039: We were surprised to find a higher overall agreement for VALIDATION, given that we worked on the masks separately, without
L0040: consultation. However, this may be because the CALIBRATION images were hand picked because of their usefulness in demonstrating
L0041: certain features that may be difficult to annotate consistently. We encourage users to split the dataset such that the test set includes all
L0042: CALIBRATION and VALIDATION images, so that an approximate comparison can be made between model performance and inter-
L0043: annotator agreement.
