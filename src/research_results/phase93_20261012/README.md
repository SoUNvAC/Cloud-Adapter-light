# Phase93: nonlinear raw spectral baselines

## 1. Data results

| Domain | Model | MacroAP % | Recall % | FPR % | FPR upper95 % | PooledAP % |
|---|---|---:|---:|---:|---:|---:|
| development | S3 | 68.9410 | 8.9890 | 2.0881 | 5.8205 | 12.0313 |
| development | R6 | 70.4643 | 10.2811 | 2.8750 | 7.7288 | 12.8017 |
| development | Source_spectral | 75.0050 | 31.2227 | 1.0880 | 2.1055 | 21.6030 |
| development | L3 | 73.6307 | 64.9087 | 0.8645 | 1.5068 | 67.1820 |
| development | G6 | 55.9056 | 47.9959 | 4.6059 | 10.6688 | 15.8712 |
| development | G7 | 75.0734 | 81.1038 | 6.7812 | 12.7994 | 55.0077 |

development G7-G6: 19.1678 pp, CI95 [8.023953013279492, 31.48238805105164], improved/stable/declined 6/1/0.
| original_main | S3 | 39.6539 | 0.2061 | 1.6675 | 4.1016 | 20.8547 |
| original_main | R6 | 38.6769 | 3.9307 | 1.7121 | 3.6221 | 19.0693 |
| original_main | Source_spectral | 63.8801 | 39.3288 | 0.5358 | 1.0396 | 52.3161 |
| original_main | L3 | 63.6966 | 58.8045 | 1.4586 | 2.1610 | 58.1249 |
| original_main | G6 | 46.8496 | 66.0604 | 7.5249 | 11.1348 | 37.7474 |
| original_main | G7 | 66.6765 | 87.1557 | 4.1617 | 5.5358 | 63.5951 |

original_main G7-G6: 19.8269 pp, CI95 [15.242829864393666, 24.725393964311202], improved/stable/declined 9/0/0.
| original_common | S3 | 42.9534 | 0.3004 | 2.1369 | 5.9388 | 16.3337 |
| original_common | R6 | 42.3325 | 4.7550 | 1.9492 | 5.0681 | 14.2261 |
| original_common | Source_spectral | 68.9935 | 41.6883 | 0.2525 | 0.6100 | 54.9902 |
| original_common | L3 | 68.3316 | 60.4031 | 0.7788 | 1.0360 | 59.9063 |
| original_common | G6 | 53.6989 | 68.9142 | 7.4669 | 12.6006 | 33.6303 |
| original_common | G7 | 73.8241 | 89.2378 | 3.1801 | 4.7525 | 65.8410 |

original_common G7-G6: 20.1253 pp, CI95 [15.372183936163859, 24.18816042085729], improved/stable/declined 6/0/0.
| revised_common | S3 | 37.8283 | 0.3047 | 2.1418 | 5.9535 | 15.3488 |
| revised_common | R6 | 36.2814 | 4.5651 | 1.9531 | 5.0797 | 13.0893 |
| revised_common | Source_spectral | 65.7835 | 33.0121 | 0.2632 | 0.6229 | 51.9815 |
| revised_common | L3 | 65.5268 | 53.1680 | 0.7345 | 1.0591 | 56.9213 |
| revised_common | G6 | 50.1845 | 58.9040 | 7.5047 | 12.6014 | 30.6062 |
| revised_common | G7 | 71.5146 | 81.7797 | 3.1146 | 4.6824 | 62.3982 |

revised_common G7-G6: 21.3300 pp, CI95 [16.09208903542286, 25.832699662061085], improved/stable/declined 6/0/0.

## 2. Interpretation boundaries

Exactly two fixed 200-round HGB fits; no segmentation training or network forward. No parameter selection or deployment winner.
All intervals are descriptive related-group bootstrap (10000, seed93012); exposed ALCD pools are not independent confirmation.
Compare G6 with R6 to assess linear baseline weakness; G7-G6 measures practical Source score increment under this fixed learner only.
G6/G7 thresholds are probabilities, used directly. Fit calibration is in-sample, not an independent FPR guarantee.
Oracle thresholds use evaluation references and must not be deployed. No fabricated three-class mIoU.
Phase89:54 reused probabilities lack pre-crash SHA. Human B was not performed; original empty fields and failed gates remain.
