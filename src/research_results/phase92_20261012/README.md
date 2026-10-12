# Phase92 data results and interpretation

## 1. Numerical results

Exploratory only. All intervals resample related groups equally (10000, seed92012); domains are separate.
S3/R6 are binary Shadow baselines, not three-class systems. Thresholds frozen on all153 fit representatives.

| Domain | Strategy | Macro AP % | Macro recall % | Macro FPR % | FPR upper95 % | Pooled AP % |
|---|---|---:|---:|---:|---:|---:|
| development | S3 | 68.9410 | 8.9890 | 2.0881 | 5.8205 | 12.0313 |
| development | R6 | 70.4643 | 10.2811 | 2.8750 | 7.7289 | 12.8017 |
| development | Source_spectral | 75.0050 | 31.2227 | 1.0880 | 2.0922 | 21.6030 |
| development | L3 | 73.6307 | 64.9087 | 0.8645 | 1.5171 | 67.1820 |
| original_main | S3 | 39.6539 | 0.2061 | 1.6675 | 4.0402 | 20.8547 |
| original_main | R6 | 38.6769 | 3.9307 | 1.7121 | 3.5437 | 19.0693 |
| original_main | Source_spectral | 63.8801 | 39.3288 | 0.5358 | 1.0618 | 52.3161 |
| original_main | L3 | 63.6966 | 58.8045 | 1.4586 | 2.1773 | 58.1249 |
| original_common | S3 | 42.9534 | 0.3004 | 2.1369 | 5.9388 | 16.3337 |
| original_common | R6 | 42.3325 | 4.7550 | 1.9492 | 5.0681 | 14.2261 |
| original_common | Source_spectral | 68.9935 | 41.6883 | 0.2525 | 0.6089 | 54.9902 |
| original_common | L3 | 68.3316 | 60.4031 | 0.7788 | 1.0360 | 59.9063 |
| revised_common | S3 | 37.8283 | 0.3047 | 2.1418 | 5.9535 | 15.3488 |
| revised_common | R6 | 36.2814 | 4.5651 | 1.9531 | 5.0797 | 13.0893 |
| revised_common | Source_spectral | 65.7835 | 33.0121 | 0.2632 | 0.6228 | 51.9815 |
| revised_common | L3 | 65.5268 | 53.1680 | 0.7345 | 1.0591 | 56.9213 |

Primary Source_spectral minus S3 macro AP (percentage points):
- development: 6.0640, CI95 [-9.912366485664105, 19.469263158656393]; improved/stable/declined 5/0/2.
- original_main: 24.2262, CI95 [15.376994945835914, 33.86827006525202]; improved/stable/declined 9/0/0.
- original_common: 26.0402, CI95 [13.262175596953997, 39.51377274327648]; improved/stable/declined 6/0/0.
- revised_common: 27.9552, CI95 [17.653498313845365, 38.52276176836238]; improved/stable/declined 6/0/0.

## 2. Interpretation limits

Read each domain and both common-support reference versions separately. Cross-zero intervals are not equivalence.
Existing Source score plus the same three spectral bands tests practical incremental value; comparison against R6 is not capacity matched.
No winner selected for deployment, no old failure conclusion changed. Oracle table uses evaluation labels and cannot supply deployed thresholds.
No new network training, forward pass or download. Independent CPU recomputation is not independent-data confirmation.
Phase89 recovery provenance limitation remains:54 reused probabilities received hashes after the disk failure, not before it.
