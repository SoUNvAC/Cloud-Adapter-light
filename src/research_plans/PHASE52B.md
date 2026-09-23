# Phase 52B — frozen thin-cloud path with a shadow relational residual

## Four frozen diagnostics

Phase 52A sends 80.4974% of true target shadow pixels to clear, 14.3310% to
thin cloud, 4.2550% to thick cloud, and only 0.9166% to shadow. Predicted
shadow occupancy falls from 6.3531% in source-only to 0.2820%, establishing a
class-collapse failure rather than a boundary-only error.

The selected 65 target patches contain 50,572 shadow pixels (0.2968% of all
selected pixels). Only 8/65 patches contain any shadow, covering forest,
grass/crops, shrubland, snow/ice, urban, and water (6/8 biomes).

At Phase 52 output index 2, cosine distance from the true-shadow prototype is
0.0769 to thin cloud, 0.0866 to a nonzero-radiance dark-clear proxy, 0.1603 to
thick cloud, and 0.1663 to all clear. The proxy is the darkest luminance
quartile of true-clear feature cells after excluding 9,906 zero-radiance
padding cells; its threshold is 25.8208/255. Water and mountain-shadow labels
do not exist, so no claim is made about those individual land-cover causes.

## Frozen mechanism

The Phase 52A iteration-1,000 checkpoint is the immutable base. DINOv2-S,
source Cloud-Adapter, MsRE, rank-8 target head delta, and Mask2Former are all
frozen. A zero-initialized target-only residual reads output-index-2 features,
the four frozen base probabilities, and reconstructed RGB luminance. It uses a
389-to-32 point projection, a 32-channel depth-wise 3x3 convolution with
dilation 3, and a 32-to-1 point output.

The residual changes only the shadow log-odds. The clear/thin/thick conditional
ratios are preserved exactly. It is trained on the same frozen 65-patch budget
for seed 52 and 4,000 iterations. Every local batch of four contains one of
the eight shadow-positive patches. Losses are fixed at balanced BCE 1.0,
Tversky (alpha 0.3, beta 0.7) 1.0, and one-pixel shadow-boundary BCE 0.5.

## Pre-registered gates

- zero initialization reproduces Phase 52A normalized probabilities within
  `1e-5` and exactly preserves predictions;
- only the residual is trainable, with at most 0.1M new parameters;
- target mIoU at least 48.0;
- shadow IoU at least 16.471;
- thin-cloud IoU at least 28.630;
- thin-cloud Boundary F1 at least 19.934;
- all target paths disabled reproduce source-val at at least 73.93 mIoU;
- all 535 source-val and 1,905 target-val images are evaluated with finite
  metrics while target-test and CloudSEN internal test remain sealed.

Any failed gate executes `stop_shadow_residual_route`. Loss weights, batch
composition, branch width/dilation, schedule, checkpoint, or thresholds must
not be adjusted afterward.

## Result and stop decision

Training completed all 4,000 iterations and selected iteration 3,000. The
first evaluation attempt stopped before its first source image because the
custom wrapper omitted the original FP16 autocast and presented Float features
to Half Mask2Former weights. Restoring the evaluation precision context did
not change the checkpoint, model, data, or thresholds; the complete evaluation
was then rerun.

The residual reached 43.9302 target mIoU. Shadow IoU recovered from Phase
52A's 0.8403 to 8.3945, but remained below 16.471. Thin-cloud IoU fell from
29.6299 to 23.5214, and thin Boundary F1 fell from 20.9335 to 15.8784. Shadow
Boundary F1 was 9.3660. The source, parameter, completeness, finite-value, and
seal gates passed, but every preregistered target performance gate failed.

Phase 52B therefore executes `stop_shadow_residual_route`. Although the branch
partially restores shadow, increasing shadow probability takes pixels from the
previously successful thin-cloud path even when the conditional ratios among
non-shadow classes are mathematically fixed. No loss, threshold, width,
dilation, sampling, or schedule adjustment is permitted.
