# Phase 45 — cross-domain feasibility and upper-bound audit

## Purpose

Phase 45 starts a new research program rather than extending the closed
Phase 31--43 compact-model search.  The proposed paper hypothesis is
physics-semantic factorization plus class-conditional adaptation for
resource-constrained cross-sensor cloud segmentation.  Before implementing
that method, Phase 45 measures whether the target-domain problem is valid and
has enough headroom.

Landsat-8 Biome is a development target only.  It was used by earlier phases
and must never be described as a pristine external holdout.  The CloudSEN
internal test remains sealed.

## Phase 45A: protocol audit

The historical Landsat-8 Biome conversion contains patch-level `train` and
`test` directories.  Phase 45A merges both directories, recovers the original
scene ID from each patch filename, and creates deterministic, biome-stratified
scene-level target-train, target-val, and locked target-test manifests.  It
does not move, copy, edit, or relabel dataset files.

The audit must establish all of the following before any diagnostic training:

- every image has exactly one mask and every filename is parseable;
- no patch is duplicated across the historical directories;
- the replacement split is scene-disjoint and contains every label in every
  partition;
- labels are exactly a subset of Landsat-8 Biome IDs 0--3;
- each partition contains at least one scene from every available biome;
- full pixel counts and class-conditional boundary/interior counts are finite.

Historical train/test scene overlap is reported as evidence and is not a
failure of the replacement protocol.  All generated manifests and reports are
written below `work_dirs/phase45_protocol_audit`; the shared dataset is
read-only.

## Phase 45B: frozen baselines and oracle

Only after Phase 45A passes:

1. evaluate the frozen Phase 22 V8 and Phase 37 compact source-only models on
   target-val without adapting or selecting checkpoints;
2. train supervised target-only oracle models using target-train labels and
   select checkpoints only on target-val;
3. run one simple teacher--student adaptation baseline with target-train
   labels hidden from optimization;
4. report per-class IoU, weak-class mean, macro boundary F1, adaptation time,
   peak adaptation memory, and source-domain forgetting separately.

Boundary F1 is computed independently for each of the four classes from
one-pixel semantic boundaries, with a fixed one-pixel matching tolerance, then
macro-averaged.  This definition is frozen before the first target-val model
evaluation.

The scene-level target-test partition is locked during method development.
Binary HRC-to-GF experiments are secondary evidence and are never averaged
directly with four-class Landsat metrics.

## Pre-registered decision

The primary feasibility gap is oracle target-val mIoU minus the corresponding
source-only target-val mIoU.  V8 is the pre-registered primary feasibility
model; the compact model is a secondary deployment diagnostic.  Both oracles
start from their generic ImageNet/DINO initialization rather than source-domain
segmentation weights and use seed 42 for the first upper-bound audit.

- gap below 5.0 points: stop this target-domain direction;
- gap from 5.0 to below 8.0: continue only if the mean thin-cloud/cloud-shadow
  oracle gap is at least 10.0 points;
- gap at least 8.0: proceed to the factorized model;
- any Phase 45A integrity failure: stop before training and repair or replace
  the dataset protocol.

No threshold may be changed after results are observed.  Phase 45 does not
claim that sensor, geography, and ontology effects are independently causal
unless controlled comparison data actually identify those effects.
