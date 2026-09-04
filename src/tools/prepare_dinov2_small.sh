#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints

raw_checkpoint="checkpoints/dinov2_vits14_pretrain.pth"
converted_checkpoint="checkpoints/dinov2_s_converted_512x512.pth"

if [[ ! -f "${raw_checkpoint}" ]]; then
  wget -O "${raw_checkpoint}" \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
fi

python tools/convert_models/convert_dinov2.py \
  "${raw_checkpoint}" \
  "${converted_checkpoint}" \
  --kernel 16 \
  --height 512 \
  --width 512

ls -lh "${raw_checkpoint}" "${converted_checkpoint}"
