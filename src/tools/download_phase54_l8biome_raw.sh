#!/usr/bin/env bash
set -euo pipefail

root="${1:-../shared/data/l8_biome_raw}"
revision="f76df19accce34d2acc1878d88b9491bc81f94c8"
base="https://hf.co/datasets/torchgeo/l8biome/resolve/${revision}"
mkdir -p "${root}"
cd "${root}"

declare -A md5=(
  [barren]="0eb691822d03dabd4f5ea8aadd0b41c3"
  [forest]="4a5645596f6bb8cea44677f746ec676e"
  [grass_crops]="a69ed5d6cb227c5783f026b9303cdd3c"
  [shrubland]="19df1d0a604faf6aab46d6a7a5e6da6a"
  [snow_ice]="af8b189996cf3f578e40ee12e1f8d0c9"
  [urban]="5450195ed95ee225934b9827bea1e8b0"
  [water]="a81153415eb662c9e6812c2a8e38c743"
  [wetlands]="1f86cc354631ca9a50ce54b7cab3f557"
)

for biome in barren forest grass_crops shrubland snow_ice urban water wetlands; do
  archive="${biome}.tar.gz"
  curl -L --fail --retry 10 -C - -o "${archive}" "${base}/${archive}"
  printf '%s  %s\n' "${md5[$biome]}" "${archive}" | md5sum --check --status
  marker=".${biome}.extracted"
  if [[ ! -f "${marker}" ]]; then
    tar -xzf "${archive}"
    touch "${marker}"
  fi
done

touch DOWNLOAD_COMPLETE

