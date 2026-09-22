#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python tools/run_phase39_pareto_audit.py
