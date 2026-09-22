#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/run_phase44_final_audit.py
