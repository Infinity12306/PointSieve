#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/scorer_pipeline.py filter --config configs/scorer.yaml
