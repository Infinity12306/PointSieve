#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/scorer_pipeline.py precompute --config configs/0923_scorer.yaml
