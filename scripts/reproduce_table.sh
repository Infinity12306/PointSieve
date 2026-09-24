#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${INFERENCE_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${GPUS}" ]]; then
  echo "Set INFERENCE_GPUS or CUDA_VISIBLE_DEVICES before formal evaluation." >&2
  exit 2
fi

python eval.py formal \
  --config configs/eval_spatiallm_test.yaml \
  --phase all \
  --gpus "${GPUS}"
