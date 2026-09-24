#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${INFERENCE_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "${GPUS}" ]]; then
  echo "Set INFERENCE_GPUS or CUDA_VISIBLE_DEVICES before formal evaluation." >&2
  exit 2
fi

python eval.py formal \
  --config configs/0923_eval_spatiallm_test.yaml \
  --phase stage1 stage1_eval \
  --gpus "${GPUS}"
python eval.py formal \
  --config configs/0923_eval_spatiallm_test.yaml \
  --phase stage2 stage2_eval \
  --methods hier_res16_max4096_14392 scorer_filtered_fullcache_7000 \
  --gpus "${GPUS}"
python eval.py formal \
  --config configs/0923_eval_spatiallm_test.yaml \
  --phase aggregate \
  --gpus "${GPUS}"
