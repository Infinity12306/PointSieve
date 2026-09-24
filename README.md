# PointSieve

Minimal training and evaluation code for **PointSieve: Enhancing Spatial Perception of Large Language Models for Long-Context 3D Object Detection**. The release follows the final four-page paper: region based two-stage detection, supervised point-token scoring, scorer filtered Stage 2 adaptation, and paper-compatible SpatialLM-Dataset evaluation.

The repository contains source code and small YAML files only. Model weights, point clouds, generated caches, logs, and prediction files are intentionally external.

## What is reproduced

The default protocol is the 20,000-scene SpatialLM-Dataset training subset and the official 500-scene test split. It runs three fixed repeats with seeds `42, 43, 44`, BF16 generation, class-wise NMS at `0.1`, and reports `F1_all` and class-balanced `F1_class` at 3D IoU `0.25` and `0.50`.

| method | F1-all @0.25 | F1-all @0.50 | F1-class @0.25 | F1-class @0.50 |
| --- | ---: | ---: | ---: | ---: |
| SpatialLM | 72.85 ± 0.37 | 60.39 ± 0.20 | 65.64 ± 0.45 | 54.78 ± 0.12 |
| SpatialLM + staged | 73.74 ± 0.50 | 63.93 ± 0.13 | 65.52 ± 0.16 | 57.98 ± 0.31 |
| PointSieve | 73.71 ± 1.02 | 64.96 ± 0.45 | 65.73 ± 0.98 | 58.95 ± 0.59 |

The scorer threshold table uses the same adapted Stage 2 model and evaluates thresholds `0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0`. Numeric-token and classification auxiliary losses are not part of PointSieve.

## Installation

Use Python 3.11, PyTorch 2.4.1, and a CUDA 12.4 environment. Install PyTorch for the CUDA version on the target machine first, then install the Python dependencies:

```bash
git clone <your-github-url>/PointSieve.git
cd PointSieve
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

SpatialLM 1.1 uses Sonata sparse operators. Install matching wheels for `spconv`, `torch-scatter`, `torchsparse`, and `flash-attn` in the same environment. Their versions depend on the local PyTorch/CUDA build; the upstream [SpatialLM installation guide](https://github.com/manycore-research/SpatialLM#installation) is the reference for these compiled dependencies.

## External files

Download the training [SpatialLM1.1-Qwen-0.5B checkpoint](https://huggingface.co/manycore-research/SpatialLM1.1-Qwen-0.5B), the released [ScanNet-SFT checkpoint](https://huggingface.co/ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT), and the [SpatialLM-Dataset](https://huggingface.co/datasets/manycore-research/SpatialLM-Dataset). Keep large files outside Git and expose them through this layout:

```text
data/spatiallm/
  spatiallm_train.json
  spatiallm_val.json
  spatiallm_test.json
  pcd/
  layout/
  split.csv
```

The small label map used by the evaluator is included as `benchmark_categories.tsv`.

For example, with the Hugging Face CLI:

```bash
hf download manycore-research/SpatialLM-Dataset --repo-type dataset --local-dir data/spatiallm
hf download ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT --local-dir artifacts/base
```

Set `base.model_name_or_path` in the training YAML to a local copy of the training checkpoint if it is not available from the Hub at runtime.

The dataset download must include the official point clouds, layout annotations, category mapping, and split metadata. The commands below create derived region JSON files and region point clouds under `data/spatiallm/`; these generated files are also excluded from Git. For training, download `manycore-research/SpatialLM1.1-Qwen-0.5B`; for the released one-stage comparison, download `ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT` into `artifacts/base/`.

For exact paper numbers, provide the five derived artifacts listed below. They are not bundled because they contain model weights or generated caches:

```text
artifacts/stage1/checkpoint-5000/
artifacts/stage2_plain/checkpoint-14392/                 # plain two-stage ablation
artifacts/stage2_bboxmask/checkpoint-14392/       # scorer source model
artifacts/stage2_scorer_filtered/checkpoint-7000/ # PointSieve model
artifacts/scorer/checkpoint-29488/scorer.pt
```

You can instead train every artifact with the configs in `configs/`. Set `model_name_or_path`, `output_dir`, and artifact paths to local paths before running.

## Reproduction workflow

Run commands from the repository root.

### 1. Build region supervision

```bash
python -m utils.build_region_dataset \
  --dataset_root data/spatiallm \
  --train_json data/spatiallm/spatiallm_train.json \
  --split train --sample_size 20000 --seed 0 --k 3 \
  --expand_fraction 0.25 --copy_mode symlink
```

This writes `spatiallm_stage1_region_train_20000.json`, `spatiallm_stage2_bbox_train_20000.json`, and the corresponding region files. For the complete validation split used by the training and scorer configs, run:

```bash
python -m utils.build_region_dataset \
  --dataset_root data/spatiallm \
  --train_json data/spatiallm/spatiallm_val.json \
  --split val --sample_size 0 --seed 0 --k 3 \
  --expand_fraction 0.25 --copy_mode symlink
```

Build the official test split for the final three-seed report in the same way:

```bash
python -m utils.build_region_dataset \
  --dataset_root data/spatiallm \
  --train_json data/spatiallm/spatiallm_test.json \
  --split test --sample_size 0 --seed 0 --k 3 \
  --expand_fraction 0.25 --copy_mode symlink
```

### 2. Train the two stages

`configs/0923_train_hierarchical.yaml` records the paper settings: full-scene Stage 1 at `world_size=32`, regional Stage 2 at `world_size=16` with `max_point_tokens=4096`, cosine decay, warmup ratio `0.03`, W&B logging, and no auxiliary numeric/classification loss.

```bash
python train.py hierarchical configs/0923_train_hierarchical.yaml
```

The config trains Stage 1 and Stage 2 from the same official base checkpoint. Use `--stage2-only --stage2-model-name-path ...` to rerun a single stage. The released one-stage comparison is the `ysmao/SpatialLM1.1-Qwen-0.5B-ScanNet-SFT` checkpoint described above.

### 3. Train the point-token scorer and adapt Stage 2

The scorer is trained on frozen Stage 2 point features with voxel-box intersection labels. The cache and adaptation settings are documented in `configs/0923_scorer.yaml` and `configs/0923_train_filtered_stage2.yaml`.

```bash
bash scripts/0923_precompute_scorer_cache.sh
bash scripts/0923_train_scorer.sh
bash scripts/0923_filter_point_tokens.sh
python train.py filtered configs/0923_train_filtered_stage2.yaml
```

The paper PointSieve row uses scorer threshold `0.5`, `min_keep=1`, `max_keep=4096`, and the scorer-adapted model at optimizer step `7000`. The SceneScript row in the paper is a separate baseline trained from scratch and is not redistributed here.

### 4. Run the three-seed evaluation

```bash
python eval.py formal \
  --config configs/0923_eval_spatiallm_test.yaml \
  --phase stage1 stage1_eval \
  --gpus 0,1,2,3

python eval.py formal \
  --config configs/0923_eval_spatiallm_test.yaml \
  --phase stage2 stage2_eval \
  --methods hier_res16_max4096_14392 scorer_filtered_fullcache_7000 \
  --gpus 0,1,2,3

python eval.py formal \
  --config configs/0923_eval_spatiallm_test.yaml \
  --phase aggregate --gpus 0
```

The runner reuses the saved Stage 1 predictions for every Stage 2 method, assigns the same seed to the matching repeat, applies class-wise NMS, evaluates each scene, and writes mean and population standard deviation reports under the configured output directory. Use `--dry_run` to inspect commands without loading a model.

## Repository layout

```text
configs/       paper training and evaluation YAML files
scripts/       small cache and scorer wrappers
train/         training implementations behind `train.py`
inference/     inference implementations behind `inference.py`
eval/          evaluation implementations behind `eval.py`
utils/         data preparation, cache, geometry, and rendering helpers
spatiallm/     required SpatialLM model, data, and trainer subset
train.py       unified training entry point
inference.py   unified inference entry point
eval.py        unified evaluation/post-processing entry point
```

The code is derived from [SpatialLM](https://github.com/manycore-research/SpatialLM). See `LICENSE-LLAMA.txt` and `NOTICE` for upstream license and attribution information. The repository does not redistribute any checkpoint or dataset.
