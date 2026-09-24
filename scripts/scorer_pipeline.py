#!/usr/bin/env python3
"""Small YAML-driven wrapper for the three scorer preparation phases."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load(path_arg: Path) -> dict:
    with path_arg.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path_arg}")
    return config


def run(command: list[str]) -> None:
    print("$", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def precompute(config: dict) -> None:
    root = path(config["dataset_root"])
    cache_root = path(config["cache_root"])
    message_root = path(config["message_cache_root"])
    model = path(config["stage2_model"])
    for split, epochs in (("train", int(config["train_epochs"])), ("val", 1)):
        suffix = "_20000" if split == "train" else ""
        dataset_json = root / f"spatiallm_stage2_bbox_{split}{suffix}.json"
        run(
            [
                sys.executable,
                "-m",
                "utils.precompute_scorer_data",
                "--dataset_json",
                str(dataset_json),
                "--dataset_root",
                str(root),
                "--model_path",
                str(model),
                "--output_dir",
                str(cache_root / split),
                "--message_output_dir",
                str(message_root / split),
                "--num_epochs",
                str(epochs),
                "--shard_size",
                str(config["cache_shard_size"]),
                "--world_size",
                str(config["world_size"]),
                "--num_bins",
                str(config["num_bins"]),
                "--bbox_expand_ratio",
                str(config["bbox_expand_ratio"]),
                "--seed",
                str(config["seed"]),
                "--torch_dtype",
                "float32",
                "--storage_dtype",
                str(config["storage_dtype"]),
                "--random_rotation",
                "--overwrite",
            ]
        )


def train(config: dict) -> None:
    scorer = config["scorer"]
    command = [
        sys.executable,
        "train.py",
        "scorer",
        "--train_cache_dir",
        str(path(config["cache_root"]) / "train"),
        "--eval_cache_dir",
        str(path(config["cache_root"]) / "val"),
        "--output_dir",
        str(path(config["scorer_output"])),
        "--projector_model_path",
        str(path(config["stage2_model"])),
        "--projector_torch_dtype",
        str(config["projector_dtype"]),
        "--num_train_epochs",
        str(config["train_epochs"]),
        "--threshold",
        str(config["threshold"]),
        "--pos_weight",
        "auto",
    ]
    argument_map = {
        "per_device_train_batch_size": "--per_device_train_batch_size",
        "per_device_eval_batch_size": "--per_device_eval_batch_size",
        "gradient_accumulation_steps": "--gradient_accumulation_steps",
        "learning_rate": "--learning_rate",
        "weight_decay": "--weight_decay",
        "hidden_dim": "--hidden_dim",
        "num_layers": "--num_layers",
        "num_heads": "--num_heads",
        "ffn_dim": "--ffn_dim",
        "dropout": "--dropout",
        "num_workers": "--num_workers",
        "eval_steps": "--eval_steps",
        "save_steps": "--save_steps",
        "logging_steps": "--logging_steps",
        "save_total_limit": "--save_total_limit",
    }
    for key, flag in argument_map.items():
        command.extend([flag, str(scorer[key])])
    command.extend(
        [
            "--wandb_project",
            "pointsieve",
            "--wandb_run_name",
            "pointsieve_scorer_steps29488",
        ]
    )
    run(command)


def filter_tokens(config: dict) -> None:
    root = path(config["dataset_root"])
    run(
        [
            sys.executable,
            "-m",
            "utils.filter_scorer_tokens",
            "--input_cache_root",
            str(path(config["cache_root"])),
            "--output_root",
            str(path(config["filtered_cache_root"])),
            "--message_cache_root",
            str(path(config["message_cache_root"])),
            "--splits",
            "train",
            "val",
            "--scorer_path",
            str(path(config["scorer_checkpoint"])),
            "--projector_model_path",
            str(path(config["stage2_model"])),
            "--dataset_root",
            str(root),
            "--dataset_json",
            f"train={root / 'spatiallm_stage2_bbox_train_20000.json'}",
            "--dataset_json",
            f"val={root / 'spatiallm_stage2_bbox_val.json'}",
            "--threshold",
            str(config["threshold"]),
            "--max_keep",
            str(config["max_keep"]),
            "--min_keep",
            str(config["min_keep"]),
            "--shard_size",
            str(config["cache_shard_size"]),
            "--batch_size",
            str(config["filter_batch_size"]),
            "--num_bins",
            str(config["num_bins"]),
            "--world_size",
            str(config["world_size"]),
            "--bbox_expand_ratio",
            str(config["bbox_expand_ratio"]),
            "--projector_torch_dtype",
            str(config["projector_dtype"]),
            "--storage_dtype",
            str(config["storage_dtype"]),
            "--overwrite",
        ]
    )
    for split in ("train", "val"):
        run(
            [
                sys.executable,
                "-m",
                "utils.merge_filtered_cache",
                "--cache_dir",
                str(path(config["filtered_cache_root"]) / split),
                "--overwrite",
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("precompute", "train", "filter"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "0923_scorer.yaml")
    args = parser.parse_args()
    config = load(args.config)
    {"precompute": precompute, "train": train, "filter": filter_tokens}[args.phase](config)


if __name__ == "__main__":
    main()
