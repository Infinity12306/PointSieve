#!/usr/bin/env python3
"""Run SpatialLM hierarchical two-stage SFT from a single config file."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"base", "stage1", "stage2"}
    }


def normalize_step_aliases(stage_config: dict[str, Any], stage_name: str) -> None:
    aliases = {
        "save_step": "save_steps",
        "eval_step": "eval_steps",
    }
    for alias, target in aliases.items():
        if alias not in stage_config:
            continue
        if target in stage_config and stage_config[target] != stage_config[alias]:
            raise ValueError(
                f"{stage_name} config has conflicting {alias} and {target} values."
            )
        stage_config[target] = stage_config.pop(alias)


def apply_optional_int_override(
    stage_config: dict[str, Any],
    key: str,
    value: int | None,
) -> None:
    if value is not None:
        stage_config[key] = value


def apply_optional_float_override(
    stage_config: dict[str, Any],
    key: str,
    value: float | None,
) -> None:
    if value is not None:
        stage_config[key] = value


def apply_optional_str_override(
    stage_config: dict[str, Any],
    key: str,
    value: str | None,
) -> None:
    if value is not None:
        stage_config[key] = value


def latest_checkpoint(output_dir: Path) -> Path:
    if not output_dir.exists():
        raise FileNotFoundError(f"Stage output directory does not exist: {output_dir}")

    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))

    if checkpoints:
        return max(checkpoints, key=lambda item: item[0])[1]

    model_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    if any((output_dir / name).exists() for name in model_files):
        return output_dir

    raise FileNotFoundError(f"No checkpoint or model file found in {output_dir}")


def validate_resume_checkpoint(checkpoint_dir: Path, stage_name: str) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"{stage_name} resume checkpoint does not exist: {checkpoint_dir}"
        )
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(
            f"{stage_name} resume checkpoint must be a directory: {checkpoint_dir}"
        )
    return checkpoint_dir


def write_stage_config(stage_config: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(stage_config, f, sort_keys=False)


def run_stage(
    train_script: Path,
    stage_name: str,
    stage_config: dict[str, Any],
    wandb_project: str | None = None,
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{stage_name}.yaml",
        delete=False,
        encoding="utf-8",
    ) as f:
        config_path = Path(f.name)
        yaml.safe_dump(stage_config, f, sort_keys=False)

    print(f"[{stage_name}] config: {config_path}")
    print(f"[{stage_name}] dataset: {stage_config.get('dataset')}")
    print(f"[{stage_name}] output_dir: {stage_config.get('output_dir')}")
    print(f"[{stage_name}] world_size: {stage_config.get('world_size')}")
    print(f"[{stage_name}] max_point_tokens: {stage_config.get('max_point_tokens')}")
    print(f"[{stage_name}] save_steps: {stage_config.get('save_steps')}")
    print(f"[{stage_name}] eval_steps: {stage_config.get('eval_steps')}")
    if stage_config.get("resume_from_checkpoint") is not None:
        print(f"[{stage_name}] resume_from_checkpoint: {stage_config['resume_from_checkpoint']}")
    env = None
    if wandb_project:
        env = dict(os.environ)
        env["WANDB_PROJECT"] = wandb_project
        print(f"[{stage_name}] wandb_project: {wandb_project}")
    subprocess.run(
        [sys.executable, str(train_script), str(config_path)],
        check=True,
        env=env,
    )


def set_stage_run_name(
    stage_config: dict[str, Any],
    stage_name: str,
    wandb_run_name: str | None,
) -> None:
    stage_wandb_run_name = stage_config.pop("wandb_run_name", None)
    if stage_config.get("run_name") is not None:
        return
    if stage_wandb_run_name:
        stage_config["run_name"] = stage_wandb_run_name
    elif wandb_run_name:
        stage_config["run_name"] = f"{wandb_run_name}_{stage_name}"


def build_stage_configs(
    config: dict[str, Any],
    require_stage1: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wandb_run_name = config.get("wandb_run_name")
    base = config.get("base", {})
    stage1 = deep_merge(base, config.get("stage1", {})) if require_stage1 or config.get("stage1") else {}
    stage2 = deep_merge(base, config.get("stage2", {}))

    if require_stage1 and "dataset" not in stage1:
        raise ValueError("stage1.dataset must be configured.")
    if "dataset" not in stage2:
        raise ValueError("stage2.dataset must be configured.")
    if require_stage1 and "output_dir" not in stage1:
        raise ValueError("stage1.output_dir must be configured.")
    if "output_dir" not in stage2:
        raise ValueError("stage2.output_dir must be configured.")

    if stage1:
        normalize_step_aliases(stage1, "stage1")
        set_stage_run_name(stage1, "stage1", wandb_run_name)
    normalize_step_aliases(stage2, "stage2")
    set_stage_run_name(stage2, "stage2", wandb_run_name)

    return flatten_config(stage1), flatten_config(stage2)


def resolve_auto_stage2_model_path(
    stage1_config: dict[str, Any],
    stage2_config: dict[str, Any],
) -> None:
    if stage2_config.get("model_name_or_path") not in {None, "AUTO_STAGE1_LATEST"}:
        return
    if "output_dir" not in stage1_config:
        raise ValueError(
            "stage2.model_name_or_path is AUTO_STAGE1_LATEST, but stage1.output_dir "
            "is unavailable. Configure stage1.output_dir or provide "
            "--stage2_model_name_or_path for stage-2-only training."
        )
    stage2_config["model_name_or_path"] = str(
        latest_checkpoint(Path(stage1_config["output_dir"]))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run first-stage region SFT followed by second-stage bbox SFT."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--train_script",
        "--train-script",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "train.py",
    )
    parser.add_argument(
        "--skip_stage1",
        "--skip-stage1",
        action="store_true",
        help="Use the latest checkpoint in stage1.output_dir and only run stage2.",
    )
    parser.add_argument(
        "--stage2_only",
        "--stage2-only",
        action="store_true",
        help=(
            "Run stage 2 only. If stage2.model_name_or_path is AUTO_STAGE1_LATEST, "
            "stage1.output_dir is still used to find the latest stage-1 checkpoint."
        ),
    )
    parser.add_argument(
        "--stage2_model_name_or_path",
        "--stage2-model-name-or-path",
        type=str,
        default=None,
        help="Override stage2.model_name_or_path, useful for stage-2-only training.",
    )
    parser.add_argument(
        "--stage1_resume_from_checkpoint",
        "--stage1-resume-from-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume stage 1 training from a full Trainer checkpoint directory "
            "including trainer/optimizer/scheduler state."
        ),
    )
    parser.add_argument(
        "--skip_stage2",
        "--skip-stage2",
        action="store_true",
        help="Only run stage1.",
    )
    parser.add_argument("--stage1_save_steps", "--stage1-save-steps", type=int)
    parser.add_argument("--stage1_eval_steps", "--stage1-eval-steps", type=int)
    parser.add_argument("--stage1_world_size", "--stage1-world-size", type=float)
    parser.add_argument(
        "--stage1_max_point_tokens",
        "--stage1-max-point-tokens",
        type=int,
    )
    parser.add_argument("--stage1_output_dir", "--stage1-output-dir", type=str)
    parser.add_argument("--stage2_save_steps", "--stage2-save-steps", type=int)
    parser.add_argument("--stage2_eval_steps", "--stage2-eval-steps", type=int)
    parser.add_argument("--stage2_world_size", "--stage2-world-size", type=float)
    parser.add_argument(
        "--stage2_max_point_tokens",
        "--stage2-max-point-tokens",
        type=int,
    )
    parser.add_argument("--stage2_output_dir", "--stage2-output-dir", type=str)
    parser.add_argument(
        "--print_configs",
        "--print-configs",
        action="store_true",
        help="Print resolved stage configs and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    wandb_project = config.get("wandb_project")

    config_stage2_only = bool(
        config.get("stage2_only", False) or config.get("skip_stage1", False)
    )
    config_skip_stage2 = bool(config.get("skip_stage2", False))
    stage2_only = args.stage2_only or args.skip_stage1 or config_stage2_only
    skip_stage2 = args.skip_stage2 or config_skip_stage2
    if stage2_only and skip_stage2:
        raise ValueError("Cannot combine stage-2-only mode with skip_stage2.")
    if stage2_only and args.stage1_resume_from_checkpoint is not None:
        raise ValueError(
            "--stage1_resume_from_checkpoint is not meaningful in stage-2-only mode."
        )

    configured_stage2_model = config.get("stage2", {}).get(
        "model_name_or_path",
        config.get("base", {}).get("model_name_or_path"),
    )
    stage1_required = not stage2_only or (
        args.stage2_model_name_or_path is None
        and configured_stage2_model in {None, "AUTO_STAGE1_LATEST"}
    )
    stage1_config, stage2_config = build_stage_configs(
        config,
        require_stage1=stage1_required,
    )

    if not stage2_only and args.stage1_resume_from_checkpoint is not None:
        stage1_config["resume_from_checkpoint"] = str(
            validate_resume_checkpoint(
                args.stage1_resume_from_checkpoint,
                "stage1",
            )
        )
    elif not stage2_only and stage1_config.get("resume_from_checkpoint") is not None:
        stage1_config["resume_from_checkpoint"] = str(
            validate_resume_checkpoint(
                Path(stage1_config["resume_from_checkpoint"]),
                "stage1",
            )
        )

    if args.stage2_model_name_or_path is not None:
        stage2_config["model_name_or_path"] = args.stage2_model_name_or_path

    apply_optional_int_override(stage1_config, "save_steps", args.stage1_save_steps)
    apply_optional_int_override(stage1_config, "eval_steps", args.stage1_eval_steps)
    apply_optional_float_override(stage1_config, "world_size", args.stage1_world_size)
    apply_optional_int_override(
        stage1_config,
        "max_point_tokens",
        args.stage1_max_point_tokens,
    )
    apply_optional_str_override(stage1_config, "output_dir", args.stage1_output_dir)
    apply_optional_int_override(stage2_config, "save_steps", args.stage2_save_steps)
    apply_optional_int_override(stage2_config, "eval_steps", args.stage2_eval_steps)
    apply_optional_float_override(stage2_config, "world_size", args.stage2_world_size)
    apply_optional_int_override(
        stage2_config,
        "max_point_tokens",
        args.stage2_max_point_tokens,
    )
    apply_optional_str_override(stage2_config, "output_dir", args.stage2_output_dir)

    if args.print_configs:
        print(
            json.dumps(
                {
                    "stage2_only": stage2_only,
                    "skip_stage2": skip_stage2,
                    "wandb_project": wandb_project,
                    "stage1": stage1_config,
                    "stage2": stage2_config,
                },
                indent=2,
            )
        )
        return

    if stage2_only:
        print("[stage1] skipped because stage2_only is enabled.")
    else:
        run_stage(
            args.train_script,
            "stage1",
            stage1_config,
            wandb_project=wandb_project,
        )

    if skip_stage2:
        return

    resolve_auto_stage2_model_path(stage1_config, stage2_config)

    run_stage(
        args.train_script,
        "stage2",
        stage2_config,
        wandb_project=wandb_project,
    )


if __name__ == "__main__":
    main()
