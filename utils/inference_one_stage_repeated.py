#!/usr/bin/env python3
"""Run reproducible one-stage SpatialLM inference for a fixed test split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from tqdm import tqdm
from transformers import set_seed

from utils.inference_base import DETECT_TYPE_PROMPT
from utils.inference_hierarchical import (
    POINT_PROMPT,
    apply_subset_args,
    decode_generated_layout,
    generate_layout_text,
    load_model_and_tokenizer,
    model_world_size,
    prepare_scene_point_cloud,
    scenes_from_json,
)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    set_seed(seed)
    if not deterministic:
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_code_template(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = Path(__file__).resolve().parents[1] / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(path)


def prompt_from_data_json(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = (
            payload.get("data")
            or payload.get("examples")
            or payload.get("items")
            or []
        )
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of examples in {path}")

    prompts: set[str] = set()
    for item in payload:
        conversations = item.get("conversations") or item.get("messages") or []
        for message in conversations:
            role = message.get("from") or message.get("role")
            if role not in {"human", "user"}:
                continue
            content = message.get("value") or message.get("content")
            if content:
                prompts.add(str(content).replace("<point_cloud>", POINT_PROMPT))
            break
    if len(prompts) != 1:
        raise ValueError(
            f"Expected one shared one-stage prompt in {path}, found {len(prompts)}."
        )
    return next(iter(prompts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the original one-stage SpatialLM on a JSON test split with "
            "the same fixed seeds used by the hierarchical comparison."
        )
    )
    parser.add_argument("--data_json", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--code_template",
        type=Path,
        default=Path("code_template.txt"),
    )
    parser.add_argument(
        "--prompt_from_data_json",
        action="store_true",
        help=(
            "Use the shared user prompt stored in --data_json. This is required "
            "for fine-tuned datasets whose task prompt differs from the default."
        ),
    )
    parser.add_argument("--inference_dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no_cleanup", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.num_shards < 1:
        parser.error("--num_shards must be positive.")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard_index must satisfy 0 <= index < num_shards.")
    if args.num_beams < 1:
        parser.error("--num_beams must be positive.")
    return args


def main() -> int:
    args = parse_args()
    configure_reproducibility(args.seed, args.deterministic)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenes = apply_subset_args(
        scenes_from_json(args.data_json, args.dataset_root),
        args.start_index,
        args.end_index,
        args.limit,
        args.num_shards,
        args.shard_index,
    )
    if not scenes:
        raise ValueError("No scenes selected for one-stage inference.")

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.inference_dtype,
        args.device,
    )
    num_bins = int(model.config.point_config["num_bins"])
    world_size = model_world_size(model)
    if args.prompt_from_data_json:
        prompt = prompt_from_data_json(args.data_json)
    else:
        code_template = resolve_code_template(args.code_template).read_text(
            encoding="utf-8"
        )
        prompt = (
            f"{POINT_PROMPT}{DETECT_TYPE_PROMPT['all']} "
            f"The reference code is as followed: {code_template}"
        )

    failures: list[tuple[str, str]] = []
    for scene in tqdm(scenes, desc="One-stage SpatialLM inference"):
        output_path = args.output_dir / f"{scene.scene_id}.txt"
        if args.skip_existing and output_path.is_file():
            continue
        try:
            scene_pcd = prepare_scene_point_cloud(
                scene.pcd_path,
                num_bins,
                args.no_cleanup,
                world_size=world_size,
            )
            generated = generate_layout_text(
                model,
                tokenizer,
                prompt,
                scene_pcd.input_tensor,
                args,
            )
            layout = decode_generated_layout(
                generated,
                scene_pcd.min_extent,
                num_bins,
                world_size=world_size,
            )
            atomic_write_text(output_path, layout.to_language_string())
            (args.output_dir / "errors" / f"{scene.scene_id}.txt").unlink(
                missing_ok=True
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append((scene.scene_id, str(exc)))
            atomic_write_text(
                args.output_dir / "errors" / f"{scene.scene_id}.txt",
                str(exc),
            )

    if failures:
        print(f"Completed with {len(failures)} failure(s).", file=sys.stderr)
        for scene_id, error in failures[:10]:
            print(f"{scene_id}: {error}", file=sys.stderr)
        return 1

    print(
        "Wrote one-stage SpatialLM predictions: "
        f"output_dir={args.output_dir}, seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
