#!/usr/bin/env python3
"""Run only hierarchical Stage 1 and persist reusable region predictions."""

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

from utils.build_region_dataset import STAGE1_PROMPT
from inference.inference_hierarchical import (
    DEFAULT_DATASET_ROOT,
    decode_generated_layout,
    generate_layout_text,
    load_model_and_tokenizer,
    load_scenes,
    model_world_size,
    prepare_scene_point_cloud,
    prompt_with_point_token,
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
                prompts.add(prompt_with_point_token(str(content)))
            break
    if len(prompts) != 1:
        raise ValueError(
            f"Expected one shared Stage-1 prompt in {path}, found {len(prompts)}."
        )
    return next(iter(prompts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the hierarchical region/layout model without loading Stage 2. "
            "The flat output directory can be reused by any later Stage-2 method."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data_json", type=Path)
    input_group.add_argument("--point_cloud", type=Path)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stage1_model_path", required=True)
    parser.add_argument(
        "--prompt_from_data_json",
        action="store_true",
        help=(
            "Use the shared user prompt stored in --data_json instead of the "
            "default SpatialLM20 walls/doors/windows/regions prompt."
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
    if args.prompt_from_data_json and args.data_json is None:
        parser.error("--prompt_from_data_json requires --data_json.")
    return args


def main() -> int:
    args = parse_args()
    configure_reproducibility(args.seed, args.deterministic)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_scenes(args)
    if not scenes:
        raise ValueError("No scenes selected for Stage-1 inference.")

    model, tokenizer = load_model_and_tokenizer(
        args.stage1_model_path,
        args.inference_dtype,
        args.device,
    )
    num_bins = int(model.config.point_config["num_bins"])
    world_size = model_world_size(model)
    prompt = (
        prompt_from_data_json(args.data_json)
        if args.prompt_from_data_json
        else prompt_with_point_token(STAGE1_PROMPT)
    )

    failures: list[tuple[str, str]] = []
    for scene in tqdm(scenes, desc="Reusable Stage-1 inference"):
        output_path = args.output_dir / f"{scene.scene_id}.txt"
        if args.skip_existing and output_path.exists():
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
        "Wrote reusable Stage-1 predictions: "
        f"output_dir={args.output_dir}, seed={args.seed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
