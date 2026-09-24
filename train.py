#!/usr/bin/env python3
"""Unified PointSieve training entry point.

Examples::

    python train.py hierarchical configs/0923_train_hierarchical.yaml
    python train.py scorer --train_cache_dir artifacts/scorer_cache/train ...
    python train.py filtered configs/0923_train_filtered_stage2.yaml

Calling ``python train.py <config>`` keeps the low-level SFT trainer
compatible with the command used by the hierarchical launcher.
"""

from __future__ import annotations

import runpy
import sys


MODES = {
    "sft": "train.sft_train",
    "hierarchical": "train.train_hierarchical",
    "scorer": "train.train_scorer",
    "filtered": "train.train_filtered_stage2",
}


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: python train.py {sft|hierarchical|scorer|filtered} ...")
        print("  sft          Run the low-level SpatialLM SFT trainer.")
        print("  hierarchical Train Stage 1 and Stage 2 from one YAML config.")
        print("  scorer       Train the point-token scorer from cached features.")
        print("  filtered     Adapt Stage 2 using scorer-filtered caches.")
        return
    mode = argv.pop(0) if argv and argv[0] in MODES else "sft"
    module = MODES[mode]
    sys.argv = [module, *argv]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
