#!/usr/bin/env python3
"""Unified PointSieve inference entry point."""

from __future__ import annotations

import runpy
import sys


MODES = {
    "base": "inference.inference_base",
    "hierarchical": "inference.inference_hierarchical",
    "stage1": "inference.inference_stage1",
    "stage2": "inference.inference_stage2",
    "scorer": "inference.inference_scorer",
    "one_stage": "inference.inference_one_stage_repeated",
}


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: python inference.py {base|hierarchical|stage1|stage2|scorer|one_stage} ...")
        print("  base         Run the original single-scene SpatialLM inference.")
        print("  hierarchical Run full hierarchical inference directly.")
        print("  stage1       Generate reusable Stage 1 region predictions.")
        print("  stage2       Run Stage 2 from saved Stage 1 predictions.")
        print("  scorer       Run scorer-filtered hierarchical inference.")
        print("  one_stage    Run fixed-seed one-stage baseline inference.")
        return
    mode = argv.pop(0) if argv and argv[0] in MODES else "base"
    module = MODES[mode]
    sys.argv = [module, *argv]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
