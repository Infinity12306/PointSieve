#!/usr/bin/env python3
"""Unified PointSieve evaluation and post-processing entry point."""

from __future__ import annotations

import runpy
import sys


MODES = {
    "base": "utils.eval_base",
    "hierarchical": "utils.eval_hierarchical",
    "token_bins": "utils.eval_token_bins",
    "nms": "utils.apply_bbox_nms",
    "aggregate": "utils.aggregate_metrics",
    "formal": "utils.formal_eval",
}


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: python eval.py {base|hierarchical|token_bins|nms|aggregate|formal} ...")
        print("  base         Evaluate the original SpatialLM output format.")
        print("  hierarchical Evaluate hierarchical object/layout predictions.")
        print("  token_bins   Report metrics by raw point-token length.")
        print("  nms          Apply class-wise 3D box NMS.")
        print("  aggregate    Aggregate repeated-run JSON reports.")
        print("  formal       Run the fixed-seed formal evaluation workflow.")
        return
    mode = argv.pop(0) if argv and argv[0] in MODES else "base"
    module = MODES[mode]
    sys.argv = [module, *argv]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
