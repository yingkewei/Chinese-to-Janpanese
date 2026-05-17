#!/usr/bin/env python3
"""
Build train/dev/test from the raw parallel file only (no stage1–5 filtering).
Writes the same metrics.json shape as the main pipeline: stages, pre_audit, flags, corpus_stats.
Intermediate stage counts are null (no filtering applied).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_DP = _ROOT / "data_processing"
if str(_DP) not in sys.path:
    sys.path.insert(0, str(_DP))

from stages import (  # noqa: E402
    compute_corpus_stats_splits,
    iter_raw_pairs,
    maybe_limit_pairs,
    split_data,
    stage0_pre_audit,
    write_metrics,
    write_tsv,
)


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw zh-ja baseline (no filtering stages)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=None, help="Override input.max_pairs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = args.config.resolve().parent.parent

    raw_file = project_root / cfg["input"]["raw_file"]
    delimiter = cfg["input"]["delimiter"]
    max_pairs = args.max_pairs if args.max_pairs is not None else cfg["input"].get("max_pairs")
    max_pairs = int(max_pairs) if max_pairs is not None else None
    progress_every = int(cfg["input"].get("progress_every", 100000))

    out_base = project_root / cfg["output"]["base_dir"]
    train_path = out_base / cfg["output"]["train_file"]
    dev_path = out_base / cfg["output"]["dev_file"]
    test_path = out_base / cfg["output"]["test_file"]
    metrics_path = out_base / cfg["output"]["metrics_file"]

    raw_pairs = []
    t0 = time.perf_counter()
    t_last = t0
    for i, pair in enumerate(
        maybe_limit_pairs(iter_raw_pairs(raw_file, delimiter), max_pairs=max_pairs), start=1
    ):
        raw_pairs.append(pair)
        if progress_every > 0 and i % progress_every == 0:
            now = time.perf_counter()
            print(f"[read] {i:,} pairs | last {progress_every:,}: {now - t_last:.2f}s | total: {now - t0:.2f}s")
            t_last = now

    n = len(raw_pairs)
    metrics: Dict[str, Any] = {
        "stages": {
            "stage0_raw_pairs": n,
            "stage1_basic_clean": None,
            "stage2_language_normalize": None,
            "stage3_length_alignment": None,
            "stage4_semantic_filter": None,
            "stage5_dedup_reweight": None,
        },
        "pre_audit": stage0_pre_audit(raw_pairs),
        "flags": {
            "raw_baseline_no_filtering": True,
            "stage4_semantic_enabled": False,
            "stage5_dedup_reweight_enabled": False,
        },
    }

    train, dev, test = split_data(
        raw_pairs,
        dev_ratio=float(cfg["split"]["dev_ratio"]),
        test_ratio=float(cfg["split"]["test_ratio"]),
        seed=int(cfg["split"]["seed"]),
    )
    metrics["stages"]["final_train"] = len(train)
    metrics["stages"]["final_dev"] = len(dev)
    metrics["stages"]["final_test"] = len(test)

    cs_cfg = cfg.get("corpus_stats") or {}
    spm_rel = cs_cfg.get("spm_model")
    spm_path = (project_root / spm_rel) if spm_rel else None
    enc_bs = int(cs_cfg.get("encode_batch_size", 8192))
    metrics["corpus_stats"] = compute_corpus_stats_splits(
        train, dev, test, sp_model_path=spm_path, encode_batch_size=enc_bs
    )

    write_tsv(train_path, train)
    write_tsv(dev_path, dev)
    write_tsv(test_path, test)
    write_metrics(metrics_path, metrics)

    print("Raw baseline done.")
    print(f"Train: {train_path} ({len(train)})")
    print(f"Dev:   {dev_path} ({len(dev)})")
    print(f"Test:  {test_path} ({len(test)})")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
