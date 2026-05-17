#!/usr/bin/env python3
"""
Recompute corpus_stats (sentence pairs, char totals, SPM piece totals, length distributions)
from existing train/dev/test TSV and merge into metrics.json — no need to rerun the full pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml

from stages import compute_corpus_stats_from_tsv_splits, write_metrics


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge corpus_stats into data_processing metrics.json")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = args.config.resolve().parent.parent
    out_base = project_root / cfg["output"]["base_dir"]
    train_path = out_base / cfg["output"]["train_file"]
    dev_path = out_base / cfg["output"]["dev_file"]
    test_path = out_base / cfg["output"]["test_file"]
    metrics_path = out_base / cfg["output"]["metrics_file"]

    cs_cfg = cfg.get("corpus_stats") or {}
    spm_rel = cs_cfg.get("spm_model")
    spm_path = (project_root / spm_rel) if spm_rel else None
    enc_bs = int(cs_cfg.get("encode_batch_size", 8192))

    corpus_stats = compute_corpus_stats_from_tsv_splits(
        train_path, dev_path, test_path, sp_model_path=spm_path, encode_batch_size=enc_bs
    )

    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {}
    metrics["corpus_stats"] = corpus_stats
    write_metrics(metrics_path, metrics)
    print(f"Updated corpus_stats in {metrics_path}")


if __name__ == "__main__":
    main()
