from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from stages import (
    compute_corpus_stats_splits,
    iter_raw_pairs,
    maybe_limit_pairs,
    split_data,
    stage0_pre_audit,
    stage1_basic_clean,
    stage2_language_normalize,
    stage3_length_alignment,
    stage4_semantic_filter,
    stage5_dedup_reweight,
    write_metrics,
    write_tsv,
)


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def count_stage(metrics: Dict[str, Any], name: str, count: int) -> None:
    metrics["stages"][name] = count


def _is_local_model_path(model_name: str) -> bool:
    candidate = Path(model_name).expanduser()
    return candidate.exists()


def _is_model_cached_in_default_hf_home(model_name: str) -> bool:
    if _is_local_model_path(model_name):
        return True
    if "/" not in model_name:
        return False

    hf_default_hub = Path.home() / ".cache" / "huggingface" / "hub"
    repo_cache = hf_default_hub / f"models--{model_name.replace('/', '--')}"
    snapshots = repo_cache / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def _set_project_model_cache_if_needed(project_root: Path, model_name: str) -> None:
    # Respect pre-existing environment settings from user/system.
    cache_env_keys = (
        "SENTENCE_TRANSFORMERS_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    )
    if any(os.environ.get(key) for key in cache_env_keys):
        return

    # Keep using global cache for already-downloaded models.
    if _is_model_cached_in_default_hf_home(model_name):
        return

    project_cache_root = project_root / ".cache"
    hf_home = project_cache_root / "huggingface"
    st_home = project_cache_root / "sentence_transformers"
    hf_hub = hf_home / "hub"
    hf_home.mkdir(parents=True, exist_ok=True)
    st_home.mkdir(parents=True, exist_ok=True)
    hf_hub.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_hub)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(st_home)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone zh-ja data processing pipeline")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional override for input.max_pairs (useful for quick smoke tests)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = args.config.resolve().parent.parent
    stage4_cfg = cfg.get("stage4_semantic", {}) or {}
    stage5_cfg = cfg.get("stage5_dedup_reweight", {}) or {}
    stage4_enabled = bool(stage4_cfg.get("enabled", True))
    stage5_enabled = bool(stage5_cfg.get("enabled", True))

    model_name = str(stage4_cfg.get("model_name", "")).strip()
    if stage4_enabled and model_name:
        _set_project_model_cache_if_needed(project_root, model_name)

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

    # Read raw pairs with progress report (time per N pairs).
    raw_pairs = []
    t0 = time.perf_counter()
    t_last = t0
    for i, pair in enumerate(maybe_limit_pairs(iter_raw_pairs(raw_file, delimiter), max_pairs=max_pairs), start=1):
        raw_pairs.append(pair)
        if progress_every > 0 and i % progress_every == 0:
            now = time.perf_counter()
            print(
                f"[read] {i:,} pairs | last {progress_every:,}: {now - t_last:.2f}s | total: {now - t0:.2f}s"
            )
            t_last = now

    metrics: Dict[str, Any] = {"stages": {}}
    metrics["pre_audit"] = stage0_pre_audit(raw_pairs)
    count_stage(metrics, "stage0_raw_pairs", len(raw_pairs))

    s1 = stage1_basic_clean(raw_pairs, cfg["stage1_basic_clean"])
    count_stage(metrics, "stage1_basic_clean", len(s1))

    s2 = stage2_language_normalize(s1, cfg["stage2_language_normalize"])
    count_stage(metrics, "stage2_language_normalize", len(s2))

    s3 = stage3_length_alignment(s2, cfg["stage3_length_alignment"])
    count_stage(metrics, "stage3_length_alignment", len(s3))

    if stage4_enabled:
        s4 = stage4_semantic_filter(s3, stage4_cfg)
    else:
        s4 = list(s3)
    count_stage(metrics, "stage4_semantic_filter", len(s4))
    metrics.setdefault("flags", {})["stage4_semantic_enabled"] = stage4_enabled

    if stage5_enabled:
        s5 = stage5_dedup_reweight(s4, stage5_cfg)
    else:
        s5 = list(s4)
    count_stage(metrics, "stage5_dedup_reweight", len(s5))
    metrics.setdefault("flags", {})["stage5_dedup_reweight_enabled"] = stage5_enabled

    train, dev, test = split_data(
        s5,
        dev_ratio=float(cfg["split"]["dev_ratio"]),
        test_ratio=float(cfg["split"]["test_ratio"]),
        seed=int(cfg["split"]["seed"]),
    )
    count_stage(metrics, "final_train", len(train))
    count_stage(metrics, "final_dev", len(dev))
    count_stage(metrics, "final_test", len(test))

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

    print("Pipeline done.")
    print(f"Train: {train_path} ({len(train)})")
    print(f"Dev:   {dev_path} ({len(dev)})")
    print(f"Test:  {test_path} ({len(test)})")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
