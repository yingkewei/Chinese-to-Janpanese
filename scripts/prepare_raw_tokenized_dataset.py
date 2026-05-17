#!/usr/bin/env python3
"""
Copy data_processing_raw/output TSVs into dataset_processed_raw/raw/ and
tokenize with the *same* SentencePiece model as the cleaned pipeline (no SPM retrain).

Layout mirrors dataset_processed/: raw/, spm/, tokenized/
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_prepare_tokenized():
    path = PROJECT_ROOT / "scripts" / "prepare_tokenized_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_tokenized_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Copy raw-baseline TSVs + tokenize with existing mixed_zh_ja.model"
    )
    ap.add_argument(
        "--src-dir",
        type=Path,
        default=PROJECT_ROOT / "data_processing_raw" / "output",
        help="Directory with train.raw.tsv, dev.raw.tsv, test.raw.tsv",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "dataset_processed_raw",
        help="Output root (raw/, spm/, tokenized/)",
    )
    ap.add_argument(
        "--spm-model",
        type=Path,
        default=PROJECT_ROOT / "dataset_processed" / "spm" / "mixed_zh_ja.model",
        help="Existing SentencePiece model (same vocab as cleaned runs)",
    )
    args = ap.parse_args()

    src = args.src_dir.resolve()
    out_root = args.out_root.resolve()
    spm_model = args.spm_model.resolve()

    for name in ("train.raw.tsv", "dev.raw.tsv", "test.raw.tsv"):
        if not (src / name).is_file():
            raise FileNotFoundError(f"Missing {src / name}")

    if not spm_model.is_file():
        raise FileNotFoundError(f"SPM model not found: {spm_model}")

    ptd = _load_prepare_tokenized()
    tokenize_tsv = ptd.tokenize_tsv

    raw_dir = out_root / "raw"
    spm_dir = out_root / "spm"
    tok_dir = out_root / "tokenized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    spm_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)

    mapping = (
        ("train.raw.tsv", "train.tsv"),
        ("dev.raw.tsv", "dev.tsv"),
        ("test.raw.tsv", "test.tsv"),
    )
    for src_name, dst_name in mapping:
        shutil.copy2(src / src_name, raw_dir / dst_name)
        print(f"Copied {src_name} -> {raw_dir / dst_name}")

    metrics_src = src / "metrics.json"
    if metrics_src.is_file():
        shutil.copy2(metrics_src, raw_dir / "metrics.json")
        print(f"Copied metrics.json -> {raw_dir / 'metrics.json'}")

    vocab_src = spm_model.with_suffix(".vocab")
    shutil.copy2(spm_model, spm_dir / spm_model.name)
    if vocab_src.is_file():
        shutil.copy2(vocab_src, spm_dir / vocab_src.name)
    local_model = spm_dir / spm_model.name
    print(f"Copied SPM -> {local_model}")

    import sentencepiece as spm

    processor = spm.SentencePieceProcessor()
    processor.load(str(local_model))

    split_names = ("train", "dev", "test")
    for split in split_names:
        in_tsv = raw_dir / f"{split}.tsv"
        out_tsv = tok_dir / f"{split}.spm.tsv"
        n = tokenize_tsv(in_tsv, out_tsv, processor)
        print(f"Tokenized {split}: {n} pairs -> {out_tsv}")

    print("Done.")
    print(f"Train with: python translation/train.py --train-tsv {tok_dir / 'train.spm.tsv'} "
          f"--dev-tsv {tok_dir / 'dev.spm.tsv'} --spm-model {local_model}")


if __name__ == "__main__":
    main()
