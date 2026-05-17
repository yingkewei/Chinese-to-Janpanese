#!/usr/bin/env python3
"""
Copy pipeline TSVs into dataset_processed/, train SentencePiece on mixed zh+ja
from train split, tokenize train/dev/test, and align dataset/test/*.xlsx to the
same column layout (raw + SPM).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inspect_dataset import iter_pairs_from_xlsx  # noqa: E402

_CELL_NEWLINE_RE = re.compile(r"[\r\n]+")


def flatten_cell_text(s: str) -> str:
    """Collapse newlines inside spreadsheet/corpus cells so one record = one TSV line."""
    return _CELL_NEWLINE_RE.sub(" ", s).strip()


def parse_filtered_tsv_line(line: str) -> Tuple[str, str, float, float] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2:
        return None
    if len(parts) >= 4:
        zh, ja = parts[-4], parts[-3]
        try:
            score = float(parts[-2])
            weight = float(parts[-1])
        except ValueError:
            zh, ja = parts[-2], parts[-1]
            score, weight = 0.0, 1.0
    else:
        zh, ja = parts[0], parts[1]
        score, weight = 0.0, 1.0
    zh, ja = flatten_cell_text(zh), flatten_cell_text(ja)
    if not zh or not ja:
        return None
    return zh, ja, score, weight


def iter_pairs_tsv(path: Path) -> Iterator[Tuple[str, str, float, float]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_filtered_tsv_line(line)
            if parsed:
                yield parsed


def write_corpus_from_train(train_tsv: Path, corpus_out: Path) -> int:
    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with corpus_out.open("w", encoding="utf-8") as out:
        for zh, ja, _, _ in iter_pairs_tsv(train_tsv):
            out.write(zh + "\n")
            out.write(ja + "\n")
            n += 2
    return n


def copy_pipeline_outputs(src_dir: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.filtered.tsv", "dev.filtered.tsv", "test.filtered.tsv", "metrics.json"):
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing pipeline output: {src}")
        shutil.copy2(src, raw_dir / name)


def train_sentencepiece(
    corpus: Path,
    model_prefix: Path,
    vocab_size: int,
    input_sentence_size: int | None,
) -> None:
    import sentencepiece as spm

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(model_prefix)
    kwargs: dict = dict(
        input=str(corpus),
        model_prefix=prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="unigram",
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
    )
    if input_sentence_size is not None and input_sentence_size > 0:
        kwargs["input_sentence_size"] = input_sentence_size
    spm.SentencePieceTrainer.train(**kwargs)


def encode_line(sp, text: str) -> str:
    return " ".join(sp.encode_as_pieces(text))


def tokenize_tsv(
    in_tsv: Path,
    out_tsv: Path,
    sp,
) -> int:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with in_tsv.open("r", encoding="utf-8", errors="ignore") as fin, out_tsv.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            parsed = parse_filtered_tsv_line(line)
            if not parsed:
                continue
            zh, ja, score, weight = parsed
            fout.write(
                f"{encode_line(sp, zh)}\t{encode_line(sp, ja)}\t{score:.6f}\t{weight:.3f}\n"
            )
            n += 1
    return n


def write_eval_from_xlsx(
    xlsx_path: Path,
    raw_out: Path,
    spm_out: Path | None,
    sp,
) -> int:
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    if spm_out:
        spm_out.parent.mkdir(parents=True, exist_ok=True)
        fspm = spm_out.open("w", encoding="utf-8")
    else:
        fspm = None
    try:
        with raw_out.open("w", encoding="utf-8") as fraw:
            for zh, ja in iter_pairs_from_xlsx(xlsx_path):
                zh, ja = flatten_cell_text(zh), flatten_cell_text(ja)
                if not zh or not ja:
                    continue
                fraw.write(f"{zh}\t{ja}\t0.000000\t1.000\n")
                if fspm:
                    fspm.write(f"{encode_line(sp, zh)}\t{encode_line(sp, ja)}\t0.000000\t1.000\n")
                n += 1
    finally:
        if fspm:
            fspm.close()
    return n


def collect_test_xlsx(test_dir: Path) -> List[Path]:
    if not test_dir.is_dir():
        return []
    return sorted(test_dir.glob("*.xlsx"))


def main() -> None:
    ap = argparse.ArgumentParser(description="dataset_processed + SentencePiece + eval align")
    ap.add_argument(
        "--pipeline-out",
        type=Path,
        default=PROJECT_ROOT / "data_processing" / "output",
        help="Directory with train.filtered.tsv etc.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "dataset_processed",
        help="Output root (raw/, spm/, tokenized/, eval/).",
    )
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument(
        "--spm-input-sentence-size",
        type=int,
        default=None,
        metavar="N",
        help="If set, randomly sample N lines from the corpus when training SPM (faster on huge data).",
    )
    ap.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "dataset" / "test")
    ap.add_argument("--skip-spm-train", action="store_true", help="Reuse existing .model under spm/")
    ap.add_argument(
        "--remove-corpus-after-train",
        action="store_true",
        help="Delete spm/train_corpus_lines.txt after SPM training to save disk space.",
    )
    ap.add_argument(
        "--only-eval",
        action="store_true",
        help="Only export dataset/test xlsx using existing spm/mixed_zh_ja.model (skip copy/train/tokenize).",
    )
    args = ap.parse_args()

    out_root: Path = args.out_root.resolve()
    raw_dir = out_root / "raw"
    spm_dir = out_root / "spm"
    tok_dir = out_root / "tokenized"
    eval_dir = out_root / "eval"
    model_prefix = spm_dir / "mixed_zh_ja"
    corpus_path = spm_dir / "train_corpus_lines.txt"
    model_path = Path(str(model_prefix) + ".model")

    if args.only_eval:
        if not model_path.exists():
            raise FileNotFoundError(f"--only-eval requires {model_path}")
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))
        xlsx_files = collect_test_xlsx(args.test_dir.resolve())
        if not xlsx_files:
            print(f"No .xlsx under {args.test_dir}; nothing to do.")
            return
        for xp in xlsx_files:
            stem = xp.stem
            raw_eval = eval_dir / f"{stem}.raw.tsv"
            spm_eval = eval_dir / f"{stem}.spm.tsv"
            c = write_eval_from_xlsx(xp, raw_eval, spm_eval, sp)
            print(f"Eval aligned ({c} pairs): {raw_eval} , {spm_eval}")
        print("Done (--only-eval).")
        return

    copy_pipeline_outputs(args.pipeline_out.resolve(), raw_dir)
    print(f"Copied pipeline outputs -> {raw_dir}")

    train_tsv = raw_dir / "train.filtered.tsv"
    n_lines = write_corpus_from_train(train_tsv, corpus_path)
    print(f"Wrote SPM corpus ({n_lines} lines) -> {corpus_path}")

    if args.skip_spm_train and model_path.exists():
        print(f"Skip SPM train, using {model_path}")
    else:
        train_sentencepiece(
            corpus_path,
            model_prefix,
            args.vocab_size,
            args.spm_input_sentence_size,
        )
        print(f"Trained SentencePiece -> {model_prefix}.(model|vocab)")
        if args.remove_corpus_after_train and corpus_path.exists():
            corpus_path.unlink()
            print(f"Removed {corpus_path}")

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    for split in ("train", "dev", "test"):
        src = raw_dir / f"{split}.filtered.tsv"
        dst = tok_dir / f"{split}.spm.tsv"
        c = tokenize_tsv(src, dst, sp)
        print(f"Tokenized {split}: {c} pairs -> {dst}")

    xlsx_files = collect_test_xlsx(args.test_dir.resolve())
    if not xlsx_files:
        print(f"No .xlsx under {args.test_dir}; skip eval export.")
    else:
        for xp in xlsx_files:
            stem = xp.stem
            raw_eval = eval_dir / f"{stem}.raw.tsv"
            spm_eval = eval_dir / f"{stem}.spm.tsv"
            c = write_eval_from_xlsx(xp, raw_eval, spm_eval, sp)
            print(f"Eval aligned ({c} pairs): {raw_eval} , {spm_eval}")

    print("Done.")


if __name__ == "__main__":
    main()
