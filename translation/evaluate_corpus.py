#!/usr/bin/env python3
"""
Greedy-decode a checkpoint on parallel eval TSV (zh \\t ja ...), print corpus BLEU,
and write a TSV with 原文 / 译文 / 模型译文.

Default eval source: dataset_processed/eval/*.raw.tsv (exactly one file expected).

Run from project root:
  python translation/evaluate_corpus.py --checkpoint translation/checkpoints/last.pt
  python translation/evaluate_corpus.py --eval-raw-tsv path/to/eval.raw.tsv --max-samples 500
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import sentencepiece as spm
import torch
from sacrebleu.metrics import BLEU

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from translation.dataset import iter_zh_ja_tsv
from translation.inference import greedy_translate
from translation.model import TransformerNMT


def _default_eval_raw(root: Path) -> Path:
    d = root / "dataset_processed/eval"
    paths = sorted(d.glob("*.raw.tsv"))
    if not paths:
        raise FileNotFoundError(f"No *.raw.tsv under {d}")
    if len(paths) > 1:
        names = ", ".join(p.name for p in paths)
        raise FileNotFoundError(f"Multiple eval raw TSV in {d}: {names}. Pass --eval-raw-tsv explicitly.")
    return paths[0]


def load_model_and_sp(
    checkpoint: Path,
    spm_model: Path,
    device: torch.device,
    max_src_len: int,
    max_tgt_len: int,
) -> Tuple[TransformerNMT, spm.SentencePieceProcessor, int]:
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("args", {})
    pad_idx = int(ckpt.get("pad_idx", cfg.get("pad_idx", 32000)))
    vocab_size = int(ckpt.get("vocab_size", cfg.get("vocab_size", pad_idx + 1)))

    model = TransformerNMT(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        d_model=int(cfg.get("d_model", 512)),
        nhead=int(cfg.get("nhead", 8)),
        num_encoder_layers=int(cfg.get("num_encoder_layers", 4)),
        num_decoder_layers=int(cfg.get("num_decoder_layers", 4)),
        dim_feedforward=int(cfg.get("dim_feedforward", 2048)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_len=max(max_src_len, max_tgt_len) + 32,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_model))
    return model, sp, pad_idx


def main() -> None:
    root = _ROOT
    ap = argparse.ArgumentParser(description="Eval zh->ja on parallel TSV: BLEU + output triples")
    ap.add_argument("--checkpoint", type=Path, default=root / "translation/checkpoints/best.pt")
    ap.add_argument("--spm-model", type=Path, default=root / "dataset_processed/spm/mixed_zh_ja.model")
    ap.add_argument(
        "--eval-raw-tsv",
        type=Path,
        default=None,
        help="Parallel eval TSV (zh \\t ja ...). Default: sole *.raw.tsv under dataset_processed/eval/",
    )
    ap.add_argument(
        "--output-tsv",
        type=Path,
        default=root / "translation/output/eval_predictions.tsv",
        help="TSV with columns 原文, 译文, 模型译文",
    )
    ap.add_argument(
        "--output-metrics",
        type=Path,
        default=None,
        help="Write metrics JSON here. Default: same stem as --output-tsv with .metrics.json",
    )
    ap.add_argument("--max-samples", type=int, default=None, help="Cap number of lines (debug / smoke test)")
    ap.add_argument("--max-src-len", type=int, default=128)
    ap.add_argument("--max-tgt-len", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--progress-every", type=int, default=500, help="Print progress every N source lines (0=off)")
    args = ap.parse_args()

    eval_tsv = args.eval_raw_tsv if args.eval_raw_tsv is not None else _default_eval_raw(root)
    if not eval_tsv.exists():
        raise FileNotFoundError(f"Eval TSV not found: {eval_tsv}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = torch.device(args.device)
    model, sp, pad_idx = load_model_and_sp(
        args.checkpoint, args.spm_model, device, args.max_src_len, args.max_tgt_len
    )

    hyps: List[str] = []
    refs: List[str] = []
    rows: List[Tuple[str, str, str]] = []

    t0 = time.perf_counter()
    n = 0
    for zh, ja_ref in iter_zh_ja_tsv(eval_tsv, max_samples=args.max_samples):
        hyp = greedy_translate(
            model, sp, zh, device, args.max_src_len, args.max_tgt_len, pad_idx
        )
        hyps.append(hyp)
        refs.append(ja_ref)
        rows.append((zh, ja_ref, hyp))
        n += 1
        pe = args.progress_every
        if pe and n % pe == 0:
            print(f"  decoded {n} lines ...", flush=True)

    if not hyps:
        raise RuntimeError("No lines decoded; check eval TSV format.")

    bleu = BLEU(tokenize="char", effective_order=True)
    bleu_score = float(bleu.corpus_score(hyps, [refs]).score)
    elapsed = time.perf_counter() - t0

    metrics = {
        "corpus_bleu_char": bleu_score,
        "n_lines": len(hyps),
        "eval_raw_tsv": str(eval_tsv.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "decode_seconds": round(elapsed, 3),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_metrics
    if metrics_path is None:
        metrics_path = args.output_tsv.with_suffix(".metrics.json")

    with args.output_tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["原文", "译文", "模型译文"])
        w.writerows(rows)

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote predictions TSV -> {args.output_tsv}")
    print(f"Wrote metrics JSON -> {metrics_path}")


if __name__ == "__main__":
    main()
