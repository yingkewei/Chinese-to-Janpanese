#!/usr/bin/env python3
"""
Evaluate a **fine-tuned NLLB** checkpoint directory (Hugging Face layout: ``config.json``,
``model.safetensors`` or ``pytorch_model.bin``, tokenizer files) on the same parallel
eval TSV convention as ``translation/evaluate_corpus.py`` (Transformer).

Default eval file: the sole ``*.raw.tsv`` under ``dataset_processed/eval/``.
Writes corpus BLEU (char), a TSV ``原文 / 译文 / 模型译文``, and a metrics JSON.

Run from project root:
  python translation/evaluate_nllb.py --model-dir translation/checkpoints_nllb_offline
  python translation/evaluate_nllb.py --model-dir ... --num-beams 4 --max-samples 500

Default decoding is **greedy** (``--num-beams 1``), aligned with ``evaluate_corpus.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if (
    os.environ.get("NLLB_SFT_ALL_GPUS", "").strip() != "1"
    and "CUDA_VISIBLE_DEVICES" not in os.environ
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer

from translation.dataset import iter_zh_ja_tsv
from translation.nllb_offline_sft import (
    _local_model_ready,
    load_seq2seq_from_local_dir,
    repair_lang_code_to_id,
    resolve_nllb_forced_bos_token_id,
)


def _default_eval_raw(root: Path) -> Path:
    d = root / "dataset_processed/eval"
    paths = sorted(d.glob("*.raw.tsv"))
    if not paths:
        raise FileNotFoundError(f"No *.raw.tsv under {d}")
    if len(paths) > 1:
        names = ", ".join(p.name for p in paths)
        raise FileNotFoundError(f"Multiple eval raw TSV in {d}: {names}. Pass --eval-raw-tsv explicitly.")
    return paths[0]


def _batched_lines(pairs: List[Tuple[str, str]], batch_size: int):
    for i in range(0, len(pairs), batch_size):
        yield pairs[i : i + batch_size]


def main() -> None:
    root = _ROOT
    ap = argparse.ArgumentParser(description="Eval fine-tuned NLLB on parallel eval TSV (same layout as evaluate_corpus.py)")
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=root / "translation/checkpoints_nllb_offline",
        help="HF checkpoint folder (fine-tuned output or full snapshot).",
    )
    ap.add_argument(
        "--eval-raw-tsv",
        type=Path,
        default=None,
        help="Parallel eval TSV (zh \\t ja ...). Default: sole *.raw.tsv under dataset_processed/eval/",
    )
    ap.add_argument(
        "--output-tsv",
        type=Path,
        default=root / "translation/output/eval_predictions_nllb.tsv",
        help="TSV with columns 原文, 译文, 模型译文",
    )
    ap.add_argument(
        "--output-metrics",
        type=Path,
        default=None,
        help="Metrics JSON path. Default: same stem as --output-tsv with .metrics.json",
    )
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-beams", type=int, default=1, help="1 = greedy (default, like Transformer eval).")
    ap.add_argument("--src-lang", type=str, default="zho_Hans")
    ap.add_argument("--tgt-lang", type=str, default="jpn_Jpan")
    ap.add_argument("--max-source-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-safetensors", action="store_true")
    ap.add_argument("--tokenizer-use-fast", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--progress-every", type=int, default=500, help="Print progress every N lines (0=off)")
    args = ap.parse_args()

    eval_tsv = args.eval_raw_tsv if args.eval_raw_tsv is not None else _default_eval_raw(root)
    if not eval_tsv.is_file():
        raise FileNotFoundError(f"Eval TSV not found: {eval_tsv}")
    if not _local_model_ready(args.model_dir):
        raise FileNotFoundError(
            f"Missing NLLB weights under {args.model_dir} (need config.json + model.safetensors or pytorch_model.bin)."
        )

    device = torch.device(args.device)
    use_fast = bool(args.tokenizer_use_fast)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=use_fast, local_files_only=True)
    repair_lang_code_to_id(tokenizer)
    model = load_seq2seq_from_local_dir(args.model_dir, use_safetensors=not args.no_safetensors)
    model.to(device)
    tokenizer.src_lang = args.src_lang
    tokenizer.tgt_lang = args.tgt_lang
    forced_bos = resolve_nllb_forced_bos_token_id(tokenizer, args.tgt_lang, model)
    model.eval()
    model.config.use_cache = True
    use_fp16 = bool(args.fp16) and device.type == "cuda"

    pairs: List[Tuple[str, str]] = []
    for zh, ja in iter_zh_ja_tsv(eval_tsv, max_samples=args.max_samples):
        pairs.append((zh, ja))
    if not pairs:
        raise RuntimeError("No lines loaded from eval TSV.")

    hyps: List[str] = []
    rows: List[Tuple[str, str, str]] = []
    t0 = time.perf_counter()
    n_done = 0
    bs = max(1, args.batch_size)
    max_new = min(int(args.max_new_tokens), 512)

    for batch in _batched_lines(pairs, bs):
        zh_batch = [p[0] for p in batch]
        ja_batch = [p[1] for p in batch]
        enc = tokenizer(
            zh_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_source_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        gkw = {
            **enc,
            "forced_bos_token_id": forced_bos,
            "max_new_tokens": max_new,
            "num_beams": max(1, args.num_beams),
            "do_sample": False,
        }
        with torch.inference_mode():
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    gen = model.generate(**gkw)
            else:
                gen = model.generate(**gkw)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for zh, ja_ref, hyp in zip(zh_batch, ja_batch, decoded):
            hyps.append(hyp.strip())
            rows.append((zh, ja_ref, hyp.strip()))
        n_done += len(batch)
        pe = args.progress_every
        if pe and n_done % pe == 0:
            print(f"  decoded {n_done} / {len(pairs)} lines ...", flush=True)

    refs = [p[1] for p in pairs]
    bleu = BLEU(tokenize="char", effective_order=True)
    bleu_score = float(bleu.corpus_score(hyps, [refs]).score)
    elapsed = time.perf_counter() - t0

    metrics = {
        "corpus_bleu_char": bleu_score,
        "n_lines": len(hyps),
        "eval_raw_tsv": str(eval_tsv.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "decode_seconds": round(elapsed, 3),
        "num_beams": max(1, args.num_beams),
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
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
    print(f"Wrote predictions TSV -> {args.output_tsv}", flush=True)
    print(f"Wrote metrics JSON -> {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
