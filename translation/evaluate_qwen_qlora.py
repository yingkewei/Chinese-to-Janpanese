#!/usr/bin/env python3
"""
Evaluate a **Qwen2.5 QLoRA adapter** (directory from ``qwen25_qlora_sft.py``) on the same
parallel eval TSV convention as ``translation/evaluate_nllb.py``.

Default eval file: the sole ``*.raw.tsv`` under ``dataset_processed/eval/`` (zh \\t ja \\t …).

Writes corpus BLEU (char), a TSV ``原文 / 译文 / 模型译文``, and a metrics JSON.

Run from project root:
  python translation/evaluate_qwen_qlora.py \\
    --adapter-dir translation/checkpoints_qwen25_qlora \\
    --base-model-dir models/Qwen2.5-7B-Instruct

If ``finetune_config.json`` exists under ``--adapter-dir``, ``system_prompt`` and
``max_seq_length`` default from it.

  CUDA_VISIBLE_DEVICES=1 python translation/evaluate_qwen_qlora.py --max-samples 500
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_HF = _ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_DEFAULT_HF))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_DEFAULT_HF / "hub"))
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if os.environ.get("QWEN_EVAL_ALL_GPUS", "").strip() != "1" and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from peft import PeftModel
from sacrebleu.metrics import BLEU
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from translation.dataset import iter_zh_ja_tsv


def _default_eval_raw(root: Path) -> Path:
    d = root / "dataset_processed/eval"
    paths = sorted(d.glob("*.raw.tsv"))
    if not paths:
        raise FileNotFoundError(f"No *.raw.tsv under {d}")
    if len(paths) > 1:
        names = ", ".join(p.name for p in paths)
        raise FileNotFoundError(f"Multiple eval raw TSV in {d}: {names}. Pass --eval-raw-tsv explicitly.")
    return paths[0]


def _local_base_ready(model_dir: Path) -> bool:
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        return False
    if (model_dir / "model.safetensors").is_file() or (model_dir / "pytorch_model.bin").is_file():
        return True
    return any(model_dir.glob("model-*.safetensors"))


def _adapter_ready(adapter_dir: Path) -> bool:
    return (adapter_dir / "adapter_config.json").is_file() and (
        (adapter_dir / "adapter_model.safetensors").is_file() or (adapter_dir / "adapter_model.bin").is_file()
    )


def _load_finetune_config(adapter_dir: Path) -> dict[str, Any]:
    p = adapter_dir / "finetune_config.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _batched_lines(pairs: List[Tuple[str, str]], batch_size: int):
    for i in range(0, len(pairs), batch_size):
        yield pairs[i : i + batch_size]


def main() -> None:
    root = _ROOT
    default_adapter = root / "translation/checkpoints_qwen25_qlora"
    default_base = root / "models" / "Qwen2.5-7B-Instruct"
    ap = argparse.ArgumentParser(description="Eval Qwen2.5 QLoRA adapter on parallel eval TSV (zh→ja)")
    ap.add_argument("--adapter-dir", type=Path, default=default_adapter, help="LoRA output dir (adapter_*.safetensors).")
    ap.add_argument(
        "--base-model-dir",
        type=Path,
        default=None,
        help="Full Qwen base weights. Default: finetune_config.json model_dir or models/Qwen2.5-7B-Instruct.",
    )
    ap.add_argument(
        "--eval-raw-tsv",
        type=Path,
        default=None,
        help="Parallel eval TSV. Default: sole *.raw.tsv under dataset_processed/eval/",
    )
    ap.add_argument(
        "--output-tsv",
        type=Path,
        default=root / "translation/output/eval_predictions_qwen.tsv",
        help="TSV: 原文, 译文, 模型译文",
    )
    ap.add_argument("--output-metrics", type=Path, default=None, help="Default: output-tsv stem + .metrics.json")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1, help="Increase if VRAM allows (left-padded batch).")
    ap.add_argument("--max-seq-length", type=int, default=None, help="Prompt token budget; default from finetune_config or 512.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override system message (default: finetune_config.json or training default).",
    )
    ap.add_argument("--bf16", action="store_true", help="Use bf16 compute dtype for 4-bit base (recommended on 4090).")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--progress-every", type=int, default=500)
    args = ap.parse_args()

    cfg = _load_finetune_config(args.adapter_dir)
    base_arg = args.base_model_dir
    if base_arg is None:
        md = cfg.get("model_dir")
        base_arg = Path(md) if md else default_base
    base_arg = Path(base_arg)

    if not _adapter_ready(args.adapter_dir):
        raise FileNotFoundError(
            f"Missing LoRA adapter under {args.adapter_dir} (need adapter_config.json + adapter_model.safetensors)."
        )
    if not _local_base_ready(base_arg):
        raise FileNotFoundError(
            f"Missing base model under {base_arg}. Download Qwen2.5-7B-Instruct into models/ (see qwen25_qlora_sft.py --help)."
        )

    system_prompt = args.system_prompt
    if system_prompt is None:
        system_prompt = cfg.get(
            "system_prompt",
            "你是专业的中文到日文翻译助手，请将用户给出的中文翻译成自然、准确的日语，只输出译文。",
        )
    max_seq = args.max_seq_length
    if max_seq is None:
        max_seq = int(cfg.get("max_seq_length", 512))

    eval_tsv = args.eval_raw_tsv if args.eval_raw_tsv is not None else _default_eval_raw(root)
    if not eval_tsv.is_file():
        raise FileNotFoundError(f"Eval TSV not found: {eval_tsv}")

    tok_path = args.adapter_dir if (args.adapter_dir / "tokenizer_config.json").is_file() else base_arg
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=True, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_bf16 = bool(args.bf16) or (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = bool(args.fp16) and not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(base_arg),
        quantization_config=bnb_config,
        device_map=args.device_map if torch.cuda.is_available() else None,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter_dir), local_files_only=True)
    model.eval()
    model.config.use_cache = True
    model_device = next(model.parameters()).device

    def build_prompt_text(zh: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(zh).strip()},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    on_cuda = model_device.type == "cuda"
    autocast_cm = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and on_cuda
        else torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_fp16 and on_cuda
        else nullcontext()
    )

    for batch in _batched_lines(pairs, bs):
        zh_batch = [p[0] for p in batch]
        ja_batch = [p[1] for p in batch]
        prompts = [build_prompt_text(z) for z in zh_batch]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq,
        )
        enc = {k: v.to(model_device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]
        with torch.inference_mode(), autocast_cm:
            out = model.generate(
                **enc,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        gen_part = out[:, input_len:]
        decoded = tokenizer.batch_decode(gen_part, skip_special_tokens=True)
        for zh, ja_ref, hyp in zip(zh_batch, ja_batch, decoded):
            h = hyp.strip()
            hyps.append(h)
            rows.append((zh, ja_ref, h))
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
        "adapter_dir": str(args.adapter_dir.resolve()),
        "base_model_dir": str(base_arg.resolve()),
        "decode_seconds": round(elapsed, 3),
        "max_seq_length": max_seq,
        "max_new_tokens": int(args.max_new_tokens),
        "batch_size": bs,
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
