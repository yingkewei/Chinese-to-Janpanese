#!/usr/bin/env python3
"""
Offline supervised fine-tuning (SFT) for zh→ja on NLLB, using only *local* weights.

This repo’s ``finetune_nllb.py`` pulls from the Hub at train time; on servers without
huggingface.co access that fails. Here the model directory must already exist (you
download once via mirror on your side — see ``_print_download_help``).

What “预训练 + 微调” means in practice on 1×4090
-----------------------------------------------
- **预训练**: use a public checkpoint (e.g. NLLB-200 distilled 600M) trained on huge
  multilingual data — you are *not* expected to redo that on one GPU.
- **微调 (SFT)**: continue training on your parallel TSV (same ``train.filtered.tsv``
  layout as ``translation/train.py`` / ``finetune_nllb.py``).

Dataset
-------
- Yes: you can use the **same** parallel set you already use for ``train.py``.
- For smoke tests or faster iterations, use ``--max-train-samples`` or
  ``--train-sample-ratio`` (deterministic after ``--seed``).

Example (after model is under ``models/nllb-200-distilled-600M``):

  conda activate cnjp-py310
  cd <project_root>
  python translation/nllb_offline_sft.py --fp16 --gradient-checkpointing \\
    --max-train-samples 50000 --epochs 1 --batch-size 8 --grad-accum 2

Env: by default ``CUDA_VISIBLE_DEVICES=0`` is set if unset (avoids ``nn.DataParallel``
when multiple CUDA devices are visible — that path can segfault). Use all GPUs only
with ``accelerate launch`` and ``NLLB_SFT_ALL_GPUS=1`` plus your own device list.

Default train TSV: ``dataset_processed/raw/train.filtered.tsv`` (4 columns:
zh, ja, score, weight — only zh/ja are used).

Artifacts written to ``--output-dir`` after training:
  - ``loss_curve.png`` — training loss vs step + dev BLEU vs epoch
  - ``metrics_history.json`` — per-epoch rows (train_loss mean, dev_bleu) like ``train.py``
  - ``trainer_log.json`` — raw ``Trainer`` ``log_history`` (optional debugging)
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DEFAULT_HF = _ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_DEFAULT_HF))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_DEFAULT_HF / "hub"))
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# HF ``Trainer`` wraps ``nn.DataParallel`` when it sees multiple CUDA devices without
# distributed launch; that path emits gather/scalar warnings and can segfault on some
# setups. Default: only expose the first GPU. Multi-GPU DDP: use ``accelerate launch``
# and set ``NLLB_SFT_ALL_GPUS=1`` (and set ``CUDA_VISIBLE_DEVICES`` yourself).
if (
    os.environ.get("NLLB_SFT_ALL_GPUS", "").strip() != "1"
    and "CUDA_VISIBLE_DEVICES" not in os.environ
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sacrebleu.metrics import BLEU
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)


def _patch_accelerate_unwrap_compat() -> None:
    """
    Newer ``transformers`` calls ``Accelerator.unwrap_model(..., keep_torch_compile=…)``.
    Older ``accelerate`` raises TypeError; strip the kw for compatibility.
    Prefer: ``pip install -U 'accelerate>=0.34'`` (see requirements-finetune.txt).
    """
    try:
        from accelerate import Accelerator
    except ImportError:
        return
    try:
        sig = inspect.signature(Accelerator.unwrap_model)
    except (TypeError, ValueError):
        return
    if "keep_torch_compile" in sig.parameters:
        return
    _orig = Accelerator.unwrap_model

    def _unwrap_compat(self, model, *args, **kwargs):
        kwargs.pop("keep_torch_compile", None)
        return _orig(self, model, *args, **kwargs)

    Accelerator.unwrap_model = _unwrap_compat  # type: ignore[method-assign]


_patch_accelerate_unwrap_compat()

_DEFAULT_LOCAL_MODEL_DIR = _ROOT / "models" / "nllb-200-distilled-600M"
_NLLB_PLAIN_LANG = re.compile(r"^[a-z]{3}_[A-Z][A-Za-z0-9_]*$")


def _print_download_help(model_dir: Path) -> None:
    root = _ROOT
    print(
        "\n本地没有模型文件。请在**你能访问镜像的网络环境**下，在项目根目录执行（复制到终端自行运行）：\n",
        flush=True,
    )
    print(
        f"""  export HF_ENDPOINT=https://hf-mirror.com
  cd {root}
  mkdir -p models
  huggingface-cli download facebook/nllb-200-distilled-600M \\
    --revision refs/pr/45 \\
    --local-dir {model_dir}

说明：
  - ``refs/pr/45`` 带 ``model.safetensors``，适合 torch 2.5 + transformers 4.57（避免仅 bin 被禁载）。
  - 若本机无 ``huggingface-cli``：``pip install huggingface_hub`` 后再试。
  - 下载完成后无需 VPN；本脚本只从 ``{model_dir}`` 读本地文件。
""",
        flush=True,
    )


def _local_model_ready(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    has_config = (model_dir / "config.json").is_file()
    has_weights = (model_dir / "model.safetensors").is_file() or (model_dir / "pytorch_model.bin").is_file()
    return has_config and has_weights


def _params_on_meta(model: AutoModelForSeq2SeqLM) -> bool:
    for p in model.parameters():
        if getattr(p, "is_meta", False) or (p.device.type == "meta"):
            return True
    return False


def _load_seq2seq_local_via_checkpoint(model_dir: Path) -> AutoModelForSeq2SeqLM:
    """Load real weights from disk when ``from_pretrained`` left parameters on ``meta``."""
    from accelerate.utils.modeling import load_checkpoint_in_model

    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    prev_dev = None
    if hasattr(torch, "get_default_device") and torch.get_default_device().type == "meta":
        prev_dev = torch.get_default_device()
        torch.set_default_device("cpu")
    try:
        # transformers>=4.50: public ``from_config``; older builds had private ``_from_config``.
        _cls = AutoModelForSeq2SeqLM
        if hasattr(_cls, "_from_config"):
            model = _cls._from_config(config)
        else:
            model = _cls.from_config(config)
    finally:
        if prev_dev is not None:
            torch.set_default_device(prev_dev)
    load_checkpoint_in_model(model, str(model_dir), device_map=None, strict=False)
    model.tie_weights()
    try:
        model.generation_config = GenerationConfig.from_pretrained(str(model_dir), local_files_only=True)
    except OSError:
        pass
    return model


def load_seq2seq_from_local_dir(model_dir: Path, *, use_safetensors: bool) -> AutoModelForSeq2SeqLM:
    load_kw: dict = {"local_files_only": True, "use_safetensors": use_safetensors}
    # PyTorch 2.x can default to ``meta``; new modules then have no data until ``.to()`` — Trainer breaks.
    prev_default = None
    if hasattr(torch, "get_default_device") and torch.get_default_device().type == "meta":
        prev_default = torch.get_default_device()
        torch.set_default_device("cpu")
    try:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), **load_kw)
        except Exception as e:
            err = str(e).lower()
            if "meta" not in err:
                raise
            model = _load_seq2seq_local_via_checkpoint(model_dir)
    finally:
        if prev_default is not None:
            torch.set_default_device(prev_default)

    if _params_on_meta(model):
        del model
        model = _load_seq2seq_local_via_checkpoint(model_dir)
    return model


def repair_lang_code_to_id(tokenizer) -> None:
    cur = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(cur, dict) and len(cur) > 0:
        return
    vocab = tokenizer.get_vocab()
    built: dict[str, int] = {}
    for tok, tid in vocab.items():
        if len(tok) >= 6 and tok.startswith("__") and tok.endswith("__"):
            code = tok[2:-2]
            if code:
                built[code] = tid
    for tok, tid in vocab.items():
        if _NLLB_PLAIN_LANG.match(tok) and len(tok) <= 32:
            built.setdefault(tok, tid)
    if not built:
        return
    setattr(tokenizer, "lang_code_to_id", built)
    inv = {v: k for k, v in built.items()}
    if len(inv) == len(built):
        setattr(tokenizer, "id_to_lang", inv)


def resolve_nllb_forced_bos_token_id(tokenizer, tgt_lang: str, model) -> int:
    lcd = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(lcd, dict) and lcd.get(tgt_lang) is not None:
        return int(lcd[tgt_lang])
    enc = getattr(tokenizer, "added_tokens_encoder", None) or {}
    for key in (f"__{tgt_lang}__", tgt_lang):
        if key in enc:
            return int(enc[key])
    vocab = tokenizer.get_vocab()
    for marker in (f"__{tgt_lang}__", tgt_lang):
        if marker in vocab:
            return int(vocab[marker])
    unk = tokenizer.unk_token_id
    for marker in (f"__{tgt_lang}__", tgt_lang):
        tid = tokenizer.convert_tokens_to_ids(marker)
        if unk is None or tid != unk:
            return int(tid)
    v = getattr(model.config, "forced_bos_token_id", None)
    if v is not None:
        return int(v)
    gc = getattr(model, "generation_config", None)
    if gc is not None and getattr(gc, "forced_bos_token_id", None) is not None:
        return int(gc.forced_bos_token_id)
    raise ValueError(f"Could not resolve forced_bos for tgt_lang={tgt_lang!r}")


def build_preprocess(tokenizer, src_lang: str, tgt_lang: str, max_source_length: int, max_target_length: int):
    def _fn(batch):
        tokenizer.src_lang = src_lang
        model_inputs = tokenizer(
            batch["zh"],
            max_length=max_source_length,
            truncation=True,
        )
        tokenizer.tgt_lang = tgt_lang
        labels = tokenizer(
            text_target=batch["ja"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return _fn


def load_parallel_hf_dataset(
    path: Path,
    *,
    max_samples: int | None,
    sample_ratio: float | None,
    seed: int,
) -> Dataset:
    if not path.is_file():
        raise FileNotFoundError(path)
    # Tab file without header: zh, ja, score, weight
    ds = load_dataset(
        "csv",
        data_files=str(path),
        delimiter="\t",
        column_names=["zh", "ja", "score", "weight"],
        split="train",
    )
    ds = ds.remove_columns([c for c in ("score", "weight") if c in ds.column_names])
    ds = ds.filter(lambda x: bool(str(x.get("zh", "")).strip()) and bool(str(x.get("ja", "")).strip()))
    n = len(ds)
    if n == 0:
        return ds
    if sample_ratio is not None and 0.0 < sample_ratio < 1.0:
        k = max(1, int(n * sample_ratio))
        ds = ds.shuffle(seed=seed).select(range(min(k, n)))
    if max_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(int(max_samples), len(ds))))
    return ds


def _trainer_tokenizer_kwargs(tokenizer) -> dict:
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in sig.parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in sig.parameters:
        return {"tokenizer": tokenizer}
    return {"processing_class": tokenizer}


def _read_bleu_jsonl(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        ep = int(float(o.get("epoch", 0)))
        rows.append((ep, float(o.get("dev_bleu", 0.0))))
    return rows


def _mean_train_loss_by_epoch(log_history: list, num_epochs: int) -> dict[int, float]:
    """Map HF fractional ``epoch`` in logs to 1..num_epochs inclusive."""
    from collections import defaultdict

    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for r in log_history:
        if "loss" not in r or "epoch" not in r:
            continue
        e = float(r["epoch"])
        k = min(num_epochs, int(e - 1e-9) + 1)
        if k < 1:
            k = 1
        sums[k] += float(r["loss"])
        counts[k] += 1
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


def _train_result_to_metrics(train_result) -> dict:
    if train_result is None:
        return {}
    if hasattr(train_result, "metrics") and isinstance(train_result.metrics, dict):
        return dict(train_result.metrics)
    if isinstance(train_result, dict):
        return dict(train_result)
    out: dict = {}
    for k in ("global_step", "training_loss"):
        if hasattr(train_result, k):
            out[k] = getattr(train_result, k)
    return out


def _save_nllb_training_curves_and_metrics(
    output_dir: Path,
    *,
    num_epochs: int,
    log_history: list,
    train_result,
    bleu_jsonl: Path,
) -> tuple[Path, Path, Path]:
    """Write ``trainer_log.json``, ``metrics_history.json``, ``loss_curve.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "trainer_log.json"
    log_path.write_text(json.dumps(log_history, ensure_ascii=False, indent=2), encoding="utf-8")

    loss_by_ep = _mean_train_loss_by_epoch(log_history, num_epochs)
    bleu_rows = _read_bleu_jsonl(bleu_jsonl)
    bleu_by_ep: dict[int, float] = {}
    for ep, sc in bleu_rows:
        bleu_by_ep[ep] = sc

    summ = _train_result_to_metrics(train_result)
    if summ:
        (output_dir / "train_summary.json").write_text(
            json.dumps(summ, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    vals_all = [float(r["loss"]) for r in log_history if "loss" in r]
    mean_all = sum(vals_all) / len(vals_all) if vals_all else None
    try:
        summ_tl = float(summ["train_loss"]) if summ and summ.get("train_loss") is not None else None
    except (TypeError, ValueError):
        summ_tl = None

    history: list[dict] = []
    for ep in range(1, num_epochs + 1):
        tl = loss_by_ep.get(ep)
        if tl is None and summ_tl is not None and ep == num_epochs:
            tl = summ_tl
        if tl is None:
            tl = mean_all
        row: dict = {"epoch": ep, "dev_loss": None}
        if tl is not None:
            row["train_loss"] = round(tl, 6)
        if ep in bleu_by_ep:
            row["dev_bleu"] = round(bleu_by_ep[ep], 4)
        history.append(row)

    metrics_path = output_dir / "metrics_history.json"
    metrics_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    steps: list[int] = []
    losses: list[float] = []
    for r in log_history:
        if "loss" not in r:
            continue
        st = r.get("step")
        if st is None:
            continue
        steps.append(int(st))
        losses.append(float(r["loss"]))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7.0), constrained_layout=True)
    ax0 = axes[0]
    if steps:
        ax0.plot(steps, losses, "b-", linewidth=1.0, alpha=0.9, label="train loss (Trainer logs)")
    ax0.set_xlabel("Global step")
    ax0.set_ylabel("Loss")
    ax0.set_title("NLLB offline SFT — training loss")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")

    ax1 = axes[1]
    if bleu_rows:
        eps_u = sorted({e for e, _ in bleu_rows})
        sc_u = [bleu_by_ep[e] for e in eps_u]
        ax1.plot(eps_u, sc_u, "r^-", markersize=6, linewidth=1.0, label="dev BLEU (beam=4, n=subset)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BLEU (char)")
    ax1.set_title("Dev BLEU (epoch callback)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    curve_path = output_dir / "loss_curve.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)

    return log_path, metrics_path, curve_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline NLLB SFT for zh→ja (local model dir only)")
    ap.add_argument("--model-dir", type=Path, default=_DEFAULT_LOCAL_MODEL_DIR, help="Local snapshot (no Hub).")
    ap.add_argument(
        "--train-tsv",
        type=Path,
        default=_ROOT / "dataset_processed/raw/train.filtered.tsv",
    )
    ap.add_argument(
        "--dev-tsv",
        type=Path,
        default=_ROOT / "dataset_processed/raw/dev.filtered.tsv",
    )
    ap.add_argument("--output-dir", type=Path, default=_ROOT / "translation/checkpoints_nllb_offline")
    ap.add_argument("--src-lang", type=str, default="zho_Hans")
    ap.add_argument("--tgt-lang", type=str, default="jpn_Jpan")
    ap.add_argument("--max-source-length", type=int, default=256)
    ap.add_argument("--max-target-length", type=int, default=256)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-dev-samples", type=int, default=None)
    ap.add_argument(
        "--train-sample-ratio",
        type=float,
        default=None,
        help="If set in (0,1), shuffle then keep this fraction of train rows (after filter).",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--bleu-dev-samples", type=int, default=500)
    ap.add_argument("--gen-batch-size", type=int, default=4)
    ap.add_argument(
        "--no-safetensors",
        action="store_true",
        help="Load pytorch_model.bin (needs torch>=2.6 with recent transformers).",
    )
    ap.add_argument("--tokenizer-use-fast", action="store_true")
    args = ap.parse_args()

    if not _local_model_ready(args.model_dir):
        _print_download_help(args.model_dir)
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_fast = bool(args.tokenizer_use_fast)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=use_fast, local_files_only=True)
    repair_lang_code_to_id(tokenizer)

    model = load_seq2seq_from_local_dir(args.model_dir, use_safetensors=not args.no_safetensors)
    tokenizer.src_lang = args.src_lang
    tokenizer.tgt_lang = args.tgt_lang

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_ds = load_parallel_hf_dataset(
        args.train_tsv,
        max_samples=args.max_train_samples,
        sample_ratio=args.train_sample_ratio,
        seed=args.seed,
    )
    dev_ds = load_parallel_hf_dataset(
        args.dev_tsv,
        max_samples=args.max_dev_samples,
        sample_ratio=None,
        seed=args.seed,
    )
    if len(train_ds) == 0:
        raise RuntimeError("No training rows after load/filter.")
    if len(dev_ds) == 0:
        raise RuntimeError("No dev rows after load/filter.")

    preprocess = build_preprocess(
        tokenizer,
        args.src_lang,
        args.tgt_lang,
        args.max_source_length,
        args.max_target_length,
    )
    num_proc = min(8, max(1, os.cpu_count() or 1))
    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names, num_proc=num_proc)
    dev_tok = dev_ds.map(preprocess, batched=True, remove_columns=dev_ds.column_names, num_proc=num_proc)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    forced_bos = resolve_nllb_forced_bos_token_id(tokenizer, args.tgt_lang, model)
    use_fp16 = args.fp16 and torch.cuda.is_available()
    use_bf16 = args.bf16 and torch.cuda.is_available()
    bleu_metric = BLEU(tokenize="char", effective_order=True)
    n_bleu = min(args.bleu_dev_samples, len(dev_ds))

    def compute_metrics_fn(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        pred_ids = np.argmax(preds, axis=-1) if preds.ndim == 3 else preds
        pred_ids[pred_ids < 0] = tokenizer.pad_token_id
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        hyps = [p.strip() for p in decoded_preds]
        refs = [[r.strip()] for r in decoded_labels]
        if not hyps:
            return {"bleu": 0.0}
        return {"bleu": float(bleu_metric.corpus_score(hyps, refs).score)}

    def bleu_via_generate(trainer: Seq2SeqTrainer) -> float:
        """
        Run beam search on dev zh. Must turn off gradient checkpointing for ``generate``:
        otherwise M2M100/NLLB often hits ``checkpoint`` warnings and can emit empty strings,
        which makes sacrebleu report 0.
        """
        zh = dev_ds["zh"][:n_bleu]
        ja_ref = dev_ds["ja"][:n_bleu]
        tokenizer.src_lang = args.src_lang
        tokenizer.tgt_lang = args.tgt_lang

        m = trainer.model
        if hasattr(trainer, "accelerator") and trainer.accelerator is not None:
            try:
                m = trainer.accelerator.unwrap_model(m, keep_torch_compile=False)
            except TypeError:
                m = trainer.accelerator.unwrap_model(m)

        was_training = m.training
        was_gc = bool(getattr(m, "is_gradient_checkpointing", False))
        prev_use_cache = bool(getattr(m.config, "use_cache", True))
        if was_gc:
            m.gradient_checkpointing_disable()
        m.eval()
        m.config.use_cache = True

        device = next(m.parameters()).device
        hyps_all: list[str] = []
        bs = max(1, args.gen_batch_size)
        max_new = min(int(args.max_target_length), 512)

        def _run_generate(enc_dict: dict) -> torch.Tensor:
            enc_dict = {k: v.to(device) for k, v in enc_dict.items()}
            gkw: dict = {
                **enc_dict,
                "forced_bos_token_id": forced_bos,
                "max_new_tokens": max_new,
                "num_beams": 4,
                "do_sample": False,
            }
            if use_fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return m.generate(**gkw)
            return m.generate(**gkw)

        try:
            for i in range(0, n_bleu, bs):
                batch_zh = zh[i : i + bs]
                enc = tokenizer(
                    batch_zh,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_source_length,
                )
                with torch.inference_mode():
                    gen = _run_generate(enc)
                hyps_all.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        finally:
            m.config.use_cache = prev_use_cache
            if was_gc:
                m.gradient_checkpointing_enable()
            if was_training:
                m.train()

        if not any(h.strip() for h in hyps_all):
            print(
                "WARNING: all generated hypotheses are empty after decode; BLEU will read as 0. "
                f"forced_bos_token_id={forced_bos} tgt_lang={args.tgt_lang!r}",
                flush=True,
            )
        return float(bleu_metric.corpus_score(hyps_all, [[r] for r in ja_ref]).score)

    class EpochBleuCallback(TrainerCallback):
        def __init__(self, holder: list[Seq2SeqTrainer], log_dir: Path, n_eval: int):
            self._holder = holder
            self._log_dir = log_dir
            self._n_eval = n_eval

        def on_epoch_end(self, train_args, state, control, **kwargs):
            tr = self._holder[0]
            bleu = bleu_via_generate(tr)
            ep = int(state.epoch) if state.epoch is not None else 0
            print(f"  [epoch {ep}] dev BLEU (char, beam=4, n={self._n_eval}): {bleu:.2f}")
            self._log_dir.mkdir(parents=True, exist_ok=True)
            with (self._log_dir / "metrics_nllb_offline.jsonl").open("a", encoding="utf-8") as lf:
                lf.write(json.dumps({"epoch": float(state.epoch or 0), "dev_bleu": bleu}, ensure_ascii=False) + "\n")

    _ta_sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    _eval_kw: dict = (
        {"eval_strategy": "no"}
        if "eval_strategy" in _ta_sig.parameters
        else {"evaluation_strategy": "no"}
        if "evaluation_strategy" in _ta_sig.parameters
        else {"eval_strategy": "no"}
    )
    _dl_kw: dict = {}
    if "dataloader_num_workers" in _ta_sig.parameters:
        _dl_kw["dataloader_num_workers"] = 0

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing,
        logging_steps=50,
        save_strategy="epoch",
        **_eval_kw,
        predict_with_generate=False,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        load_best_model_at_end=False,
        seed=args.seed,
        save_total_limit=2,
        **_dl_kw,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
        **_trainer_tokenizer_kwargs(tokenizer),
    )
    holder: list[Seq2SeqTrainer] = [trainer]
    trainer.add_callback(EpochBleuCallback(holder, args.output_dir, n_bleu))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "finetune_config.json").write_text(
        json.dumps(
            {
                "model_dir": str(args.model_dir),
                "train_tsv": str(args.train_tsv),
                "dev_tsv": str(args.dev_tsv),
                "train_sample_ratio": args.train_sample_ratio,
                "max_train_samples": args.max_train_samples,
                "src_lang": args.src_lang,
                "tgt_lang": args.tgt_lang,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Train rows: {len(train_ds)} | Dev rows: {len(dev_ds)} | Model: {args.model_dir}", flush=True)
    train_result = trainer.train()
    bleu_jsonl = args.output_dir / "metrics_nllb_offline.jsonl"
    log_path, metrics_path, curve_path = _save_nllb_training_curves_and_metrics(
        args.output_dir,
        num_epochs=args.epochs,
        log_history=list(getattr(trainer.state, "log_history", []) or []),
        train_result=train_result,
        bleu_jsonl=bleu_jsonl,
    )
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Final dev BLEU (char, beam=4, n={n_bleu}): {bleu_via_generate(trainer):.2f}", flush=True)
    print(f"Trainer log -> {log_path}", flush=True)
    print(f"Metrics history -> {metrics_path}", flush=True)
    print(f"Loss curve -> {curve_path}", flush=True)


if __name__ == "__main__":
    main()
