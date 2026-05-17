from __future__ import annotations

import json
import random
import re
import unicodedata
import warnings
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from statistics import mean
import time
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
HTML_TAG_RE = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")
HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]{2,12}|#\d{1,7}|#x[0-9a-fA-F]{1,6});")
JS_HINT_RE = re.compile(
    r"(?:\bfunction\b|\breturn\b|\bvar\b|\blet\b|\bconst\b|=>|\bdocument\b|\bwindow\b|console\.log|\beval\s*\()",
    re.IGNORECASE,
)
CSS_HINT_RE = re.compile(
    r"(?:\{[^{}]{0,200}:\s*[^{}]{0,200};[^{}]{0,200}\}|\bcolor\s*:|\bfont(-size)?\s*:|\bmargin\s*:|\bpadding\s*:)",
    re.IGNORECASE,
)
NUM_RE = re.compile(r"\d+")
SPACE_RE = re.compile(r"\s+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
JP_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]")

# Broad punctuation set shared by zh/ja.
# Note: keep this intentionally simple/fast; we only need a rough count.
PUNCT_RE = re.compile(r"[，。！？；：、,.!?;:()（）【】\[\]\"'“”‘’「」『』《》〈〉…—\-]")


@dataclass
class Pair:
    zh: str
    ja: str
    score: float = 0.0
    weight: float = 1.0


def parse_pair_from_line(line: str, delimiter: str = "\t") -> Tuple[str, str] | None:
    parts = line.rstrip("\n").split(delimiter)
    if len(parts) >= 4:
        zh = parts[-2].strip()
        ja = parts[-1].strip()
    elif len(parts) == 2:
        zh = parts[0].strip()
        ja = parts[1].strip()
    else:
        return None
    if not zh or not ja:
        return None
    return zh, ja


def iter_raw_pairs(raw_file: Path, delimiter: str) -> Iterator[Pair]:
    with raw_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_pair_from_line(line, delimiter=delimiter)
            if parsed:
                yield Pair(zh=parsed[0], ja=parsed[1])


def maybe_limit_pairs(pairs: Iterable[Pair], max_pairs: int | None) -> Iterator[Pair]:
    if max_pairs is None:
        yield from pairs
        return
    yield from islice(pairs, max_pairs)


def ratio(regex: re.Pattern[str], text: str) -> float:
    if not text:
        return 0.0
    return len(regex.findall(text)) / max(len(text), 1)


def normalize_text(text: str, use_nfkc: bool, collapse_spaces: bool) -> str:
    out = text.strip()
    if use_nfkc:
        out = unicodedata.normalize("NFKC", out)
    if collapse_spaces:
        out = SPACE_RE.sub(" ", out)
    return out


def numbers(text: str) -> Tuple[str, ...]:
    return tuple(NUM_RE.findall(text))


def kata_to_hira(text: str) -> str:
    # Katakana block: U+30A1..U+30F6 maps to Hiragana by -0x60 for the core syllabary.
    out = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def hira_to_kata(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            out.append(chr(code + 0x60))
        else:
            out.append(ch)
    return "".join(out)


def has_repeated_run(text: str, run_len: int) -> bool:
    if run_len <= 1:
        return False
    return re.search(rf"(.)\1{{{run_len - 1},}}", text) is not None


def weird_char_ratio(text: str) -> float:
    """
    Heuristic ratio for "garbage" characters.
    Counts control/unassigned/private-use/surrogate and replacement char as weird.
    """
    if not text:
        return 0.0
    weird = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if ch == "\uFFFD":
            weird += 1
            continue
        cat = unicodedata.category(ch)  # e.g. "Cn", "Cc", "Co"
        if cat in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            weird += 1
    if total == 0:
        return 0.0
    return weird / total


def stage0_pre_audit(pairs: Sequence[Pair]) -> Dict[str, Any]:
    if not pairs:
        return {"raw_pairs": 0}

    zh_lens = [len(x.zh) for x in pairs]
    ja_lens = [len(x.ja) for x in pairs]
    dup_count = len(pairs) - len({(x.zh, x.ja) for x in pairs})

    return {
        "raw_pairs": len(pairs),
        "raw_duplicate_pairs": dup_count,
        "zh_len_mean": round(mean(zh_lens), 3),
        "ja_len_mean": round(mean(ja_lens), 3),
        "zh_len_p95": sorted(zh_lens)[int(0.95 * (len(zh_lens) - 1))],
        "ja_len_p95": sorted(ja_lens)[int(0.95 * (len(ja_lens) - 1))],
    }


def stage1_basic_clean(pairs: Iterable[Pair], cfg: Dict[str, Any]) -> List[Pair]:
    out: List[Pair] = []
    min_char = int(cfg["min_char"])
    max_char = int(cfg["max_char"])
    max_url_ratio = float(cfg["max_url_ratio"])
    drop_any_url = bool(cfg.get("drop_if_any_url", False))
    drop_if_same_text = bool(cfg["drop_if_same_text"])
    drop_html = bool(cfg.get("drop_if_html", True))
    drop_email = bool(cfg.get("drop_if_email", True))
    drop_js_css = bool(cfg.get("drop_if_js_css", True))
    max_weird_ratio = cfg.get("max_weird_char_ratio", None)
    max_weird_ratio = float(max_weird_ratio) if max_weird_ratio is not None else None
    repeated_run_len = int(cfg.get("max_repeated_char_run", 0))

    for item in pairs:
        if not (min_char <= len(item.zh) <= max_char and min_char <= len(item.ja) <= max_char):
            continue
        if drop_if_same_text and item.zh == item.ja:
            continue

        if drop_any_url and (URL_RE.search(item.zh) or URL_RE.search(item.ja)):
            continue

        zh_url_ratio = len(URL_RE.findall(item.zh)) / max(len(item.zh), 1)
        ja_url_ratio = len(URL_RE.findall(item.ja)) / max(len(item.ja), 1)
        if zh_url_ratio > max_url_ratio or ja_url_ratio > max_url_ratio:
            continue

        if drop_email and (EMAIL_RE.search(item.zh) or EMAIL_RE.search(item.ja)):
            continue

        if drop_html and (
            HTML_TAG_RE.search(item.zh)
            or HTML_TAG_RE.search(item.ja)
            or HTML_ENTITY_RE.search(item.zh)
            or HTML_ENTITY_RE.search(item.ja)
        ):
            continue

        if drop_js_css and (
            JS_HINT_RE.search(item.zh)
            or JS_HINT_RE.search(item.ja)
            or CSS_HINT_RE.search(item.zh)
            or CSS_HINT_RE.search(item.ja)
        ):
            continue

        if max_weird_ratio is not None:
            if weird_char_ratio(item.zh) > max_weird_ratio or weird_char_ratio(item.ja) > max_weird_ratio:
                continue

        if repeated_run_len > 0:
            if has_repeated_run(item.zh, repeated_run_len) or has_repeated_run(item.ja, repeated_run_len):
                continue

        out.append(item)
    return out


def stage2_language_normalize(pairs: Iterable[Pair], cfg: Dict[str, Any]) -> List[Pair]:
    out: List[Pair] = []
    zh_ratio_min = float(cfg["require_zh_cjk_ratio"])
    ja_ratio_min = float(cfg["require_ja_jp_ratio"])
    use_nfkc = bool(cfg["use_nfkc"])
    collapse_spaces = bool(cfg["collapse_spaces"])
    zh_opencc = str(cfg.get("zh_opencc", "")).strip()
    ja_kana = str(cfg.get("ja_kana_normalize", "none")).strip().lower()
    if ja_kana not in {"none", "kata_to_hira", "hira_to_kata"}:
        raise ValueError("stage2_language_normalize.ja_kana_normalize must be one of: none, kata_to_hira, hira_to_kata")

    for item in pairs:
        zh = normalize_text(item.zh, use_nfkc=use_nfkc, collapse_spaces=collapse_spaces)
        ja = normalize_text(item.ja, use_nfkc=use_nfkc, collapse_spaces=collapse_spaces)
        if zh_opencc:
            try:
                from opencc import OpenCC  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Stage 2 zh_opencc requires OpenCC. Install with: pip install opencc-python-reimplemented"
                ) from exc
            zh = OpenCC(zh_opencc).convert(zh)

        if ja_kana == "kata_to_hira":
            ja = kata_to_hira(ja)
        elif ja_kana == "hira_to_kata":
            ja = hira_to_kata(ja)

        if ratio(CJK_RE, zh) < zh_ratio_min:
            continue
        if ratio(JP_RE, ja) < ja_ratio_min:
            continue
        out.append(Pair(zh=zh, ja=ja))
    return out


def stage3_length_alignment(pairs: Iterable[Pair], cfg: Dict[str, Any]) -> List[Pair]:
    out: List[Pair] = []
    min_len = int(cfg["min_len"])
    max_len = int(cfg["max_len"])
    min_ratio = float(cfg["min_ratio"])
    max_ratio = float(cfg["max_ratio"])
    check_numbers = bool(cfg["enforce_number_consistency"])
    check_punct = bool(cfg.get("enforce_punctuation_consistency", False))
    punct_mismatch_max = cfg.get("punct_mismatch_max", None)
    punct_mismatch_max = int(punct_mismatch_max) if punct_mismatch_max is not None else None

    for item in pairs:
        lz = len(item.zh)
        lj = len(item.ja)
        if not (min_len <= lz <= max_len and min_len <= lj <= max_len):
            continue
        r = lz / max(lj, 1)
        if r < min_ratio or r > max_ratio:
            continue
        if check_numbers and numbers(item.zh) != numbers(item.ja):
            continue
        if check_punct and punct_mismatch_max is not None:
            zh_punct = len(PUNCT_RE.findall(item.zh))
            ja_punct = len(PUNCT_RE.findall(item.ja))
            if abs(zh_punct - ja_punct) > punct_mismatch_max:
                continue
        out.append(item)
    return out


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _l2(v: Sequence[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = _l2(a)
    nb = _l2(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


def _format_for_model(texts: Sequence[str], model_style: str, is_source: bool) -> List[str]:
    if model_style == "e5":
        prefix = "query: " if is_source else "passage: "
        return [prefix + t for t in texts]
    return list(texts)


def _batch_iter(items: Sequence[Pair], batch_size: int) -> Iterator[Sequence[Pair]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def stage4_semantic_filter(pairs: Iterable[Pair], cfg: Dict[str, Any]) -> List[Pair]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Stage 4 requires sentence-transformers. Install with: pip install sentence-transformers"
        ) from exc

    model_name = str(cfg["model_name"])
    model_style = str(cfg.get("model_style", "e5")).lower()
    device = str(cfg.get("device", "auto")).lower()
    batch_size = int(cfg.get("batch_size", 128))
    normalize_embeddings = bool(cfg.get("normalize_embeddings", True))
    progress_every = int(cfg.get("progress_every", 100000))
    use_multi_gpu = bool(cfg.get("use_multi_gpu", False))
    gpu_ids = cfg.get("gpu_ids", [0, 1])

    high_threshold = float(cfg["high_threshold"])
    low_threshold = float(cfg["low_threshold"])
    mid_min_char = int(cfg.get("mid_min_char", 8))
    mid_require_number_match = bool(cfg.get("mid_require_number_match", True))
    mid_max_len_ratio_delta = float(cfg.get("mid_max_len_ratio_delta", 0.7))
    mid_punct_mismatch_max = int(cfg.get("mid_punct_mismatch_max", 6))

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    local_files_only = bool(cfg.get("local_files_only", False))
    trust_remote_code = bool(cfg.get("trust_remote_code", False))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDA initialization: The NVIDIA driver on your system is too old.*")
        try:
            model = SentenceTransformer(
                model_name,
                device=device,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Stage 4 embedding model. "
                "If network is restricted, pre-download the model or set stage4_semantic.local_files_only=true "
                "and point model_name to a local path."
            ) from exc
    cached_pairs = list(pairs)
    out: List[Pair] = []
    t0 = time.perf_counter()
    t_last = t0
    seen = 0

    for batch in _batch_iter(cached_pairs, batch_size):
        zh_texts = [x.zh for x in batch]
        ja_texts = [x.ja for x in batch]
        zh_inputs = _format_for_model(zh_texts, model_style=model_style, is_source=True)
        ja_inputs = _format_for_model(ja_texts, model_style=model_style, is_source=False)

        if use_multi_gpu and str(device).startswith("cuda"):
            import torch

            if torch.cuda.device_count() >= 2:
                target_devices = [f"cuda:{int(i)}" for i in gpu_ids]
                pool = model.start_multi_process_pool(target_devices=target_devices)
                try:
                    zh_emb = model.encode_multi_process(
                        zh_inputs,
                        pool,
                        batch_size=batch_size,
                        normalize_embeddings=normalize_embeddings,
                    )
                    ja_emb = model.encode_multi_process(
                        ja_inputs,
                        pool,
                        batch_size=batch_size,
                        normalize_embeddings=normalize_embeddings,
                    )
                finally:
                    model.stop_multi_process_pool(pool)
            else:
                zh_emb = model.encode(
                    zh_inputs,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_numpy=True,
                )
                ja_emb = model.encode(
                    ja_inputs,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_numpy=True,
                )
        else:
            zh_emb = model.encode(
                zh_inputs,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
            )
            ja_emb = model.encode(
                ja_inputs,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
            )

        # With normalized embeddings, cosine equals dot product.
        if normalize_embeddings:
            scores = np.sum(zh_emb * ja_emb, axis=1)
        else:
            scores = np.array([_cosine(z_vec, j_vec) for z_vec, j_vec in zip(zh_emb, ja_emb)])

        for item, score in zip(batch, scores):
            score = float(score)
            seen += 1

            if progress_every > 0 and seen % progress_every == 0:
                now = time.perf_counter()
                print(
                    f"[stage4] {seen:,} pairs | last {progress_every:,}: {now - t_last:.2f}s | total: {now - t0:.2f}s"
                )
                t_last = now

            if score >= high_threshold:
                out.append(Pair(zh=item.zh, ja=item.ja, score=score))
                continue

            if score < low_threshold:
                continue

            if len(item.zh) < mid_min_char or len(item.ja) < mid_min_char:
                continue
            len_ratio = len(item.zh) / max(len(item.ja), 1)
            if abs(1.0 - len_ratio) > mid_max_len_ratio_delta:
                continue
            zh_punct = len(re.findall(r"[，。！？；：、,.!?;:()（）【】\[\]\"'“”‘’]", item.zh))
            ja_punct = len(re.findall(r"[，。！？；：、,.!?;:()（）【】\[\]\"'“”‘’]", item.ja))
            if abs(zh_punct - ja_punct) > mid_punct_mismatch_max:
                continue
            if mid_require_number_match and numbers(item.zh) != numbers(item.ja):
                continue
            out.append(Pair(zh=item.zh, ja=item.ja, score=score))

    return out


def normalized_key(zh: str, ja: str) -> str:
    zh_n = SPACE_RE.sub("", zh.lower())
    ja_n = SPACE_RE.sub("", ja.lower())
    return f"{zh_n}\t{ja_n}"


def stage5_dedup_reweight(pairs: Iterable[Pair], cfg: Dict[str, Any]) -> List[Pair]:
    dedup_exact = bool(cfg["dedup_exact"])
    dedup_near = bool(cfg["dedup_near_by_normalized_key"])
    default_weight = float(cfg["default_weight"])
    short_th = int(cfg["short_sentence_penalty_threshold"])
    short_w = float(cfg["short_sentence_weight"])

    seen_exact: set[Tuple[str, str]] = set()
    seen_near: set[str] = set()
    out: List[Pair] = []

    for item in pairs:
        exact = (item.zh, item.ja)
        if dedup_exact and exact in seen_exact:
            continue
        if dedup_exact:
            seen_exact.add(exact)

        near = normalized_key(item.zh, item.ja)
        if dedup_near and near in seen_near:
            continue
        if dedup_near:
            seen_near.add(near)

        weight = default_weight
        if len(item.zh) <= short_th or len(item.ja) <= short_th:
            weight = min(weight, short_w)
        out.append(Pair(zh=item.zh, ja=item.ja, score=item.score, weight=weight))

    return out


def split_data(pairs: Sequence[Pair], dev_ratio: float, test_ratio: float, seed: int) -> Tuple[List[Pair], List[Pair], List[Pair]]:
    items = list(pairs)
    random.Random(seed).shuffle(items)

    n = len(items)
    n_dev = int(n * dev_ratio)
    n_test = int(n * test_ratio)
    dev = items[:n_dev]
    test = items[n_dev : n_dev + n_test]
    train = items[n_dev + n_test :]
    return train, dev, test


def write_tsv(path: Path, pairs: Sequence[Pair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in pairs:
            f.write(f"{item.zh}\t{item.ja}\t{item.score:.6f}\t{item.weight:.3f}\n")


def write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def _len_distribution(arr: np.ndarray) -> Dict[str, Any]:
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0, "max": 0, "p50": 0, "p75": 0, "p95": 0, "p99": 0}
    return {
        "mean": round(float(arr.mean()), 3),
        "std": round(float(arr.std()), 3),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p50": int(np.percentile(arr, 50)),
        "p75": int(np.percentile(arr, 75)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
    }


def _try_load_sentencepiece(model_path: Path | None) -> Any | None:
    if model_path is None or not model_path.is_file():
        return None
    try:
        import sentencepiece as spm
    except ImportError:
        return None
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    return sp


def _spm_piece_total(texts: Sequence[str], sp: Any, batch_size: int) -> int:
    total = 0
    n = len(texts)
    for i in range(0, n, batch_size):
        chunk = list(texts[i : i + batch_size])
        encoded = sp.encode(chunk, out_type=int)
        total += sum(len(x) for x in encoded)
    return total


def compute_corpus_stats_for_pairs(
    pairs: Sequence[Pair],
    sp_model_path: Path | None = None,
    encode_batch_size: int = 8192,
) -> Dict[str, Any]:
    """
    Sentence-pair count, total Unicode characters, total SentencePiece tokens (if model path given),
    and character-length distribution per side.
    """
    n = len(pairs)
    if n == 0:
        out: Dict[str, Any] = {
            "sentence_pairs": 0,
            "zh_unicode_char_total": 0,
            "ja_unicode_char_total": 0,
            "zh_spm_piece_total": None,
            "ja_spm_piece_total": None,
            "zh_char_len": _len_distribution(np.array([], dtype=np.int64)),
            "ja_char_len": _len_distribution(np.array([], dtype=np.int64)),
        }
        return out

    zh_lens = np.fromiter((len(p.zh) for p in pairs), dtype=np.int64, count=n)
    ja_lens = np.fromiter((len(p.ja) for p in pairs), dtype=np.int64, count=n)
    zh_char_total = int(zh_lens.sum())
    ja_char_total = int(ja_lens.sum())

    sp = _try_load_sentencepiece(sp_model_path)
    zh_spm: int | None
    ja_spm: int | None
    if sp is None:
        zh_spm, ja_spm = None, None
    else:
        zh_texts = [p.zh for p in pairs]
        ja_texts = [p.ja for p in pairs]
        zh_spm = _spm_piece_total(zh_texts, sp, encode_batch_size)
        ja_spm = _spm_piece_total(ja_texts, sp, encode_batch_size)

    return {
        "sentence_pairs": n,
        "zh_unicode_char_total": zh_char_total,
        "ja_unicode_char_total": ja_char_total,
        "zh_spm_piece_total": zh_spm,
        "ja_spm_piece_total": ja_spm,
        "zh_char_len": _len_distribution(zh_lens),
        "ja_char_len": _len_distribution(ja_lens),
    }


def compute_corpus_stats_splits(
    train: Sequence[Pair],
    dev: Sequence[Pair],
    test: Sequence[Pair],
    sp_model_path: Path | None = None,
    encode_batch_size: int = 8192,
) -> Dict[str, Any]:
    sp_path = Path(sp_model_path) if sp_model_path is not None else None
    stats_train = compute_corpus_stats_for_pairs(train, sp_path, encode_batch_size)
    stats_dev = compute_corpus_stats_for_pairs(dev, sp_path, encode_batch_size)
    stats_test = compute_corpus_stats_for_pairs(test, sp_path, encode_batch_size)
    return {
        "spm_model": str(sp_path) if sp_path is not None else None,
        "encode_batch_size": encode_batch_size,
        "train": stats_train,
        "dev": stats_dev,
        "test": stats_test,
        "all_final_splits": {
            "sentence_pairs": stats_train["sentence_pairs"] + stats_dev["sentence_pairs"] + stats_test["sentence_pairs"],
            "zh_unicode_char_total": stats_train["zh_unicode_char_total"]
            + stats_dev["zh_unicode_char_total"]
            + stats_test["zh_unicode_char_total"],
            "ja_unicode_char_total": stats_train["ja_unicode_char_total"]
            + stats_dev["ja_unicode_char_total"]
            + stats_test["ja_unicode_char_total"],
            "zh_spm_piece_total": (
                None
                if stats_train["zh_spm_piece_total"] is None
                else stats_train["zh_spm_piece_total"]
                + stats_dev["zh_spm_piece_total"]
                + stats_test["zh_spm_piece_total"]
            ),
            "ja_spm_piece_total": (
                None
                if stats_train["ja_spm_piece_total"] is None
                else stats_train["ja_spm_piece_total"]
                + stats_dev["ja_spm_piece_total"]
                + stats_test["ja_spm_piece_total"]
            ),
        },
    }


def iter_pairs_from_filtered_tsv(path: Path) -> Iterator[Pair]:
    """Read lines written by write_tsv: zh \\t ja \\t score \\t weight (first two columns are sentences)."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            zh, ja = parts[0].strip(), parts[1].strip()
            if not zh or not ja:
                continue
            score, weight = 0.0, 1.0
            if len(parts) >= 4:
                try:
                    score = float(parts[2])
                    weight = float(parts[3])
                except ValueError:
                    pass
            yield Pair(zh=zh, ja=ja, score=score, weight=weight)


def compute_corpus_stats_from_tsv_splits(
    train_path: Path,
    dev_path: Path,
    test_path: Path,
    sp_model_path: Path | None = None,
    encode_batch_size: int = 8192,
) -> Dict[str, Any]:
    """Load pairs from existing TSV outputs (full lists) and compute the same corpus_stats as the main pipeline."""
    train = list(iter_pairs_from_filtered_tsv(train_path))
    dev = list(iter_pairs_from_filtered_tsv(dev_path))
    test = list(iter_pairs_from_filtered_tsv(test_path))
    return compute_corpus_stats_splits(train, dev, test, sp_model_path, encode_batch_size)
