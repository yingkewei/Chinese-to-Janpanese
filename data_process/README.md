# Data Processing Pipeline (JParaCrawl zh-ja)

This folder contains a standalone data processing pipeline for building high-quality Chinese-Japanese parallel training data.

## Pipeline Stages

0. **Pre-audit**
   - Scan raw file stats: line count, parseable pair count, duplicate ratio, length distribution.
1. **Basic Rule Cleaning**
   - Remove empty/malformed lines, control-char heavy lines, noisy template-like lines.
2. **Language Detection & Normalization**
   - Normalize Unicode/whitespace/punctuation.
   - Lightweight script checks (Chinese side must contain CJK; Japanese side must contain Kana/Kanji).
3. **Length & Alignment Filtering**
   - Min/max length filters.
   - Length-ratio constraints.
   - Number consistency check.
4. **Semantic Quality Filtering (core)**
   - Uses embedding cosine similarity (`multilingual-e5` or `LaBSE`).
   - Tiered threshold policy:
     - `score >= high_threshold`: keep directly
     - `low_threshold <= score < high_threshold`: extra checks (length/number), then keep
     - `score < low_threshold`: drop
5. **Dedup + Reweight**
   - Exact pair dedup.
   - Near-duplicate removal by normalized-key fingerprints.
   - Keep a numeric weight per sample.
6. **Finalize**
   - Write final train/dev/test files and JSON metrics report.

## Files

- `config.yaml`: pipeline configuration.
- `run_pipeline.py`: entry point.
- `stages.py`: stage implementations.

## Usage

From project root:

```bash
pip install sentence-transformers pyyaml
python data_processing/run_pipeline.py --config data_processing/config.yaml
```

Quick smoke test:

```bash
python data_processing/run_pipeline.py --config data_processing/config.yaml --max-pairs 50000
```

Outputs will be written under `data_processing/output/` by default.

## Stage 4 model switch

- Use `intfloat/multilingual-e5-large` with `model_style: e5` (quality-first recommended).
- Use `intfloat/multilingual-e5-base` with `model_style: e5` (faster alternative).
- Use `sentence-transformers/LaBSE` with `model_style: labse`.
- If your environment has no outbound network, pre-download model files and set `local_files_only: true`.
