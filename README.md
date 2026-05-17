# Chinese → Japanese 翻译实验

本仓库包含中日平行语料清洗流水线、SentencePiece 与分词数据构建，以及多条翻译模型路线：**自研 Transformer**（`translation/train.py`）、**NLLB 离线微调**（`translation/nllb_offline_sft.py`）、**Qwen2.5 QLoRA**（`translation/qwen25_qlora_sft.py`）。

以下命令均在**项目根目录**执行：

```bash
cd /path/to/Chinese_to_Japanese
```

建议使用 **Python 3.10+**，并优先用 GPU（训练与评测会快很多）。

---

## 1. 环境依赖

按你要跑的模块安装对应 requirements（可合并安装）：

| 用途 | 文件 |
|------|------|
| 自研 Transformer 训练 / 评测 | `requirements-translation.txt` |
| 清洗流水线 SPM 统计、`dataset/test` 下 xlsx 对齐 | `requirements-tokenized.txt` |
| NLLB 微调与评测 | `requirements-finetune.txt` |
| Qwen QLoRA | `requirements-qwen-qlora.txt` |

示例（跑通主流程至少需要前两份 + 流水线依赖）：

```bash
pip install -r requirements-translation.txt -r requirements-tokenized.txt
pip install sentence-transformers pyyaml
```

说明：

- 主清洗流水线默认开启 **Stage 4 语义过滤**，需要下载嵌入模型（如 `intfloat/multilingual-e5-large`），并会用到 PyTorch。若只想快速验证管线能否跑完，可在 `data_processing/config.yaml` 里把 `stage4_semantic.enabled` 设为 `false`（跳过语义阶段，结果会变差但省时）。
- 无网或离线时：预先下载 Stage 4 所用模型到本地，并在 `config.yaml` 的 `stage4_semantic` 中把 `model_name` 指到本地目录，且设 `local_files_only: true`。详见 `data_processing/README.md`。

---

## 2. 数据说明

- 默认原始平行语料路径由 `data_processing/config.yaml` 中 `input.raw_file` 指定，当前为：
  `dataset/train/zh-ja/zh-ja.crowdsourcing_b05l07.txt`（制表符分隔 zh / ja）。
- 评测集若存在 `dataset/test/*.xlsx`，会在下一步被对齐到 `dataset_processed/eval/`；没有 xlsx 时仅跳过评测导出，不影响训练。
  (注意，由于训练所用数据太大，这里就不放置了http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/，大家可以在该链接去下载训练数据)
---

## 3. 跑通主流程：清洗 → SPM → 自研 Transformer → 评测

### 3.1 数据清洗（写出 `train/dev/test.filtered.tsv`）

```bash
# 全量较慢；可先小规模冒烟
python data_processing/run_pipeline.py --config data_processing/config.yaml --max-pairs 50000
```

输出默认在 `data_processing/output/`。

### 3.2 构建 `dataset_processed`（SentencePiece + 分词 TSV + 可选 eval）

```bash
python scripts/prepare_tokenized_dataset.py
```

生成目录大致包括：

- `dataset_processed/raw/` — 复制自上一步的 filtered TSV  
- `dataset_processed/spm/mixed_zh_ja.model` — SentencePiece  
- `dataset_processed/tokenized/*.spm.tsv` — 训练用子词 id  
- `dataset_processed/eval/*.raw.tsv` — 若有 xlsx 则有评测 raw/spm  

### 3.3 训练自研 Transformer（小规模示例）

```bash
python translation/train.py --max-train-samples 10000 --epochs 1 --batch-size 32
```

检查点默认写在 `translation/checkpoints/`（如 `last.pt`）。可按需调整 `--device`、`--epochs`、`--max-train-samples` 等。

### 3.4 评测（语料级 BLEU + 输出 TSV）

```bash
python translation/evaluate_corpus.py --checkpoint translation/checkpoints/last.pt
```

默认读取 `dataset_processed/eval/` 下**唯一**的 `*.raw.tsv`；也可显式指定 `--eval-raw-tsv`。

---

## 4. 其他常用路径

### 4.1 无过滤「原始」划分（对照实验）

使用 `data_processing_raw/config.yaml`，生成未经过滤的划分：

```bash
python data_processing_raw/run_raw_baseline.py --config data_processing_raw/config.yaml --max-pairs 50000
```

输出在 `data_processing_raw/output/`。接着复用清洗流水线训练好的 SPM，生成 `dataset_processed_raw` 并可直接训练：

```bash
python scripts/prepare_raw_tokenized_dataset.py
```

脚本结束时会打印带 `--train-tsv` / `--spm-model` 的 `translation/train.py` 示例命令。

### 4.2 NLLB 离线微调（需本地模型目录）

默认期望权重位于：`models/nllb-200-distilled-600M/`（若不存在，运行脚本会打印下载说明）。

```bash
pip install -r requirements-finetune.txt
python translation/nllb_offline_sft.py --fp16 --gradient-checkpointing --max-train-samples 50000 --epochs 1 --batch-size 8 --grad-accum 2
```

评测微调后的目录（与 `evaluate_corpus.py` 同格式的平行 TSV）：

```bash
python translation/evaluate_nllb.py --model-dir translation/checkpoints_nllb_offline
```

### 4.3 Qwen2.5 QLoRA（需本地 7B 目录）

默认期望：`models/Qwen2.5-7B-Instruct/`。

```bash
pip install -r requirements-qwen-qlora.txt
python translation/qwen25_qlora_sft.py --bf16 --gradient-checkpointing --epochs 1 --batch-size 1 --grad-accum 16 --max-seq-length 512
```

---

## 5. 说明

- 所有脚本均设计为在**项目根目录**运行；若从别处调用，请自行确认相对路径。
- NLLB / Qwen 脚本在无外网机器上需事先把模型拉到 `models/` 下对应子目录。
- 更细的流水线阶段说明见 `data_processing/README.md`；任务背景与技术报告见 `docs/`。
