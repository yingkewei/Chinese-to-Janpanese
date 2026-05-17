from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import sentencepiece as spm
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _parse_tsv_line(line: str) -> Tuple[str, str] | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2:
        return None
    if len(parts) >= 4:
        zh, ja = parts[-4], parts[-3]
    else:
        zh, ja = parts[0], parts[1]
    zh, ja = zh.strip(), ja.strip()
    if not zh or not ja:
        return None
    return zh, ja


def iter_zh_ja_tsv(path: Path, max_samples: int | None = None) -> Iterator[Tuple[str, str]]:
    """Yield (zh, ja) from raw or tokenized TSV (same column layout as pipeline)."""
    n = 0
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = _parse_tsv_line(line)
            if not p:
                continue
            yield p
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def piece_string_to_ids(sp: spm.SentencePieceProcessor, piece_str: str) -> List[int]:
    if not piece_str:
        return []
    return [sp.piece_to_id(p) for p in piece_str.split()]


@dataclass
class Batch:
    src: Tensor
    tgt_in: Tensor
    tgt_out: Tensor
    src_pad_mask: Tensor
    tgt_pad_mask: Tensor


class SubwordTSVDataset(Dataset):
    """
    Reads tokenized TSV: zh_spm \\t ja_spm \\t score \\t weight
    Encoder: zh pieces + EOS. Decoder: BOS + ja pieces (in) vs ja pieces + EOS (labels).
    """

    def __init__(
        self,
        tsv_path: Path,
        sp_model: Path,
        max_src_len: int,
        max_tgt_len: int,
        max_samples: int | None = None,
    ) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(sp_model))
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_idx = self._padding_index()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

        self.pairs: List[Tuple[str, str]] = []
        with Path(tsv_path).open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = _parse_tsv_line(line)
                if not p:
                    continue
                self.pairs.append(p)
                if max_samples is not None and len(self.pairs) >= max_samples:
                    break

    def _padding_index(self) -> int:
        pid = self.sp.pad_id()
        if pid is not None and pid >= 0:
            return pid
        return self.sp.get_piece_size()

    @property
    def vocab_size(self) -> int:
        return self.pad_idx + 1

    def __len__(self) -> int:
        return len(self.pairs)

    def encode_src(self, zh_pieces: str) -> List[int]:
        ids = piece_string_to_ids(self.sp, zh_pieces)
        ids = ids[: max(1, self.max_src_len) - 1]
        ids.append(self.eos_id)
        return ids[: self.max_src_len]

    def encode_tgt(self, ja_pieces: str) -> Tuple[List[int], List[int]]:
        ids = piece_string_to_ids(self.sp, ja_pieces)
        max_ids = max(1, self.max_tgt_len - 2)
        ids = ids[:max_ids]
        tgt_in = [self.bos_id] + ids
        tgt_out = ids + [self.eos_id]
        return tgt_in, tgt_out

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[int]]:
        zh, ja = self.pairs[idx]
        src = self.encode_src(zh)
        tgt_in, tgt_out = self.encode_tgt(ja)
        return src, tgt_in, tgt_out


def collate_nmt_batch(pad_idx: int):
    def _collate(batch: Sequence[Tuple[List[int], List[int], List[int]]]) -> Batch:
        src_list, tgt_in_list, tgt_out_list = zip(*batch)
        bs = len(batch)
        sl = max(len(s) for s in src_list)
        tl = max(max(len(a), len(b)) for a, b in zip(tgt_in_list, tgt_out_list))

        src = torch.full((bs, sl), pad_idx, dtype=torch.long)
        tgt_in = torch.full((bs, tl), pad_idx, dtype=torch.long)
        tgt_out = torch.full((bs, tl), pad_idx, dtype=torch.long)

        for i, (s, ti, to) in enumerate(zip(src_list, tgt_in_list, tgt_out_list)):
            src[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            L = min(len(ti), len(to), tl)
            tgt_in[i, :L] = torch.tensor(ti[:L], dtype=torch.long)
            tgt_out[i, :L] = torch.tensor(to[:L], dtype=torch.long)

        return Batch(
            src=src,
            tgt_in=tgt_in,
            tgt_out=tgt_out,
            src_pad_mask=src == pad_idx,
            tgt_pad_mask=tgt_in == pad_idx,
        )

    return _collate
