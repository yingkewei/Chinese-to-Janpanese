from __future__ import annotations

import sentencepiece as spm
import torch

from translation.model import TransformerNMT


@torch.no_grad()
def greedy_translate(
    model: TransformerNMT,
    sp: spm.SentencePieceProcessor,
    zh_text: str,
    device: torch.device,
    max_src_len: int,
    max_tgt_len: int,
    pad_idx: int,
) -> str:
    model.eval()
    bos = sp.bos_id()
    eos = sp.eos_id()
    src_ids = sp.encode(zh_text.strip(), out_type=int)[: max_src_len - 1] + [eos]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_kpm = src == pad_idx

    ys = torch.tensor([[bos]], dtype=torch.long, device=device)
    for _ in range(max_tgt_len - 1):
        tgt_kpm = ys == pad_idx
        logits = model(src, ys, src_key_padding_mask=src_kpm, tgt_key_padding_mask=tgt_kpm)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
        if nxt.item() == eos:
            break

    out_ids = ys[0].tolist()
    if out_ids and out_ids[0] == bos:
        out_ids = out_ids[1:]
    if eos in out_ids:
        out_ids = out_ids[: out_ids.index(eos)]
    return sp.decode_ids(out_ids)
