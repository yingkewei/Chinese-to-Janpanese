from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerNMT(nn.Module):
    """
    Shared subword embedding, standard encoder–decoder Transformer (Vaswani et al.).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = math.sqrt(self.d_model)
        src_emb = self.pos_enc(self.embed(src) * scale)
        tgt_emb = self.pos_enc(self.embed(tgt_in) * scale)

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        tgt_len = tgt_in.size(1)
        # Bool causal mask (same dtype family as key_padding_mask) to avoid PyTorch warnings.
        causal = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt_in.device, dtype=torch.bool),
            diagonal=1,
        )

        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_proj(out)
