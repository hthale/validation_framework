import torch
import torch.nn as nn

class CharTokenizer:
    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-:")
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        vocab = [self.pad_token, self.unk_token] + chars
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, max_len: int) -> list[int]:
        text = "" if text is None else str(text)
        ids = [self.stoi.get(ch, self.stoi[self.unk_token]) for ch in text[:max_len]]
        if len(ids) < max_len:
            ids += [self.stoi[self.pad_token]] * (max_len - len(ids))
        return ids

class MeanCharEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, out_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.proj = nn.Linear(emb_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)                      # [B, L, E]
        mask = (x != 0).unsqueeze(-1).float()       # pad_idx = 0
        summed = (emb * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        return self.proj(pooled)