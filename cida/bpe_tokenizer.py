"""
cida/bpe_tokenizer.py — БПЕ токенизатор для CIDA (ablation по tokenizer).

Используется внешняя библиотека `tokenizers` (в venv уже установлена).
Идея: обучаем BPE merges на train_texts, затем кодируем в фиксированную длину max_len.

Важно: encoder в `encoder.py` считает padding по значению токена `0`,
поэтому мы обучаем BPE со special token `<pad>` первым, чтобы pad_id == 0,
а `<unk>` шёл вторым (unk_id == 1).
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class BPETokenizer:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, vocab_size: int = 12000, max_len: int = 128):
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Будет создан после .fit()
        self._tok: Optional[Tokenizer] = None
        self._actual_vocab_size: Optional[int] = None

    def fit(self, texts: Iterable[str]):
        tokenizer = Tokenizer(BPE(unk_token=self.UNK))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[self.PAD, self.UNK],
        )
        tokenizer.train_from_iterator((str(t) for t in texts), trainer=trainer)

        # Проверка критичных id для совместимости с encoder.py
        pad_id = tokenizer.token_to_id(self.PAD)
        unk_id = tokenizer.token_to_id(self.UNK)
        if pad_id != 0 or unk_id != 1:
            raise ValueError(
                f"BPETokenizer special token ids mismatch: "
                f"<pad> id={pad_id}, <unk> id={unk_id}. "
                f"Expected <pad>=0 and <unk>=1 to keep compatibility with encoder padding mask."
            )

        tokenizer.enable_padding(length=self.max_len, pad_id=0, pad_token=self.PAD)
        tokenizer.enable_truncation(max_length=self.max_len)

        self._tok = tokenizer
        self._actual_vocab_size = len(tokenizer.get_vocab())
        return self

    def encode(self, text: str) -> List[int]:
        assert self._tok is not None, "Tokenizer is not fitted. Call .fit() first."
        ids = self._tok.encode(str(text)).ids
        # tokenizers уже добавляет padding/truncation до max_len
        return ids

    def encode_batch(self, texts: Iterable[str]) -> np.ndarray:
        return np.array([self.encode(t) for t in texts], dtype=np.int64)

    @property
    def actual_vocab_size(self) -> int:
        if self._actual_vocab_size is None:
            raise RuntimeError("Tokenizer is not fitted. Call .fit() first.")
        return self._actual_vocab_size

