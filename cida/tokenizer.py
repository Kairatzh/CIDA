"""
cida/tokenizer.py — SOTA BPE & Word-level токенизаторы
"""
import re
import numpy as np
from collections import Counter
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False


class WordTokenizer:
    """Word-level токенизатор с частотным словарём (Legacy/Baseline)."""
    PAD = 0
    UNK = 1
    def __init__(self, vocab_size: int = 2000, max_len: int = 128):
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.word2idx   = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word   = {0: '<PAD>', 1: '<UNK>'}
        self._fitted    = False

    @staticmethod
    def _clean(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def _tokenize(self, text: str) -> list:
        return self._clean(text).split()

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(str(text)))
        for word, _ in counter.most_common(self.vocab_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word
        self._fitted = True
        return self

    def encode(self, text: str) -> list:
        tokens = self._tokenize(str(text))[:self.max_len]
        ids    = [self.word2idx.get(w, self.UNK) for w in tokens]
        ids   += [self.PAD] * (self.max_len - len(ids))
        return ids

    def encode_batch(self, texts) -> np.ndarray:
        return np.array([self.encode(t) for t in texts], dtype=np.int64)

    @property
    def actual_vocab_size(self) -> int:
        return len(self.word2idx)


class BPETokenizer:
    """Subword BPE Tokenizer (SOTA choice for V8)."""
    def __init__(self, vocab_size: int = 8000, max_len: int = 128):
        if not HAS_TOKENIZERS:
            raise ImportError("библиотека 'tokenizers' не установлена. Используйте 'pip install tokenizers'")
        
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.tokenizer  = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.trainer    = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )
        self._fitted = False

    def fit(self, texts):
        def batch_iterator():
            for i in range(0, len(texts), 1000):
                yield [str(t) for t in texts[i : i + 1000]]
        
        self.tokenizer.train_from_iterator(batch_iterator(), trainer=self.trainer)
        self.tokenizer.enable_padding(length=self.max_len, pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=self.max_len)
        self._fitted = True
        return self

    def encode(self, text: str) -> list:
        return self.tokenizer.encode(str(text)).ids

    def encode_batch(self, texts) -> np.ndarray:
        encodings = self.tokenizer.encode_batch([str(t) for t in texts])
        return np.array([e.ids for e in encodings], dtype=np.int64)

    @property
    def actual_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
