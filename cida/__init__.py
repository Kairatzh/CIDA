"""
Public API for CIDA V8.
"""
from .trainer import CIDAClassifier
from .config import CIDAConfig
from .model import CIDAModel
from .tokenizer import WordTokenizer

try:
    from .bpe_tokenizer import BPETokenizer
except ImportError:
    BPETokenizer = None

__version__ = "8.1.0"
__all__ = ["CIDAClassifier", "CIDAConfig", "CIDAModel", "WordTokenizer", "BPETokenizer"]
