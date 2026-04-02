"""
cida/config.py - Configuration for CIDA V8.
"""
from dataclasses import dataclass


@dataclass
class CIDAConfig:
    # Encoder
    vocab_size: int = 2000
    d_model: int = 64
    n_layers: int = 3
    n_attn_heads: int = 4
    n_kv_heads: int = 2
    ffn_mult: int = 4
    max_seq_len: int = 128
    dropout: float = 0.10
    cdp_dropout: float = 0.10

    # Collective Deliberation Protocol V8
    n_agents: int = 8
    n_rounds: int = 3
    n_rounds_min: int = 1
    act_threshold: float = 0.05
    n_stages: int = 3
    bottleneck_ratio: float = 0.5
    n_theses: int = 4
    mamba_d_state: int = 16
    use_moe: bool = True
    deliberation_cell: str = "mamba"
    share_deliberation_layers: bool = True
    d_meta: int = 512
    use_bpe: bool = True

    # Output
    n_classes: int = 4

    # Training
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 128
    max_epochs: int = 30
    patience: int = 8
    label_smooth: float = 0.05
    grad_clip: float = 0.3

    # Loss weights
    lambda_conv: float = 0.5
    lambda_nt: float = 0.1
    lambda_div: float = 0.2
    lambda_dom: float = 0.1
    lambda_ponder: float = 0.01
    lambda_ent: float = 0.01
    nontrivial_margin: float = 0.35
    temp_min: float = 0.5
    temp_max: float = 1.5

    # Logging
    verbose: bool = True
    log_every: int = 1

    def __post_init__(self):
        assert self.d_model % self.n_attn_heads == 0
        assert self.n_attn_heads % self.n_kv_heads == 0
        assert self.deliberation_cell in {"mamba", "gru", "none"}

    @classmethod
    def small(cls, **kw):
        n_agents = kw.pop("n_agents", 8)
        return cls(d_model=64, n_layers=3, n_agents=n_agents, n_rounds=3, **kw)

    @classmethod
    def medium(cls, **kw):
        n_agents = kw.pop("n_agents", 8)
        return cls(d_model=96, n_layers=4, n_agents=n_agents, n_rounds=3, **kw)
