"""
cida/model.py - TransformerEncoder + CollectiveDeliberation for CIDA V8.
"""
import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .cdp import CollectiveDeliberation


class CIDAModel(nn.Module):
    """
    CIDA V8: encoder backbone + CDP refinement + residual consensus.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = TransformerEncoder(
            vocab_size  = cfg.vocab_size,
            d_model     = cfg.d_model,
            n_layers    = cfg.n_layers,
            n_heads     = cfg.n_attn_heads,
            n_kv_heads  = getattr(cfg, 'n_kv_heads', cfg.n_attn_heads),
            ffn_mult    = cfg.ffn_mult,
            max_seq_len = cfg.max_seq_len,
            dropout     = cfg.dropout,
        )
        self.cdp = CollectiveDeliberation(cfg)

        # Residual fusion between base encoder logits and CDP logits.
        self.residual_alpha = nn.Parameter(torch.tensor([0.0]))

        # Residual Consensus: Base classifier on the encoder's CLS
        self.base_classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.n_classes)
        )

    def forward(self, x):
        """
        x: [B, L] token ids
        Returns: logits [B, C]
        """
        # 1. Hierarchical Resolution
        layer_states = self.encoder(x, return_layers=True)
        h_last_cls   = layer_states[-1][:, 0, :]

        # 2. Base Prediction
        base_logits  = self.base_classifier(h_last_cls)

        # 3. CDP Refinement
        B, L = x.shape
        pad_mask = (x == 0)
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)

        cdp_logits, _, _, _ = self.cdp(layer_states[-1], layer_states=layer_states, mask=full_mask)

        alpha = torch.sigmoid(self.residual_alpha)
        return alpha * cdp_logits + (1 - alpha) * base_logits

    def forward_full(self, x):
        """
        Full forward with deliberation trace.
        """
        layer_states = self.encoder(x, return_layers=True)
        h_last_cls   = layer_states[-1][:, 0, :]
        base_logits  = self.base_classifier(h_last_cls)

        B, L = x.shape
        pad_mask = (x == 0)
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)

        cdp_logits, rb, cw, av = self.cdp(layer_states[-1], layer_states=layer_states, mask=full_mask)

        alpha = torch.sigmoid(self.residual_alpha)
        return alpha * cdp_logits + (1 - alpha) * base_logits, rb, cw, av

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self):
        enc  = sum(p.numel() for p in self.encoder.parameters())
        cdp  = sum(p.numel() for p in self.cdp.parameters())
        base = sum(p.numel() for p in self.base_classifier.parameters())
        return {'encoder': enc, 'cdp': cdp, 'base': base, 'total': enc + cdp + base}


from transformers import AutoModel

class CIDABertModel(nn.Module):
    def __init__(self, cfg, bert_model_name: str = "bert-base-uncased"):
        super().__init__()
        self.cfg = cfg
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size
        self.bert_proj = nn.Linear(bert_dim, cfg.d_model) if bert_dim != cfg.d_model else nn.Identity()
        self.cdp = CollectiveDeliberation(cfg)
        self.base_classifier = nn.Linear(cfg.d_model, cfg.n_classes)
        self.residual_alpha = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x, attention_mask=None):
        return self.forward_full(x, attention_mask)[0]

    def forward_full(self, x, attention_mask=None):
        # BERT forward
        outputs = self.bert(x, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]               # [B, L, bert_dim]
        projected = self.bert_proj(last_hidden)              # [B, L, d_model]

        # маска для CDP (True = real token)
        if attention_mask is None:
            attention_mask = (x != 0).long()
        mask = attention_mask.bool()

        # CDP
        cdp_logits, round_beliefs, consensus_weights, agent_vars = self.cdp(
            projected, layer_states=None, mask=mask
        )

        # базовый классификатор на CLS токене BERT
        cls_token = projected[:, 0, :]
        base_logits = self.base_classifier(cls_token)

        # слияние
        alpha = torch.sigmoid(self.residual_alpha)
        logits = alpha * cdp_logits + (1 - alpha) * base_logits
        return logits, round_beliefs, consensus_weights, agent_vars

    def count_params(self):
        bert = sum(p.numel() for p in self.bert.parameters())
        cdp = sum(p.numel() for p in self.cdp.parameters())
        proj = sum(p.numel() for p in self.bert_proj.parameters())
        base = sum(p.numel() for p in self.base_classifier.parameters())
        return bert + cdp + proj + base

    def param_breakdown(self):
        bert = sum(p.numel() for p in self.bert.parameters())
        cdp = sum(p.numel() for p in self.cdp.parameters())
        proj = sum(p.numel() for p in self.bert_proj.parameters())
        base = sum(p.numel() for p in self.base_classifier.parameters())
        return {
            'encoder': bert + proj,
            'cdp': cdp,
            'base': base,
            'total': bert + cdp + proj + base
        }