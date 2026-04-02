import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int = None, dropout: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated * sin)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, q, k):
        seq_len = q.shape[2]
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]
        return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)


class MambaCell(nn.Module):
    """
    Mamba-inspired selective SSM cell.

    This is intentionally a lightweight approximation rather than a full
    implementation of the official Mamba selective scan kernel. The naming
    is kept for backward compatibility, but documentation should refer to it
    as a Mamba-inspired or selective SSM-style deliberation cell.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.bc_proj = nn.Linear(d_model, d_model * d_state * 2, bias=False)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.shape[0], self.d_model, self.d_state, device=x.device, dtype=x.dtype)

        u = self.in_proj(x)
        delta = F.softplus(self.dt_proj(x)).unsqueeze(-1)
        bc = self.bc_proj(x).view(x.shape[0], self.d_model, 2, self.d_state)
        B_t = bc[:, :, 0, :]
        C_t = bc[:, :, 1, :]

        A = -torch.exp(self.A_log).unsqueeze(0)
        dA = torch.exp(delta * A)
        input_term = delta * B_t * u.unsqueeze(-1)
        h_new = dA * h + input_term

        y = (h_new * C_t).sum(dim=-1) + self.D * u
        return self.out_proj(F.silu(y)), h_new


class GRUCellDeliberation(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.gru = nn.GRUCell(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, h=None):
        if h is None:
            h = torch.zeros(x.shape[0], self.d_model, device=x.device, dtype=x.dtype)
        h_new = self.gru(x, h)
        return self.out(F.silu(h_new)), h_new


class SDPA_Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int = None, dropout: float = 0.0):
        super().__init__()
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert d_model % num_heads == 0
        assert num_heads % self.num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, rope=None):
        B, L_q, _ = query.shape
        _, L_k, _ = key.shape

        q = self.q_proj(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L_k, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L_k, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            q, k = rope(q, k)

        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(B, self.num_heads, L_k, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(B, self.num_heads, L_k, self.head_dim)

        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        out = self.out_proj(out)

        weights = None
        if need_weights:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                attn_weights.masked_fill_(~mask, -1e4)
            attn_weights = F.softmax(attn_weights, dim=-1)
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask, 0.0)
            weights = attn_weights.mean(dim=1)

        return out, weights


class MultiViewPerspectiveGenerator(nn.Module):
    def __init__(self, d_model: int, n_agents: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_agents = n_agents
        self.queries = nn.Parameter(torch.randn(1, n_agents, d_model) * 0.02)
        self.cross_attn = SDPA_Attention(d_model, n_heads, dropout=dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, encoder_output: torch.Tensor, mask: torch.Tensor = None):
        B = encoder_output.shape[0]
        q = self.queries.expand(B, -1, -1)
        out, _ = self.cross_attn(q, encoder_output, encoder_output, key_padding_mask=mask)
        return self.norm(out + q)


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x, surprise_gate):
        gate = torch.sigmoid(surprise_gate)
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        return self.norm(x + gate * h)


class SparseTokenCommunication(nn.Module):
    def __init__(self, d_model: int, n_theses: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_theses = n_theses
        self.compressor = nn.Linear(d_model, d_model * n_theses)
        self.cross_attn = SDPA_Attention(d_model, n_heads, dropout=dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, beliefs: torch.Tensor):
        B, N, d = beliefs.shape
        theses = self.compressor(beliefs).view(B, N, self.n_theses, d)
        theses = theses.view(B, N * self.n_theses, d)
        attended, _ = self.cross_attn(beliefs, theses, theses)
        return self.norm(attended + beliefs)


class ParallelSwiGLUExperts(nn.Module):
    def __init__(self, n_experts: int, d_model: int, hidden_dim: int):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        scale = 0.02
        self.w1 = nn.Parameter(torch.randn(n_experts, d_model, hidden_dim) * scale)
        self.w2 = nn.Parameter(torch.randn(n_experts, d_model, hidden_dim) * scale)
        self.w3 = nn.Parameter(torch.randn(n_experts, hidden_dim, d_model) * scale)

    def forward(self, x):
        h1 = torch.einsum("...d,edh->...eh", x, self.w1)
        h2 = torch.einsum("...d,edh->...eh", x, self.w2)
        hidden = F.silu(h1) * h2
        return torch.einsum("...eh,eho->...eo", hidden, self.w3)


class SparseMoE(nn.Module):
    """
    Vectorized Top-2 MoE without per-expert Python loops in the forward path.
    """

    def __init__(self, d_model: int, n_experts: int = 4, expert_hidden: int = None):
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Linear(d_model, n_experts)
        expert_hidden = expert_hidden or d_model * 2
        self.experts = ParallelSwiGLUExperts(n_experts, d_model, expert_hidden)

    def forward(self, x):
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 2, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        dispatch = F.one_hot(top_indices, num_classes=self.n_experts).to(x.dtype)
        dispatch = (dispatch * top_probs.unsqueeze(-1)).sum(dim=-2)
        expert_out = self.experts(x)
        return (expert_out * dispatch.unsqueeze(-1)).sum(dim=-2)


class SlotAttentionPerspectiveGenerator(nn.Module):
    def __init__(self, d_model: int, n_agents: int, n_iters: int = 4, eps: float = 1e-8):
        super().__init__()
        self.d_model = d_model
        self.n_agents = n_agents
        self.n_iters = n_iters
        self.eps = eps

        self.slots_mu = nn.Parameter(torch.randn(1, 1, d_model))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, d_model)))

        self.project_q = nn.Linear(d_model, d_model, bias=False)
        self.project_k = nn.Linear(d_model, d_model, bias=False)
        self.project_v = nn.Linear(d_model, d_model, bias=False)
        self.update_proj = SwiGLU(d_model, d_model * 2, d_model)

        self.gru = nn.GRUCell(d_model, d_model)
        self.norm_input = RMSNorm(d_model)
        self.norm_slots = RMSNorm(d_model)

    def forward(self, encoder_output: torch.Tensor, mask: torch.Tensor = None):
        B, _, D = encoder_output.shape
        inputs = self.norm_input(encoder_output)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        slots = self.slots_mu + self.slots_sigma * torch.randn(B, self.n_agents, D, device=inputs.device)

        for _ in range(self.n_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)

            dots = torch.einsum("bnd,bld->bnl", q, k) * (D ** -0.5)
            if mask is not None:
                dots.masked_fill_(mask.unsqueeze(1), -1e4)

            attn = F.softmax(dots, dim=1)
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1), 0.0)

            updates = torch.einsum("bnl,bld->bnd", attn + self.eps, v)
            updates = updates / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = self.update_proj(updates)

            slots = self.gru(updates.reshape(-1, D), slots_prev.reshape(-1, D))
            slots = slots.view(B, self.n_agents, D)

        return slots


class ConsensusAggregation(nn.Module):
    def __init__(self, d_model: int, n_agents: int):
        super().__init__()
        self.d_model = d_model
        self.trust_param = nn.Parameter(torch.ones(1, n_agents))
        self.norm = RMSNorm(d_model)

    def forward(self, agent_feats, agent_vars, temp=1.0):
        weights = F.softmax(-agent_vars.squeeze(-1) / temp, dim=-1)
        trust = F.softmax(self.trust_param, dim=-1)
        combined_weights = F.softmax(torch.log(weights + 1e-8) + torch.log(trust + 1e-8), dim=-1)
        consensus = torch.einsum("bn,bnd->bd", combined_weights, agent_feats)
        return self.norm(consensus), combined_weights


class AgentLoRA(nn.Module):
    def __init__(self, d_model: int, n_agents: int, rank: int = 8):
        super().__init__()
        self.n_agents = n_agents
        self.rank = rank
        self.lora_a = nn.Parameter(torch.randn(n_agents, d_model, rank) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(n_agents, rank, d_model))

    def forward(self, x):
        delta = torch.matmul(x.unsqueeze(2), self.lora_a.unsqueeze(0))
        delta = torch.matmul(delta, self.lora_b.unsqueeze(0))
        return x + delta.squeeze(2)


class UncertaintyHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.feature_proj = nn.Linear(d_model, d_model)
        self.var_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        feat = self.feature_proj(x)
        var = F.softplus(self.var_proj(x)) + 1e-4
        return feat, var
