import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    RMSNorm,
    MambaCell,
    GRUCellDeliberation,
    SlotAttentionPerspectiveGenerator,
    SparseTokenCommunication,
    SparseMoE,
    UncertaintyHead,
    ConsensusAggregation,
    AgentLoRA,
)


class DeliberationLayer(nn.Module):
    """
    V8 deliberation layer with gated opinion revision.
    """

    def __init__(
        self,
        d_model: int,
        n_agents: int,
        n_theses: int,
        bottleneck_ratio: float,
        cell_type: str = "mamba",
        d_state: int = 16,
        dropout: float = 0.1,
        use_moe: bool = False,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.comm = SparseTokenCommunication(d_model, n_theses=n_theses, dropout=dropout)
        self.lora = AgentLoRA(d_model, n_agents, rank=8)

        if use_moe:
            self.cell = SparseMoE(d_model, n_experts=4)
        else:
            if cell_type == "mamba":
                self.cell = MambaCell(d_model, d_state=d_state)
            elif cell_type == "gru":
                self.cell = GRUCellDeliberation(d_model)
            else:
                self.cell = None

        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, beliefs, context=None, state=None):
        B, N, D = beliefs.shape
        h_old = beliefs

        attended = self.comm(beliefs)
        personalized = self.lora(attended)

        if context is not None:
            ctx_summary = context.mean(dim=1, keepdim=True)
            personalized = personalized + ctx_summary

        if self.cell is not None:
            if self.use_moe:
                upd_beliefs = self.cell(personalized)
                new_state = None
            else:
                flat = personalized.reshape(B * N, D)
                # передаём state (может быть None)
                updated, new_state = self.cell(flat, state)
                upd_beliefs = updated.reshape(B, N, D)
        else:
            upd_beliefs = personalized
            new_state = None

        gate = self.update_gate(torch.cat([h_old, upd_beliefs], dim=-1))
        revised = h_old + gate * self.drop(upd_beliefs)
        return self.norm(revised), new_state


class MetaReviewer(nn.Module):
    def __init__(self, d_model, d_meta, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_meta),
            nn.GELU(),
            nn.LayerNorm(d_meta),
            nn.Linear(d_meta, d_meta),
            nn.GELU(),
            nn.Linear(d_meta, d_meta),
            nn.GELU(),
            nn.LayerNorm(d_meta),
            nn.Linear(d_meta, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CollectiveDeliberation(nn.Module):
    """
    CIDA V8 collective deliberation with configurable weight tying across rounds.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_agents = cfg.n_agents
        self.n_rounds = cfg.n_rounds
        self.n_rounds_min = getattr(cfg, "n_rounds_min", 1)
        self.act_threshold = getattr(cfg, "act_threshold", 0.05)
        self.d = cfg.d_model
        self.share_deliberation_layers = getattr(cfg, "share_deliberation_layers", True)

        self.perspective_gen = SlotAttentionPerspectiveGenerator(cfg.d_model, cfg.n_agents)

        if self.share_deliberation_layers:
            self.universal_layer = DeliberationLayer(
                cfg.d_model,
                cfg.n_agents,
                cfg.n_theses,
                cfg.bottleneck_ratio,
                cell_type=cfg.deliberation_cell,
                d_state=cfg.mamba_d_state,
                dropout=cfg.cdp_dropout,
                use_moe=getattr(cfg, "use_moe", False),
            )
            self.deliberation_layers = None
        else:
            self.universal_layer = None
            self.deliberation_layers = nn.ModuleList(
                [
                    DeliberationLayer(
                        cfg.d_model,
                        cfg.n_agents,
                        cfg.n_theses,
                        cfg.bottleneck_ratio,
                        cell_type=cfg.deliberation_cell,
                        d_state=cfg.mamba_d_state,
                        dropout=cfg.cdp_dropout,
                        use_moe=getattr(cfg, "use_moe", False),
                    )
                    for _ in range(cfg.n_rounds)
                ]
            )

        self.uncertainty_heads = nn.ModuleList([UncertaintyHead(cfg.d_model) for _ in range(cfg.n_agents)])
        self.consensus_agg = ConsensusAggregation(cfg.d_model, cfg.n_agents)
        self.classifier = MetaReviewer(cfg.d_model, cfg.d_meta, cfg.n_classes)
        self.temp_param = nn.Parameter(torch.ones(1) * 2.0)

    def _get_layer(self, round_idx: int):
        if self.share_deliberation_layers:
            return self.universal_layer
        return self.deliberation_layers[round_idx]

    def forward(self, encoder_output: torch.Tensor, layer_states: list = None, mask: torch.Tensor = None):
        B, L_enc, _ = encoder_output.shape

        if mask is not None and mask.shape[1] < L_enc:
            diff = L_enc - mask.shape[1]
            cls_mask = torch.zeros((B, diff), dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        beliefs = self.perspective_gen(encoder_output, mask=mask)
        round_beliefs = [beliefs]

        prev_consensus = None
        initial_diff = None
        final_rounds = self.n_rounds
        state = None  # инициализация состояния

        for r in range(self.n_rounds):
            context = None
            if layer_states is not None:
                stage_idx = min(len(layer_states) - 1, int((r / self.n_rounds) * len(layer_states)))
                context = layer_states[stage_idx]

            layer = self._get_layer(r)
            beliefs, state = layer(beliefs, context=context, state=state)
            round_beliefs.append(beliefs)

            if r >= self.n_rounds_min:
                curr_outputs = []
                curr_vars = []
                for i in range(self.n_agents):
                    feat, var = self.uncertainty_heads[i](beliefs[:, i])
                    curr_outputs.append(feat)
                    curr_vars.append(var)
                c_feat, _ = self.consensus_agg(torch.stack(curr_outputs, 1), torch.stack(curr_vars, 1))

                if prev_consensus is not None:
                    diff = (c_feat - prev_consensus).norm(dim=-1).max()
                    if initial_diff is None:
                        initial_diff = diff.detach().clone() + 1e-4
                    if diff < self.act_threshold * initial_diff:
                        final_rounds = r + 1
                        break
                prev_consensus = c_feat

        agent_outputs = []
        agent_vars = []
        for i in range(self.n_agents):
            feat, var = self.uncertainty_heads[i](beliefs[:, i])
            agent_outputs.append(feat)
            agent_vars.append(var)

        agent_outputs = torch.stack(agent_outputs, dim=1)
        agent_vars = torch.stack(agent_vars, dim=1)

        curr_temp = torch.exp(self.temp_param) / max(1, final_rounds)
        consensus, consensus_weights = self.consensus_agg(agent_outputs, agent_vars, temp=curr_temp)
        logits = self.classifier(consensus)
        return logits, round_beliefs, consensus_weights, agent_vars.squeeze(-1)


def pairwise_decorrelation_loss(agent_states: torch.Tensor) -> torch.Tensor:
    """
    Penalizes off-diagonal cosine similarity between agents inside a sample.
    """

    B, N, _ = agent_states.shape
    normed = F.normalize(agent_states, dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2))
    eye = torch.eye(N, device=agent_states.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_fill(eye, 0.0)
    denom = max(1, B * N * (N - 1))
    return off_diag.pow(2).sum() / denom


def cdp_loss(logits, labels, round_beliefs, agent_vars, consensus_weights, cfg):
    """
    Coherent V8 loss:
    - task loss
    - early non-trivial update (margin-based)
    - late-round stability
    - pairwise decorrelation between agents
    - dominance penalty
    - ponder cost
    """

    B, C = logits.shape
    device = logits.device
    smooth = getattr(cfg, "label_smooth", 0.05)

    soft_labels = torch.full_like(logits, smooth / C)
    soft_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smooth + smooth / C)
    l_task = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(-1).mean()

    if len(round_beliefs) > 1:
        first_delta = (round_beliefs[1] - round_beliefs[0]).norm(dim=-1).mean()
        l_nt = F.relu(getattr(cfg, "nontrivial_margin", 0.35) - first_delta)
    else:
        l_nt = torch.tensor(0.0, device=device)

    if len(round_beliefs) > 2:
        later_steps = []
        for t in range(2, len(round_beliefs)):
            later_steps.append((round_beliefs[t] - round_beliefs[t - 1]).norm(dim=-1).mean())
        l_conv = torch.stack(later_steps).mean()
    elif len(round_beliefs) > 1:
        l_conv = (round_beliefs[-1] - round_beliefs[-2]).norm(dim=-1).mean()
    else:
        l_conv = torch.tensor(0.0, device=device)

    l_div = pairwise_decorrelation_loss(round_beliefs[0])
    avg_weights = consensus_weights.mean(dim=0)
    l_dom = (avg_weights - 1.0 / cfg.n_agents).pow(2).sum()
    ponder_cost = torch.tensor(float(len(round_beliefs) - 1) / max(1, cfg.n_rounds), device=device)

    total = (
        l_task
        + cfg.lambda_conv * l_conv
        + cfg.lambda_nt * l_nt
        + getattr(cfg, "lambda_div", 0.2) * l_div
        + getattr(cfg, "lambda_dom", 0.1) * l_dom
        + getattr(cfg, "lambda_ponder", 0.01) * ponder_cost
    )

    return total, {
        "task": l_task.item(),
        "conv": (l_conv * cfg.lambda_conv).item(),
        "progress": (l_nt * cfg.lambda_nt).item(),
        "div": (l_div * getattr(cfg, "lambda_div", 0.2)).item(),
        "dom": (l_dom * getattr(cfg, "lambda_dom", 0.1)).item(),
        "ponder": (ponder_cost * getattr(cfg, "lambda_ponder", 0.01)).item(),
    }
