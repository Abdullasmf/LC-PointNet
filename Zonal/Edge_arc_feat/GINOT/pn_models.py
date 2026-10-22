from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "GINOT_A",
    "count_trainable_parameters",
]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MLP(nn.Module):
    def __init__(self, dims: List[int], activation: str = "gelu", dropout: float = 0.0, final_activation: bool = False):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("MLP dims must contain at least input and output size")
        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        if activation not in acts:
            raise ValueError(f"Unsupported activation: {activation}")
        act_cls = acts[activation]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if (not is_last) or final_activation:
                layers.append(act_cls())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierPositionalEncoding(nn.Module):
    """NeRF/Fourier-style positional encoding."""

    def __init__(self, in_dim: int, num_frequencies: int, include_input: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        if self.num_frequencies < 0:
            raise ValueError("num_frequencies must be >= 0")
        if self.num_frequencies > 0:
            freq = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
            self.register_buffer("freq_bands", freq, persistent=False)
        else:
            self.register_buffer("freq_bands", torch.empty(0), persistent=False)

    @property
    def out_dim(self) -> int:
        base = self.in_dim if self.include_input else 0
        return base + (2 * self.in_dim * self.num_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_dim:
            raise RuntimeError(f"Expected input last dim {self.in_dim}, got {x.shape[-1]}")
        parts: List[torch.Tensor] = []
        if self.include_input:
            parts.append(x)
        if self.num_frequencies > 0:
            freqs = self.freq_bands.to(dtype=x.dtype, device=x.device)
            xb = x.unsqueeze(-2) * freqs.view(*([1] * (x.ndim - 1)), -1, 1)
            xb = xb.reshape(*x.shape[:-1], -1)
            parts.append(torch.sin(xb))
            parts.append(torch.cos(xb))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]


def batched_index_select(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points with idx. points:[B,N,C], idx:[B,S] or [B,S,K] -> gathered:[B,S,C] or [B,S,K,C]."""
    b = points.shape[0]
    batch_idx_shape = [b] + [1] * (idx.ndim - 1)
    batch_idx = torch.arange(b, device=points.device).view(*batch_idx_shape).expand_as(idx)
    return points[batch_idx, idx]


def masked_farthest_point_sampling(
    xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    n_centroids: int,
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask-safe FPS. Never samples padded points."""
    bsz, n, _ = xyz.shape
    device = xyz.device
    n_centroids = int(n_centroids)
    if n_centroids < 1:
        raise ValueError("n_centroids must be >= 1")

    valid_counts = valid_mask.sum(dim=1)
    if torch.any(valid_counts == 0):
        raise RuntimeError("Each batch item must have at least one valid geometry point")

    centroids = torch.zeros((bsz, n_centroids), device=device, dtype=torch.long)
    centroid_valid = torch.zeros((bsz, n_centroids), device=device, dtype=torch.bool)

    # deterministic eval: fixed first point = smallest (x+r) valid point
    if deterministic:
        score = torch.sum(xyz, dim=-1)
        score = torch.where(valid_mask, score, torch.full_like(score, float("inf")))
        farthest = torch.argmin(score, dim=1)
    else:
        noise = torch.rand((bsz, n), device=device)
        noise = torch.where(valid_mask, noise, torch.full_like(noise, -1.0))
        farthest = torch.argmax(noise, dim=1)

    min_dist = torch.full((bsz, n), float("inf"), device=device, dtype=xyz.dtype)
    min_dist = torch.where(valid_mask, min_dist, torch.full_like(min_dist, -1.0))
    batch_ids = torch.arange(bsz, device=device)

    for i in range(n_centroids):
        centroids[:, i] = farthest
        still_valid = i < valid_counts
        centroid_valid[:, i] = still_valid

        centroid_xyz = xyz[batch_ids, farthest, :].unsqueeze(1)
        dist2 = torch.sum((xyz - centroid_xyz) ** 2, dim=-1)
        active = still_valid.unsqueeze(1)
        improved = (dist2 < min_dist) & active
        min_dist = torch.where(improved, dist2, min_dist)

        masked = torch.where(valid_mask & active, min_dist, torch.full_like(min_dist, -1.0))
        next_far = torch.argmax(masked, dim=1)

        # once valid points exhausted, keep repeating first valid centroid
        first_centroid = centroids[:, 0]
        farthest = torch.where(still_valid, next_far, first_centroid)

    return centroids, centroid_valid


def knn_group_masked(
    xyz: torch.Tensor,
    centroids_xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """kNN grouping by coordinates only; excludes padded points."""
    b, n, _ = xyz.shape
    s = centroids_xyz.shape[1]
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")

    dist2 = torch.cdist(centroids_xyz, xyz, p=2) ** 2  # [B,S,N]
    invalid_fill = torch.full_like(dist2, float("inf"))
    dist2 = torch.where(valid_mask[:, None, :], dist2, invalid_fill)

    k_eff = min(k, n)
    idx = torch.topk(dist2, k=k_eff, largest=False, dim=-1).indices  # [B,S,k_eff]
    nbr_valid = torch.isfinite(torch.topk(dist2, k=k_eff, largest=False, dim=-1).values)

    if k_eff < k:
        pad_idx = idx[..., :1].expand(-1, -1, k - k_eff)
        idx = torch.cat([idx, pad_idx], dim=-1)
        pad_valid = nbr_valid[..., :1].expand(-1, -1, k - k_eff)
        nbr_valid = torch.cat([nbr_valid, pad_valid], dim=-1)

    first_valid = idx[..., :1].expand(-1, -1, k)
    idx = torch.where(nbr_valid, idx, first_valid)
    return idx, nbr_valid


class ResidualSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP([dim, dim * 4, dim], activation="gelu", dropout=dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.lnq = nn.LayerNorm(dim)
        self.lnkv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP([dim, dim * 4, dim], activation="gelu", dropout=dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qn = self.lnq(q)
        kvn = self.lnkv(kv)
        q = q + self.attn(qn, kvn, kvn, key_padding_mask=key_padding_mask, need_weights=False)[0]
        q = q + self.mlp(self.ln2(q))
        return q


@dataclass
class GINOTConfig:
    geom_coord_dim: int = 2
    query_coord_dim: int = 2
    geom_posenc_freqs: int = 10
    query_posenc_freqs: int = 10
    n_centroids: int = 256
    n_neighbors: int = 32
    local_mlp_widths: Tuple[int, ...] = (128, 128)
    token_dim: int = 128
    encoder_heads: int = 4
    encoder_cross_attn_layers: int = 1
    encoder_self_attn_layers: int = 2
    decoder_cross_attn_layers: int = 3
    decoder_heads: int = 4
    decoder_mlp_widths: Tuple[int, ...] = (256, 128)
    head_mlp_widths: Tuple[int, ...] = (256, 128)
    dropout: float = 0.0
    encoder_gf_dim: int = 0
    head_gf_dim: int = 0


class GINOT_A(nn.Module):
    """GINOT-A: boundary-node geometry encoder + query-conditioned decoder."""

    def __init__(self, cfg: Dict[str, Any], out_dim: int, in_channels: int = 0, headfeatdim: int = 0):
        super().__init__()
        c = dict(cfg)
        c["encoder_gf_dim"] = int(in_channels)
        c["head_gf_dim"] = int(headfeatdim)
        self.cfg = GINOTConfig(**c)
        self.out_dim = int(out_dim)

        # toggles for structural tests
        self.enable_encoder_gf_route = True
        self.enable_head_gf_route = True

        self.geom_pe = FourierPositionalEncoding(
            in_dim=self.cfg.geom_coord_dim,
            num_frequencies=self.cfg.geom_posenc_freqs,
            include_input=True,
        )
        self.query_pe = FourierPositionalEncoding(
            in_dim=self.cfg.query_coord_dim,
            num_frequencies=self.cfg.query_posenc_freqs,
            include_input=True,
        )

        local_in_dim = self.cfg.geom_coord_dim + self.geom_pe.out_dim + max(0, self.cfg.encoder_gf_dim)
        local_dims = [local_in_dim] + list(self.cfg.local_mlp_widths) + [self.cfg.token_dim]
        self.local_mlp = MLP(local_dims, activation="gelu", dropout=self.cfg.dropout)
        self.global_proj = nn.Linear(self.geom_pe.out_dim, self.cfg.token_dim)

        self.encoder_cross_blocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(
                    dim=self.cfg.token_dim,
                    heads=self.cfg.encoder_heads,
                    dropout=self.cfg.dropout,
                )
                for _ in range(self.cfg.encoder_cross_attn_layers)
            ]
        )
        self.encoder_self_blocks = nn.ModuleList(
            [
                ResidualSelfAttentionBlock(
                    dim=self.cfg.token_dim,
                    heads=self.cfg.encoder_heads,
                    dropout=self.cfg.dropout,
                )
                for _ in range(self.cfg.encoder_self_attn_layers)
            ]
        )

        q_dims = [self.query_pe.out_dim] + list(self.cfg.decoder_mlp_widths) + [self.cfg.token_dim]
        self.query_token_mlp = MLP(q_dims, activation="gelu", dropout=self.cfg.dropout)
        self.decoder_cross_blocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(
                    dim=self.cfg.token_dim,
                    heads=self.cfg.decoder_heads,
                    dropout=self.cfg.dropout,
                )
                for _ in range(self.cfg.decoder_cross_attn_layers)
            ]
        )

        head_in = self.cfg.token_dim + max(0, self.cfg.head_gf_dim)
        head_dims = [head_in] + list(self.cfg.head_mlp_widths) + [self.out_dim]
        self.head = MLP(head_dims, activation="gelu", dropout=self.cfg.dropout)

        self._last_debug: Dict[str, torch.Tensor] = {}

    def get_arch(self) -> Dict[str, Any]:
        return {
            "model_family": "GINOT-A",
            "model_class": "GINOT_A",
            "model_config_identity": "GINOT_A",
            "ginot_cfg": {
                "geom_coord_dim": self.cfg.geom_coord_dim,
                "query_coord_dim": self.cfg.query_coord_dim,
                "geom_posenc_freqs": self.cfg.geom_posenc_freqs,
                "query_posenc_freqs": self.cfg.query_posenc_freqs,
                "n_centroids": self.cfg.n_centroids,
                "n_neighbors": self.cfg.n_neighbors,
                "local_mlp_widths": list(self.cfg.local_mlp_widths),
                "token_dim": self.cfg.token_dim,
                "encoder_heads": self.cfg.encoder_heads,
                "encoder_cross_attn_layers": self.cfg.encoder_cross_attn_layers,
                "encoder_self_attn_layers": self.cfg.encoder_self_attn_layers,
                "decoder_cross_attn_layers": self.cfg.decoder_cross_attn_layers,
                "decoder_heads": self.cfg.decoder_heads,
                "decoder_mlp_widths": list(self.cfg.decoder_mlp_widths),
                "head_mlp_widths": list(self.cfg.head_mlp_widths),
                "dropout": self.cfg.dropout,
                "encoder_gf_dim": self.cfg.encoder_gf_dim,
                "head_gf_dim": self.cfg.head_gf_dim,
            },
            "out_dim": self.out_dim,
        }

    def _build_valid_mask(self, geom_xyz: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(geom_xyz).all(dim=-1)

    def encode_geometry(
        self,
        geom_xyz: torch.Tensor,
        geom_feats: Optional[torch.Tensor],
        geom_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b, n, c = geom_xyz.shape
        if c != self.cfg.geom_coord_dim:
            raise RuntimeError(f"Expected geometry coord dim={self.cfg.geom_coord_dim}, got {c}")

        if geom_valid_mask is None:
            geom_valid_mask = self._build_valid_mask(geom_xyz)
        else:
            geom_valid_mask = geom_valid_mask.bool()

        # sanitize padded coordinates to 0 so they never leak through arithmetic
        geom_xyz_safe = torch.where(geom_valid_mask.unsqueeze(-1), geom_xyz, torch.zeros_like(geom_xyz))

        fps_idx, centroid_valid = masked_farthest_point_sampling(
            geom_xyz_safe,
            geom_valid_mask,
            n_centroids=self.cfg.n_centroids,
            deterministic=(not self.training),
        )
        centroid_xyz = batched_index_select(geom_xyz_safe, fps_idx)

        nbr_idx, nbr_valid = knn_group_masked(
            geom_xyz_safe,
            centroid_xyz,
            geom_valid_mask,
            k=self.cfg.n_neighbors,
        )

        grouped_xyz = batched_index_select(geom_xyz_safe, nbr_idx)
        rel_xyz = grouped_xyz - centroid_xyz.unsqueeze(2)
        grouped_abs_pe = self.geom_pe(grouped_xyz)

        local_parts = [rel_xyz, grouped_abs_pe]

        if self.cfg.encoder_gf_dim > 0:
            if geom_feats is None:
                raise RuntimeError("encoder_gf_dim > 0 but geom_feats is None")
            if geom_feats.shape[:2] != (b, n):
                raise RuntimeError(
                    f"geom_feats shape must align with geom_xyz: expected {(b, n, self.cfg.encoder_gf_dim)}, got {tuple(geom_feats.shape)}"
                )
            if geom_feats.shape[-1] != self.cfg.encoder_gf_dim:
                raise RuntimeError(
                    f"Expected encoder GF dim {self.cfg.encoder_gf_dim}, got {geom_feats.shape[-1]}"
                )
            geom_feats_safe = torch.where(
                geom_valid_mask.unsqueeze(-1), geom_feats, torch.zeros_like(geom_feats)
            )
            grouped_gf = batched_index_select(geom_feats_safe, nbr_idx)
            if self.enable_encoder_gf_route:
                local_parts.append(grouped_gf)
            else:
                local_parts.append(torch.zeros_like(grouped_gf))

        local_in = torch.cat(local_parts, dim=-1)
        local_token_per_nbr = self.local_mlp(local_in)

        nbr_valid_f = nbr_valid.unsqueeze(-1).to(local_token_per_nbr.dtype)
        local_sum = (local_token_per_nbr * nbr_valid_f).sum(dim=2)
        denom = nbr_valid_f.sum(dim=2).clamp(min=1.0)
        local_tokens = local_sum / denom

        global_tokens = self.global_proj(self.geom_pe(geom_xyz_safe))

        q_tokens = local_tokens
        for blk in self.encoder_cross_blocks:
            q_tokens = blk(q_tokens, global_tokens, key_padding_mask=~geom_valid_mask)
        q_tokens = torch.where(centroid_valid.unsqueeze(-1), q_tokens, torch.zeros_like(q_tokens))

        for blk in self.encoder_self_blocks:
            q_tokens = blk(q_tokens, key_padding_mask=~centroid_valid)
        q_tokens = torch.where(centroid_valid.unsqueeze(-1), q_tokens, torch.zeros_like(q_tokens))

        dbg = {
            "fps_idx": fps_idx.detach(),
            "centroid_valid": centroid_valid.detach(),
            "neighbor_idx": nbr_idx.detach(),
            "neighbor_valid": nbr_valid.detach(),
            "geom_valid_mask": geom_valid_mask.detach(),
        }
        return q_tokens, centroid_valid, dbg

    def forward(
        self,
        geom_xyz: torch.Tensor,
        query_points: torch.Tensor,
        geom_feats: Optional[torch.Tensor] = None,
        head_feats: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        geom_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if geom_xyz.ndim != 3 or query_points.ndim != 3:
            raise RuntimeError("geom_xyz and query_points must be rank-3 tensors [B,N,C]")
        if query_points.shape[-1] != self.cfg.query_coord_dim:
            raise RuntimeError(f"Expected query coord dim={self.cfg.query_coord_dim}, got {query_points.shape[-1]}")

        geom_tokens, geom_token_valid, dbg = self.encode_geometry(geom_xyz, geom_feats, geom_valid_mask=geom_mask)

        query_pe = self.query_pe(query_points)
        q = self.query_token_mlp(query_pe)

        for blk in self.decoder_cross_blocks:
            q = blk(q, geom_tokens, key_padding_mask=~geom_token_valid)

        if self.cfg.head_gf_dim > 0:
            if head_feats is None:
                raise RuntimeError("head_gf_dim > 0 but head_feats is None")
            if head_feats.shape[:2] != query_points.shape[:2]:
                raise RuntimeError(
                    f"head_feats must align with query_points [B,Q,*], got {tuple(head_feats.shape)} vs {tuple(query_points.shape)}"
                )
            if head_feats.shape[-1] != self.cfg.head_gf_dim:
                raise RuntimeError(f"Expected head GF dim {self.cfg.head_gf_dim}, got {head_feats.shape[-1]}")
            if self.enable_head_gf_route:
                q = torch.cat([q, head_feats], dim=-1)
            else:
                q = torch.cat([q, torch.zeros_like(head_feats)], dim=-1)

        out = self.head(q)

        if query_mask is not None:
            out = torch.where(query_mask.unsqueeze(-1), out, torch.zeros_like(out))

        self._last_debug = dbg
        return out




def _default_ginot_cfg() -> Dict[str, Any]:
    return {
        "geom_coord_dim": 2,
        "query_coord_dim": 2,
        "geom_posenc_freqs": 8,
        "query_posenc_freqs": 8,
        "n_centroids": 128,
        "n_neighbors": 16,
        "local_mlp_widths": [128, 128],
        "token_dim": 128,
        "encoder_heads": 4,
        "encoder_cross_attn_layers": 1,
        "encoder_self_attn_layers": 2,
        "decoder_cross_attn_layers": 3,
        "decoder_heads": 4,
        "decoder_mlp_widths": [256, 128],
        "head_mlp_widths": [256, 128],
        "dropout": 0.0,
    }

