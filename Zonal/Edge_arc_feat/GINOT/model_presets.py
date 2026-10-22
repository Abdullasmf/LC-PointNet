from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


def _compat_preset(name: str, ginot_cfg: Dict[str, Any]) -> Dict[str, Any]:
    token_dim = int(ginot_cfg["token_dim"])
    return {
        "model_name": f"ginot_a_{name.lower()}",
        "epochs": 10000,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        # compatibility keys expected by existing training scripts
        "latent_dim": token_dim,
        "pre_hidden": [token_dim, token_dim],
        "sa_blocks": [{"npoint": int(ginot_cfg["n_centroids"]), "nsample": int(ginot_cfg["n_neighbors"])}],
        "gf_hidden": [token_dim],
        "head_hidden": list(ginot_cfg["head_mlp_widths"]),
        "posenc": {"n_freqs": int(ginot_cfg["geom_posenc_freqs"]), "scale": 1.0},
        "head_posenc": {"n_freqs": int(ginot_cfg["query_posenc_freqs"]), "scale": 1.0},
        "norm": "layer",
        "num_groups": 1,
        "pool": "mean",
        "head_norm": "layer",
        "head_dropout": float(ginot_cfg["dropout"]),
        "fp": {"ginot_cfg": dict(ginot_cfg)},
        # explicit GINOT-A declaration
        "ginot_cfg": dict(ginot_cfg),
    }


PRESETS: Dict[str, Dict[str, Any]] = {
    "XS": _compat_preset(
        "XS",
        {
            "geom_coord_dim": 2,
            "query_coord_dim": 2,
            "geom_posenc_freqs": 4,
            "query_posenc_freqs": 4,
            "n_centroids": 32,
            "n_neighbors": 8,
            "local_mlp_widths": [64, 64],
            "token_dim": 64,
            "encoder_heads": 4,
            "encoder_cross_attn_layers": 1,
            "encoder_self_attn_layers": 1,
            "decoder_cross_attn_layers": 1,
            "decoder_heads": 4,
            "decoder_mlp_widths": [96, 64],
            "head_mlp_widths": [96, 64],
            "dropout": 0.0,
        },
    ),
    "S": _compat_preset(
        "S",
        {
            "geom_coord_dim": 2,
            "query_coord_dim": 2,
            "geom_posenc_freqs": 6,
            "query_posenc_freqs": 6,
            "n_centroids": 64,
            "n_neighbors": 12,
            "local_mlp_widths": [96, 96],
            "token_dim": 96,
            "encoder_heads": 4,
            "encoder_cross_attn_layers": 1,
            "encoder_self_attn_layers": 2,
            "decoder_cross_attn_layers": 2,
            "decoder_heads": 4,
            "decoder_mlp_widths": [160, 96],
            "head_mlp_widths": [160, 96],
            "dropout": 0.0,
        },
    ),
    "M": _compat_preset(
        "M",
        {
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
        },
    ),
    "L": _compat_preset(
        "L",
        {
            "geom_coord_dim": 2,
            "query_coord_dim": 2,
            "geom_posenc_freqs": 10,
            "query_posenc_freqs": 10,
            "n_centroids": 192,
            "n_neighbors": 24,
            "local_mlp_widths": [160, 160],
            "token_dim": 160,
            "encoder_heads": 8,
            "encoder_cross_attn_layers": 2,
            "encoder_self_attn_layers": 3,
            "decoder_cross_attn_layers": 4,
            "decoder_heads": 8,
            "decoder_mlp_widths": [320, 160],
            "head_mlp_widths": [320, 160],
            "dropout": 0.0,
        },
    ),
    "XL": _compat_preset(
        "XL",
        {
            "geom_coord_dim": 2,
            "query_coord_dim": 2,
            "geom_posenc_freqs": 12,
            "query_posenc_freqs": 12,
            "n_centroids": 256,
            "n_neighbors": 32,
            "local_mlp_widths": [192, 192],
            "token_dim": 192,
            "encoder_heads": 8,
            "encoder_cross_attn_layers": 2,
            "encoder_self_attn_layers": 4,
            "decoder_cross_attn_layers": 5,
            "decoder_heads": 8,
            "decoder_mlp_widths": [384, 192],
            "head_mlp_widths": [384, 192],
            "dropout": 0.0,
        },
    ),
    "MATCH_250K": _compat_preset(
        "MATCH_250K",
        {
            "geom_coord_dim": 2,
            "query_coord_dim": 2,
            "geom_posenc_freqs": 6,
            "query_posenc_freqs": 6,
            "n_centroids": 128,
            "n_neighbors": 16,
            "local_mlp_widths": [72, 72],
            "token_dim": 72,
            "encoder_heads": 4,
            "encoder_cross_attn_layers": 1,
            "encoder_self_attn_layers": 1,
            "decoder_cross_attn_layers": 1,
            "decoder_heads": 4,
            "decoder_mlp_widths": [144, 72],
            "head_mlp_widths": [144, 72],
            "dropout": 0.0,
        },
    ),
    "MATCH_250K_HIRES": _compat_preset(
        "MATCH_250K_HIRES",
        {
            "geom_coord_dim": 2,
            "query_coord_dim": 2,
            "geom_posenc_freqs": 6,
            "query_posenc_freqs": 6,
            "n_centroids": 256,
            "n_neighbors": 32,
            "local_mlp_widths": [72, 72],
            "token_dim": 72,
            "encoder_heads": 4,
            "encoder_cross_attn_layers": 1,
            "encoder_self_attn_layers": 1,
            "decoder_cross_attn_layers": 1,
            "decoder_heads": 4,
            "decoder_mlp_widths": [144, 72],
            "head_mlp_widths": [144, 72],
            "dropout": 0.0,
        },
    ),
}


def list_presets() -> List[str]:
    return sorted(PRESETS.keys())


def get_preset(name: str) -> Dict[str, Any]:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available presets: {', '.join(list_presets())}")
    return deepcopy(PRESETS[name])
