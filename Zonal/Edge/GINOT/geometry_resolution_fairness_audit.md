# Geometry-Resolution Fairness Audit (Zonal/Edge)

## Scope
Read-only audit targets:
- `Zonal/Edge/PointNetMLPJoint`
- `Zonal/Edge/PointNetMLPJoint_FP`
- `Zonal/Edge/ArGEnT_self_att_noSDF`
- `Zonal/Edge/GINOT`

---

## A. Raw geometry input

### Common upstream data path
- All four scripts read the same zonal edge HDF5 file name (`disc_dataset_edge_deriv_zonal.h5`) and require `representation == "edge"`.
  - PointNet: `Zonal/Edge/PointNetMLPJoint/Training_script.py:33-34, 710-733`
  - PointNet FP: `Zonal/Edge/PointNetMLPJoint_FP/Training_script.py:33-34, 714-737`
  - ArGEnT-A: `Zonal/Edge/ArGEnT_self_att_noSDF/Training_script.py:34-35, 680-699`
  - GINOT-A: `Zonal/Edge/GINOT/Training_script.py:33-34, 733-758`
- All loaders build per-geometry tensors from raw boundary-node arrays (`node_coords_mm`, `zone_id`, optional edge derivatives, then targets at the end).
  - PointNet: `.../PointNetMLPJoint/Training_script.py:79-135`
  - PointNet FP: `.../PointNetMLPJoint_FP/Training_script.py:79-135`
  - ArGEnT-A: `.../ArGEnT_self_att_noSDF/Training_script.py:102-158`
  - GINOT-A: `.../GINOT/Training_script.py:80-136`

### Per-model input tensors and feature meaning
- **PointNetMLPJoint**
  - Encoder coords: `geom_xyz = t[:, [0,1]]` (x, r), query coords: `query_xy = t[:, [0,1]]`.
  - No extra encoder GF in this ablation (`EXTRA_FEAT_COLS=[]`).
  - Shapes per geometry item: `geom_xyz [N,2]`, `points [N,2]`, `target [N,2]`.
  - Source: `.../PointNetMLPJoint/Training_script.py:31-33, 171-177, 193-197`.
- **PointNetMLPJoint_FP**
  - Same as PointNetMLPJoint for this ablation (`EXTRA_FEAT_COLS=[]`), so encoder and query are both (x, r).
  - Shapes per geometry item: `geom_xyz [N,2]`, `points [N,2]`, `target [N,2]`.
  - Source: `.../PointNetMLPJoint_FP/Training_script.py:31-33, 170-177, 192-196`.
- **ArGEnT_self_att_noSDF**
  - Encoder features are selected by `INPUT_COLS=[0,1]` (x, r), query is also (x, r).
  - Shapes per geometry item: `enc_feats [N,2]`, `points [N,2]`, `target [N,2]`.
  - Source: `.../ArGEnT_self_att_noSDF/Training_script.py:33, 200-203, 215-217`.
- **GINOT-A**
  - Encoder coords: `geom_xyz = t[:, [0,1]]`, query coords: `query_xy = t[:, [0,1]]`.
  - No extra encoder GF in this ablation (`EXTRA_FEAT_COLS=[]`).
  - Shapes per geometry item: `geom_xyz [N,2]`, `points [N,2]`, `target [N,2]`.
  - Source: `.../GINOT/Training_script.py:32, 172-179, 193-197`.

### Whether all valid boundary nodes enter initially / pre-subsampling
- In all four pipelines, dataset items are created from all nodes in each geometry tensor before model-side processing.
- Default training mode in all four scripts is `train_mode = "batched_all"`, not sampled mode.
  - PointNet: `.../PointNetMLPJoint/Training_script.py:895`
  - PointNet FP: `.../PointNetMLPJoint_FP/Training_script.py:894`
  - ArGEnT-A: `.../ArGEnT_self_att_noSDF/Training_script.py:793`
  - GINOT-A: `.../GINOT/Training_script.py:886`

### Padding behavior and masks
- **PointNetMLPJoint / PointNetMLPJoint_FP / GINOT-A**: batch padding in `batched_all` repeats valid indices (not zero coordinate padding), with mask indicating real vs repeated slots.
  - PointNet: `.../PointNetMLPJoint/Training_script.py:294-342`
  - PointNet FP: `.../PointNetMLPJoint_FP/Training_script.py:292-340`
  - GINOT-A: `.../GINOT/Training_script.py:294-342`
- **ArGEnT-A**: `batched_all` pads with zeros and provides boolean masks (`mask`, `kv_mask`) for valid query/geometry positions.
  - Source: `.../ArGEnT_self_att_noSDF/Training_script.py:306-352`

### GF handling (feature-augmented counterpart relevance)
- In this audited `Zonal/Edge` path, all four use coordinate-only encoder/query features (no extra GF columns enabled).
- For the feature-augmented GINOT counterpart (`Zonal/Edge_arc_feat/GINOT`), policy is: encoder receives coords + 5 GF descriptors; head receives aligned 5 query GF; FPS/grouping still coordinate-only.
  - Source: `Zonal/Edge_arc_feat/GINOT/Training_script.py:31-32, 750-756, 803-804, 1044-1048`; coordinate-only grouping/FPS in `Zonal/Edge_arc_feat/GINOT/pn_models.py:351-364`.

---

## B. Geometry encoder resolution

### PointNetMLPJoint
- Encoder class: `PointNet2Encoder2D` with `SetAbstraction` blocks.
  - Source: `.../PointNetMLPJoint/pn_models.py:237-390`.
- Downsampling method: **none for configured presets** because SA block `n_samples=0` and code path treats `n_samples<=0` as “all points are centers”.
  - Logic: `.../PointNetMLPJoint/pn_models.py:171-177`.
  - Config evidence: `.../PointNetMLPJoint/model_presets.json` uses `n_samples: 0` broadly, including active `S_full_ln_pos12`: `2587-2631`.
- Local aggregation: each center applies radius/`max_k` neighborhood pooling.
  - Source: `.../PointNetMLPJoint/pn_models.py:182-210`.
- Global pooling: max pool to one latent vector.
  - Source: `.../PointNetMLPJoint/pn_models.py:228-234`.
- Static/configurable/data-dependent:
  - `n_samples`, `max_k`, radii are configurable via preset.
  - Effective neighbors can be data-dependent due to radius/mask, but center count here is full `N` due to `n_samples=0`.
- Contribution of original nodes:
  - All nodes appear as centers (no encoder center subsampling).
  - All nodes can contribute through local neighborhoods and then through global pooling.
  - Decoder receives only pooled latent (not per-node native-resolution features).

### PointNetMLPJoint_FP
- Same PointNet2 SA behavior with `n_samples<=0 => S=N` in `SetAbstraction`.
  - Source: `.../PointNetMLPJoint_FP/pn_models.py:171-177`.
- Active FP preset also sets `n_samples: 0` in both SA blocks (`S_full_ln_pos12_fp`).
  - Source: `.../PointNetMLPJoint_FP/model_presets.json:2783-2797`.
- Additional feature propagation path restores dense per-node features from SA2→SA1→original resolution.
  - Source: `.../PointNetMLPJoint_FP/pn_models.py:635-699`.
- Contribution of original nodes:
  - All nodes are encoder centers.
  - All nodes get native-resolution propagated FP features (`fp1_feats` at original `N`).
  - Decoder receives both per-query node-local FP features and global latent context.

### ArGEnT-A (`ArGEnT_self_att_noSDF` folder, actual instantiated mode is cross-attention)
- Model built as `ArGEnTDeepONet(attention_type="cross", use_sdf=False, in_ch_geom=len(INPUT_COLS)=2)`.
  - Source: `.../ArGEnT_self_att_noSDF/Training_script.py:855-864`.
- No internal FPS/token downsampling in `ArGEnTDeepONet` trunk itself; cross-attention keys/values are projected from provided `geom_points`.
  - Source: `.../ArGEnT_self_att_noSDF/benchmarks.py:805-817`.
- In the active `batched_all` mode, encoder/query sets are full-node padded batches, with mask/kv_mask.
  - Source: `.../ArGEnT_self_att_noSDF/Training_script.py:793, 824-846, 529, 537-540`.
- In optional sampled mode only, `k_enc`/`k_q=8192` would cap both sets.
  - Source: `.../ArGEnT_self_att_noSDF/Training_script.py:807-822`.
- Contribution of original nodes:
  - In configured mode, all nodes are available as geometry K/V and queries (with mask on padded slots).

### GINOT-A
- Encoder performs masked FPS to fixed centroid count, then coordinate-only kNN grouping around centroids.
  - FPS: `.../GINOT/pn_models.py:351-356`
  - Grouping: `.../GINOT/pn_models.py:359-364`
- Local token extraction over grouped neighborhoods and encoder attention stacks (cross then self).
  - Source: `.../GINOT/pn_models.py:393-409`.
- Decoder receives fixed token set and performs query→geometry cross-attention.
  - Source: `.../GINOT/pn_models.py:434-440`.
- Configured counts are static per preset (`n_centroids`, `n_neighbors` in preset config).
  - Source: `.../GINOT/model_presets.py:41-42, 62-63, 83-84, 104-105, 125-126, 146-147, 167-168`.
- Contribution of original nodes:
  - All valid nodes can contribute to grouped neighborhoods and to `global_tokens` (computed on all valid points).
    - `global_tokens`: `.../GINOT/pn_models.py:400-404`
  - Final decoder-accessible set is centroid-token count `N_t = n_centroids`, not full native node count `N`.

---

## C. Query handling

### Training query sampling
- All four scripts default to `train_mode="batched_all"`, where query set per geometry is all nodes, batch-padded with mask.
  - PointNet: `.../PointNetMLPJoint/Training_script.py:895-939`
  - PointNet FP: `.../PointNetMLPJoint_FP/Training_script.py:894-938`
  - ArGEnT-A: `.../ArGEnT_self_att_noSDF/Training_script.py:793-837`
  - GINOT-A: `.../GINOT/Training_script.py:886-930`
- Optional sampled mode exists in each script (`k_enc=k_q=8192`) but is not the configured default.

### Validation query coverage
- All four validation loaders use `batch_size=1` with variable-size collate and evaluate each geometry’s full `N` points.
  - PointNet: `.../PointNetMLPJoint/Training_script.py:940-948, 565-606`
  - PointNet FP: `.../PointNetMLPJoint_FP/Training_script.py:939-947, 566-609`
  - ArGEnT-A: `.../ArGEnT_self_att_noSDF/Training_script.py:838-846, 568-596`
  - GINOT-A: `.../GINOT/Training_script.py:931-939, 558-599`
- Held-out geometry split is aligned (`train_test_split(..., test_size=0.2, random_state=42)`) in all four.
  - PointNet: `.../PointNetMLPJoint/Training_script.py:839`
  - PointNet FP: `.../PointNetMLPJoint_FP/Training_script.py:838`
  - ArGEnT-A: `.../ArGEnT_self_att_noSDF/Training_script.py:768`
  - GINOT-A: `.../GINOT/Training_script.py:850`

**Result:** validation prediction coverage is shared and full-node across models; no model is validated on sampled-subset-only predictions under current settings.

---

## D. Decoder-accessible geometry information

- **PointNetMLPJoint**: global pooled geometry latent only (`z`) + query encoding in head; no native per-node decoder feature path.
  - Source: `.../PointNetMLPJoint/pn_models.py:389-390, 497-515`.
- **PointNetMLPJoint_FP**: per-query local FP-derived node features + global latent `z` + query encoding.
  - Source: `.../PointNetMLPJoint_FP/pn_models.py:698-713`.
- **ArGEnT-A**: fixed full geometry sequence used as K/V in cross-attention trunk (in configured mode), with query tokens as Q.
  - Source: `.../ArGEnT_self_att_noSDF/benchmarks.py:805-817`.
- **GINOT-A**: fixed-size FPS-derived geometry token set; decoder uses query→token cross-attention.
  - Source: `.../GINOT/pn_models.py:351-364, 434-440`.

---

## E. Complexity (conceptual)

Let:
- \(N\) = valid input boundary nodes
- \(N_t\) = geometry token count available to decoder
- \(N_q\) = query-node count

### PointNetMLPJoint
- SA with `n_samples=0` keeps center count near \(N\); each center aggregates up to `max_k` neighbors, so neighborhood aggregation scales about \(O(N \cdot K)\) with capped \(K\).
- Global pooling is \(O(N)\).
- Head scales roughly \(O(N_q)\) for per-query MLP once latent is formed.

### PointNetMLPJoint_FP
- Same SA-stage behavior as above plus feature propagation interpolation (`cdist`/kNN-style interpolation) and query-to-node nearest lookup.
  - `FeaturePropagation` uses `torch.cdist(xyz_dense, xyz_sparse)`; query-node lookup also uses `torch.cdist(query_points, geom_xyz)`.
  - Source: `.../PointNetMLPJoint_FP/pn_models.py:505, 703`.
- So dense node/query interactions include \(O(N\cdot M)\) and \(O(N_q\cdot N)\) style terms.

### ArGEnT-A
- Cross-attention trunk uses query length \(N_q\) against geometry length \(N\) in configured mode, per attention layer, so attention-like work scales about \(O(N_q N)\).
  - Source: `.../ArGEnT_self_att_noSDF/benchmarks.py:805-817`.

### GINOT-A
- Encoder tokenization does local neighborhood processing over centroid groups (roughly \(O(N_t \cdot K)\) local aggregation plus attention stack costs on \(N_t\) tokens).
- Decoder cross-attention cost is approximately \(O(N_q N_t)\) because each query attends over \(N_t\) geometry tokens.
- If every boundary node were exposed as an attention token instead, this would become approximately \(O(N_q N)\), typically much larger when \(N \gg N_t\).

---

## F. Fairness conclusion

1. **Same raw data fairness condition:**
   Fair comparison requires shared raw boundary-node geometry input, shared split, and shared validation query coverage. In this repository configuration, that condition is satisfied (same H5 representation checks, same split seed/fraction, full-node validation coverage).

2. **Legitimate architectural difference:**
   Different internal representations are acceptable: LC-PointNet can retain/propagate native-resolution local features; GINOT-A can summarize into FPS-derived tokens; ArGEnT-A can use its own attention geometry/query policy.

3. **Key risk identified:**
   If GINOT token count is too small, localized bore/web/rim/C-groove/drive-arm detail can be under-resolved and low-life prediction can degrade for token-resolution reasons rather than core architecture quality.

4. **Required classification for current MATCH_250K tokenization:**
   **Uncertain** (not clearly too low, not clearly proven adequate) for paper-level claims on subtle localized effects without a higher-resolution sensitivity run.

5. **Should MATCH_250K_HIRES be trained before paper-level claim?**
   **Yes.** A higher token-resolution sensitivity check is warranted before making final paper-level conclusions about GINOT-A in this benchmark.

