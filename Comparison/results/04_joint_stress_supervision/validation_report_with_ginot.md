# Joint stress supervision validation report

Repository commit: `cfca80fc56caf8f51553030c6f394e8254175e73`

Evaluation label: **validation-split evaluation**.

Checkpoint pairing now uses verified metadata instead of the previous brittle `fair_families` rule.
The prior empty-summary failure was caused by the old pairing pipeline treating missing checkpoint split metadata as `NaN` rather than missing and by relying on hard-coded family pairing assumptions.

## Valid stress/no-stress model pair(s)
| model_family          | fp_status   | stress_variant               | no_stress_variant            | stress_checkpoint_path                                                                                                      | no_stress_checkpoint_path                                                                                                             |
|:----------------------|:------------|:-----------------------------|:-----------------------------|:----------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------|
| ArGEnT_self_att_noSDF | non-FP      | argent_self_nosdf_s_177b264d | argent_self_nosdf_s_cf025919 | c:\Users\abfat\Desktop\Disc_lifing_paper_v2\Zonal\Edge\ArGEnT_self_att_noSDF\Trained_models\argent_self_nosdf_s_177b264d.pt | c:\Users\abfat\Desktop\Disc_lifing_paper_v2\Zonal\Edge_no_stress\ArGEnT_self_att_noSDF\Trained_models\argent_self_nosdf_s_cf025919.pt |
| GINOT-A               | non-FP      | ginot_a_match_250k_35ccca67  | ginot_a_match_250k_25c02577  | c:\Users\abfat\Desktop\Disc_lifing_paper_v2\Zonal\Edge\GINOT\Trained_models\ginot_a_match_250k_35ccca67.pt                  | c:\Users\abfat\Desktop\Disc_lifing_paper_v2\Zonal\Edge_no_stress\GINOT\Trained_models\ginot_a_match_250k_25c02577.pt                  |
| PointNetMLPJoint_FP   | FP          | pnmlp_16932d84               | pnmlp_1bbaae12               | c:\Users\abfat\Desktop\Disc_lifing_paper_v2\Zonal\Edge\PointNetMLPJoint_FP\Trained_models\pnmlp_16932d84.pt                 | c:\Users\abfat\Desktop\Disc_lifing_paper_v2\Zonal\Edge_no_stress\PointNetMLPJoint_FP\Trained_models\pnmlp_1bbaae12.pt                 |

- No nonexistent regular `PointNetMLPJoint` no-stress model was included.
- Delta convention used everywhere: `delta = no_stress - stress`; positive means the stress-supervised model has lower error.

## Quantitative outcome
- Full test set MAE: is mixed / model-dependent (median Δ `+0.0008` decades).
- Full test set RMSE: is mixed / model-dependent (median Δ `+0.0007` decades).
- Critical short-life bin `log_life < 2`: suggests improvement from stress supervision (median Δ `+0.0052` decades).
