# Source provenance

The source was exported from the user's `aqr_relation_query` project on 2026-09-29. The selected experiment is `aqr_dp3_stackcube_a6_post_h8_anchor_b16_gpu0_20260801_203534`, associated with `selected_epoch77.pt`. No checkpoints were copied.

The original directory had no `.git` history. This is a source snapshot of the corresponding implementation branch, not an authenticated reconstruction of the exact historical source revision.

Packaging changes: flattened A6_POST config; batch size 16 for the selected variant; portable output and provenance paths; extracted runtime point sampling and URDF loading; removed data preparation, unrelated experiment scripts and unused DP3 modules; corrected one lightweight test fixture's missing device anchor. The AQR algorithm was retained.

`diffusion_policy_3d` is inherited third-party DP3 code from the supplied project. Existing source attribution is retained; this snapshot does not assign a new license to that code or to the user's research implementation.
