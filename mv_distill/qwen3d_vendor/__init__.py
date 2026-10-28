"""Qwen-3D's modified Qwen2.5-VL (github.com/ll220/qwen3d @ bf91573, Apache-2.0
via Qwen / HF transformers 4.51.1), vendored for `qwen_model.py`. Changes:

  modeling_qwen2_5_vl_modified.py  imports only: no ipdb, no unused
                                   PositionEmbeddingLearned, package-relative
  rope.py                          RotaryPositionEncoding4D returns cos / sin as
                                   [B, 1, N, d] (upstream [1, B, N, d] broadcasts
                                   the batch onto the heads unless B = 1)
"""
