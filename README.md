# Multi-view SVM reward models (scan_object)

A dense process reward for RL fine-tuning, following *Learning Process Rewards via
Success Visitation Matching (SVM)*: a discriminator `f(s, a)` trained to tell
decision steps of successful episodes from those of failed ones. Its logit is
`log f/(1-f)`, and the dense reward is

```
r(s, a) = λ · clip(logit, -β, β)      λ = 0.1, β = 20   (+ outcome reward r_out, added by your RL code)
```

In both models `s` is the **head + left-wrist + right-wrist camera** observation
plus the robot state. They differ in `a`:

| model | `a` | checkpoint |
|---|---|---|
| `MVTrackSVMReward` | **3D point track** of the next 16 steps (how each head-camera patch's 3D point moves) | https://huggingface.co/Mercury007/mv-track-svm-scan-object |
| `MVActionSVMReward` | **16-step commanded EEF pose chunk** (both arms) | https://huggingface.co/Mercury007/mv-action-svm-scan-object |

Both are trained on the same RoboTwin `scan_object` (aloha-agilex) rollouts of
the GAP policy and score equally well (see below). The action model needs no
tracks, so it is the easier one to plug into an RL loop.

## Install

```bash
pip install -r requirements.txt     # torch, numpy, opencv-python, safetensors, huggingface_hub
```

DINOv2 (ViT-L/14 with registers) is loaded from `torch.hub` (`facebookresearch/dinov2`)
the first time; if you already run a GAP policy, pass its DINOv2 model in instead.

## Quick start

```python
from svm import MVActionSVMReward, MVTrackSVMReward, svm_reward

scorer = MVActionSVMReward.from_pretrained("Mercury007/mv-action-svm-scan-object")
# or MVTrackSVMReward.from_pretrained("Mercury007/mv-track-svm-scan-object")
# or MVActionSVMReward("/path/to/local/ckpt_dir", dino=my_dinov2_model)

logits = scorer.logits(obs)          # [B]   log f/(1-f)
reward = svm_reward(logits)          # [B]   0.1 * clip(logit, -20, 20)
```

Score a whole collector HDF5 part (reference usage and a format check):

```bash
python examples/score_episode.py --part scan_wdepth-w7-part0004.hdf5 \
    --model action --ckpt Mercury007/mv-action-svm-scan-object
python examples/score_episode.py --part scan_wdepth-w7-part0004.hdf5 \
    --model track --ckpt Mercury007/mv-track-svm-scan-object
```

## Inputs

`obs` is a dict of numpy arrays, each with a leading batch dim `B` (one entry per
decision step). For `cam` in `head`, `left_wrist`, `right_wrist`:

| key | shape | content |
|---|---|---|
| `{cam}_rgb` | `[B, 240, 320, 3]` uint8 | RGB image |
| `{cam}_depth` | `[B, 240, 320]` float | depth in **millimetres**, 0 where nothing is rendered |
| `{cam}_K` | `[B, 3, 3]` | pinhole intrinsic |
| `{cam}_E` | `[B, 3, 4]` | extrinsic **world → camera**, OpenCV convention (`x_cam = R x_world + t`) |
| `state` | `[B, 16]` | `[L xyz, L quat wxyz, L gripper, R xyz, R quat wxyz, R gripper]`, world frame (RoboTwin `endpose`) |
| `action` | `[B, 16, 16]` | **action model:** the 16 commanded poses of the chunk, each in the `state` format (the collector's `action`) |
| `track` | `[B, 391, 16, 3]` | **track model:** displacement (m, world frame) of each head patch's 3D point from frame `t` to `t+h+1`, `h = 0..15` |
| `track_valid` | `[B, 391]` bool | **track model:** patches whose point is tracked (depth-valid and on a scene object) |

Optional `head_dino=[B, 391, 1024]` reuses head DINOv2 patch features the policy already computed.

**Patch grid.** 391 = 17 × 23 DINOv2 patches: the 240×320 image is resized
(bilinear) to 238×322 and cut into 14-px patches, row-major. Patch `i` of the
track corresponds to DINOv2 token `i` of the head image; its 3D anchor is the
head depth at the patch centre (`svm/preprocess.py: PATCH_PX`) unprojected to
the world. The same is done for each wrist camera with its own depth, giving
every wrist patch a world-frame 3D position for the RoPE.

**One sample = one policy decision.** GAP executes all 16 actions of a chunk
before observing again, so the model scores the observation at the first frame
of each chunk (frames `0, 16, 32, …` of an episode) together with the 16-step
action chunk / track of that chunk, and you get **one reward per chunk**. If an
episode ends mid-chunk, the chunk holds the last action and the track the last
frame's positions for the remaining steps.

**Tracks.** Training used simulator ground-truth tracks of the executed chunk
(object poses replayed in SAPIEN; robot and background points are invalid).
Feeding predicted tracks (e.g. GAP's track head) is a distribution shift: check
that the score separates success/failure on your data before relying on it.

Normalization (state / action min/max → [-1, 1], tracks / RMS) is stored in
`config.json` and applied inside `MVTrackSVMReward`; pass everything raw.

## Model

`svm/model.py`, 6 pre-norm layers, d_model 1024, 8 heads; `MultiViewTrackSVM`
90.7 M parameters, `MultiViewActionSVM` 91.3 M.

```
self-attention   track:  [CLS] | state L, R | head DINO ×391 | track ×391         (785 tokens)
                 action: [CLS] | state L, R | action L, R | head DINO ×391        (396 tokens)
cross-attention  left-wrist DINO ×391 | right-wrist DINO ×391 (memory, K/V only)  (782 tokens)
readout          CLS → LayerNorm → MLP(256, 256) → logit
```

- Action token (one per arm) = `MLP(flatten(16×8 normalized poses))` + arm embedding, at the 3D coordinate of that arm's first commanded xyz.
- Track token = `MLP(flatten(16×3 track))` + learned type embedding (+ invalid embedding where `track_valid` is false), sitting at the same grid position / 3D coordinate as its head patch.
- Every token carries a **3D RoPE** over world-frame metres (scale 1 cm ≈ 1 rad): state tokens at their arm's EEF xyz, action tokens at their first commanded xyz, head DINO and track tokens at the head patch centre, wrist patches at their unprojected world xyz. In cross-attention q and k are rotated by their own coordinates, so attention depends on the 3D offset between a head-side token and a wrist patch. CLS and depth-invalid patches are not rotated.
- Plus a 2D sin-cos patch-grid encoding (and a per-wrist view embedding on the memory) added to Q/K.

## Training data and results

- 504 success + 504 failure GAP rollouts of `scan_object` (step limit 300), with wrist depth recorded; failure episodes capped at their first 250 frames.
- Split per class: 454 train, last 50 seeds held out (used for both checkpoint selection and the numbers below, so they are mildly optimistic).
- BCE, AdamW 1e-4, wd 1e-6, batch 64 balanced, bf16; selected by val loss
  (track at epoch 1, action at epoch 0; both overfit fast after that).

| held-out metric (50 + 50 episodes) | track | action |
|---|---|---|
| per-step AUC (success vs failure steps) | 0.923 | 0.917 |
| episode ranking accuracy | 0.835 | 0.814 |
| first-step AUC | 0.63 | 0.61 |
| mean reward, success / failure episodes | +0.227 / −0.407 | +0.141 / −0.314 |
| fraction of steps hitting the ±β clip | 0.0 | 0.0 |

Known limitations:

- The first decision step barely separates the classes since nothing has
  happened yet; and some failure episodes are scored like successes throughout
  (e.g. held-out failure seed 870187: track logit > 2 for 17 of its 19 steps;
  the action model also scores it positive, peaking at 2.4, but drops toward 0
  at the end).
- **The action model barely reads its action input.** The training rollouts
  come from one near-deterministic policy, so each state comes with
  essentially one action and `f(s, a)` cannot learn action dependence: across
  32 GAP candidate chunks for the same observation the logit varies by ~0.004,
  and 10 cm offsets or chunks from other states move it by ~0.03, vs ~3 across
  states. Treat it as a state(-visitation) value: fine as a dense RL reward,
  **useless for ranking candidate actions at test time**. The track model
  likewise uses the track mostly as a motion-magnitude cue.

## Files

```
svm/model.py          MultiViewTrackSVM, MultiViewActionSVM (networks only)
svm/rope3d.py         3D RoPE
svm/preprocess.py     patch centres from depth, DINOv2 features, normalization, batch assembly
svm/reward_model.py   MVTrackSVMReward, MVActionSVMReward: load checkpoint, obs -> logits / rewards
examples/score_episode.py
```

Each Hugging Face repo holds `model.safetensors` (weights) and `config.json`
(model hyper-parameters, normalizer statistics, training metadata).
