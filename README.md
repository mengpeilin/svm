# mv_distill — multi-view value distillation for `scan_object`

A dense per-chunk reward for GAP on RoboTwin `scan_object` (aloha-agilex), built in two stages:

1. **Teacher**: a state-only multi-view SVM discriminator `f(s)`. It sees the head and both wrist cameras plus the robot state, and is trained with BCE on success vs failure. Its logit `T_t = V(s_t)` is a state value.
2. **Student**: `V(s_t, track)` reads the same observation together with the **16-step 3D point track** the policy is about to execute (one track per head-camera patch). It predicts the teacher's value 16 frames ahead:
   - `V(s_t, 0) → T_t`
   - `V(s_t, track_t) → T_{t+16}`

   This gives a reward for a chunk *before* executing it:

```
r(s_t, chunk) = V(s_t, track_t) - V(s_t, 0)          ≈ T_{t+16} - T_t
```

This package contains the full pipeline of the best model from the scan_object ablations:
- data caches
- teacher training and labelling
- student training
- inference
- evaluation (RMSE and reverse acc)
- videos (absolute value, every 4 frames)

It also ships that model's checkpoints.

| (50 + 50 eval episodes, frames t ≥ 16, class-balanced) | value |
|---|---|
| RMSE of `r` vs the teacher's `g = T_{t+16} - T_t` (balanced std of `g` = 1.28) | **0.736** |
| reverse acc: `r(real track) > r(-track)` on chunks with `g > 0.5` | **0.844** |

## The recipe (what each ablation showed helps)

**Student architecture**
- Multi-view `track_fused`: the head DinoV2 patches fused with their tracks go through self-attention; the wrist patches are a cross-attention memory; all tokens use 3D RoPE in world metres.
- Removing the wrist cameras costs the most: RMSE 0.736 → 0.907.

**Absolute target with the "minus" loss**
- The loss is `MSE(V_s, T_t) + MSE(V_f - V_s, T_{t+16} - T_t)`.
- Predicting the delta directly is worse (RMSE 0.851, reverse acc 0.779).
- Using two absolute fits without the minus term is also worse (RMSE 0.822).

**Change-biased sampling**
- 75% of each half-batch is drawn from frames with `|T_{t+16} - T_t| > 0.5`.
- This mainly helps drop detection.

**Noise branches** (908 rollouts)
- Two per train success episode: replay the episode to a random chunk, add an xyz offset to that chunk, then let GAP continue.
- 609 of the branches succeeded and 299 failed.
- They help direction sensitivity: reverse acc 0.799 → 0.844.

**Training data**
- Students: scan 454 + 454 train episodes plus the 908 branches, every 4th frame (73,128 frames).
- Teacher: every 16th frame of the same episodes.
- The eval set is 50 + 50 held-out episodes. It is **also** used for checkpoint selection, so the eval numbers are mildly optimistic.

## Layout

```
README.md
run.sbatch                    run one CLI as a grogu-4-23 job (dheld partition)
checkpoints/
  teacher/teacher.pt          teacher (state-only, 90 M params), epoch 5, eval step AUC 0.912
  teacher/labels_scan.npz     its logit on every frame of "scan"   (227,864 frames)
  teacher/labels_branch.npz   its logit on every frame of "branch" (187,013 frames)
  student/student.pt          student (track_fused, 93 M params), epoch 60
mv_distill/
  config.py        paths (env vars), dataset layout, constants
  parts.py         reading collector parts: episode order, metadata, patch geometry
  dino.py          DinoV2 feature extraction (GAP's preprocessing)
  data.py          caches, splits, Normalizer, multi-view batches
  model.py         MultiViewDiscriminator (inputs "state" | "track_fused")
  student.py       student data: teacher labels, train / eval frames, V(s, track)
  metrics.py       class-balanced RMSE, reverse acc
  build_cache.py   CLI  head every-frame cache from the parts
  build_wrist.py   CLI  wrist-camera rows (init / fill / check)
  branch_plan.py   CLI  plan the noise branches for the collector
  train_teacher.py CLI
  label.py         CLI  teacher logit on every frame
  train_student.py CLI
  infer.py         CLI  student on every eval frame -> eval_frames.npz
  evaluate.py      CLI  RMSE + reverse acc -> metrics.json
  render.py        CLI  videos
```

Every CLI runs from this folder as `python -m mv_distill.<name>`. Each module's docstring has the details.

On grogu, run every step as a job, not on the login node. `run.sbatch` wraps one CLI:
- the `dheld` partition on `grogu-4-23`;
- the `gap` env;
- `MV_DISTILL_RAW` / `MV_DISTILL_WORK` defaulting to `/scratch/pmeng/mv_distill/{raw,work}`;
- logs in `/project_data/held/peilin/svm/logs/mv_distill/`.

```bash
sbatch run.sbatch mv_distill.train_student --teacher teacher --run student
sbatch -J infer run.sbatch mv_distill.infer --run student
```

## Setup

**Environment**
- Use the `gap` conda env: torch, numpy, h5py, opencv, PyAV.
- On grogu, `~/miniconda3/envs/gap` works on `grogu-4-23` (Blackwell, sm_120).
- The DinoV2 weights are loaded from the local torch hub. Both files already exist on grogu and locally:
  - `~/.cache/torch/hub/facebookresearch_dinov2_main`
  - `~/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`

**GPU memory** (bf16 autocast)
- Student training peaks at **24.5 GiB**. It only fits on `grogu-4-23`: the 3090s have 24 GB and `grogu-4-13` has 12 GB.
- The teacher, labelling, inference and wrist DinoV2 extraction are much smaller.

**Two environment variables** locate all data:

```bash
export MV_DISTILL_RAW=/scratch/pmeng/mv_distill/raw    # collector parts (read)
export MV_DISTILL_WORK=/scratch/pmeng/mv_distill/work  # caches, runs, plans (written)
```

```
$MV_DISTILL_RAW/scan/scan_wdepth-*.hdf5     scan_object rollouts: 504 success + 504 failure, 327 GB
$MV_DISTILL_RAW/branch/scan_branch-*.hdf5   their 908 noise branches, 267 GB
$MV_DISTILL_WORK/cache/{scan,branch}/       memmapped inputs: 291 GB + 218 GB (incl. wrist rows 111 + 70 GB)
$MV_DISTILL_WORK/runs/<run>/                checkpoints, labels, eval frames, metrics, videos
$MV_DISTILL_WORK/branch_plans/<name>.npz
```

**On grogu, raw data and caches go on node-local `/scratch`, never on `/project_data`.** Training reads the memmaps at random, which is exactly the I/O pattern the shared NAS must not see. Pin jobs to the node holding the data with `--nodelist`, and always pass `--partition=dheld`.

Raw-data requirements:
- The parts must record **all three cameras with RGB and depth**: `image`, `depth`, `left_wrist_*`, `right_wrist_*`, plus their intrinsics and extrinsics.
- That is what `collect_svm_rollouts.py --wrist` writes.
- The local copies are under `/data/peilin/reward_model/data/tracks/svm_scan_object_wdepth/` and `.../svm_scan_object_branch/mv/`.

## Quick start: score the shipped student

```bash
cd /project_data/held/peilin/svm/mv_distill
mkdir -p $MV_DISTILL_WORK/runs && cp -r checkpoints/teacher checkpoints/student $MV_DISTILL_WORK/runs/

python -m mv_distill.build_cache --dataset scan
python -m mv_distill.build_wrist init --dataset scan
python -m mv_distill.build_wrist fill --dataset scan --shard 0 --shards 2 --device cuda:0 &
python -m mv_distill.build_wrist fill --dataset scan --shard 1 --shards 2 --device cuda:1 &
wait
python -m mv_distill.build_wrist check --dataset scan --shards 2

python -m mv_distill.infer    --run student      # ~6 min: V(s,0), V(s,track), V(s,-track) on 19,390 eval frames
python -m mv_distill.evaluate --run student      # RMSE 0.736, reverse acc 0.844
python -m mv_distill.render   --run student --n 5 --seed 1
```

- The wrist rows of both datasets took about 50 min in total on 2 GPUs.
- `fill` first re-extracts a sample of head frames and stops if they do not match the features the collector stored. This catches wrong DinoV2 weights or preprocessing.

## Full pipeline

| step | command | notes |
|---|---|---|
| 1. collect scan rollouts | on grogu: `scripts/collect_scan_object_wdepth.sbatch` (in `/project_data/held/peilin/svm/scripts/`) | GAP-JOPAT `scan_object_vae` ckpt 400, `--wrist`, step limit 300; `grogu-4-13`, one worker per GPU, `dp` env |
| 2. scan cache | `build_cache --dataset scan`; `build_wrist init/fill/check --dataset scan` | |
| 3. branch plan | `python -m mv_distill.branch_plan --name train --k 0,1 --sigmas 2,4,8` | 2 branches per train success, σ ∈ {2, 4, 8} cm |
| 4. collect branches | copy `branch_plans/train.npz` to `/project_data/held/peilin/svm/branch_plans/`, then `sbatch scripts/collect_scan_object_branch.sbatch train` | ~4 min per branch per worker |
| 5. branch cache | `build_cache --dataset branch`; `build_wrist init/fill/check --dataset branch` | |
| 6. teacher | `python -m mv_distill.train_teacher --run teacher` | 50 steps/epoch, stops at patience 5 (the shipped one: best epoch 5, ~11 epochs) |
| 7. labels | `python -m mv_distill.label --teacher teacher --dataset scan` (then `--dataset branch`) | every frame from the parts; ~30 min for scan |
| 8. student | `python -m mv_distill.train_student --teacher teacher --run student` | ~190 s/epoch on an A6000; the shipped one: best epoch 60 of 71 (~3.8 h) |
| 9. infer / evaluate / render | as in the quick start | |

Notes:
- Steps 6–8 each have a `--smoke` flag: 3 steps on a small eval subset, then exit. It checks data, memory and code in about a minute.
- Every reported number uses the **shipped teacher's labels** as GT. A retrained teacher gives different labels, so its students are scored against a different GT.

## Outputs

**`runs/<student>/eval_frames.npz`** holds every eval sample frame: each eval episode's frames with a full chunk ahead, with failures capped at frame 250.

| key | meaning |
|---|---|
| `frames`, `episode`, `t`, `label` | cache frame, scan episode, frame in episode, 1 = success / 0 = failure |
| `T`, `T_next` | teacher logit `T_t` and `T_{t+16}` (the last frame's when the episode ends inside the chunk) |
| `static`, `forward`, `mirrored` | `V(s_t, 0)`, `V(s_t, track_t)`, `V(s_t, -track_t)` |

From these:
- the reward is `forward - static`, with GT `T_next - T`;
- the forecast of the value 16 frames ahead is `forward`, with GT `T_next`.

**`runs/<student>/metrics.json`** has `rmse`, `reverse_acc`, `balanced_std_g` and the frame counts. All are computed on t ≥ 16, with success and failure frames weighing 1/2 each; failure episodes hold about 2/3 of the frames.

- **RMSE** is `sqrt(MSE(r, g))` in teacher-logit units.
- **reverse acc** is computed on the chunks the teacher says make progress (`g > 0.5`). It is the share where the reward with the real track beats the reward with the reversed track (same observation). 0.5 means blind to the direction of the motion.

**`runs/<student>/videos/{success|failure}_ep<ep>.mp4`**
- The cameras: head at 2x, both wrists on the right.
- Below them, a chart of the teacher's `T_{t+16}` (orange) vs the student's forecast `V(s_t, track_t)` (blue).
- A point is drawn every `--step` (4) frames at the decision frame and held for 4 frames.
- The title gives the episode's MSE (t ≥ 16).
- Episodes are drawn at random with `--seed`; `--exclude` skips some.

## Checkpoint format

`teacher.pt` / `student.pt` are `torch.save` dicts:

| key | content |
|---|---|
| `model` | `state_dict` |
| `config` | `MultiViewDiscriminator` kwargs (`inputs`, `n_layers`, `dropout`, `rope_spatial_scale`) |
| `normalizer` | `state_lo` / `state_hi` (robot state min / max) and `track_rms` |
| `epoch`, `args` | training metadata |
| `teacher` (student only) | name of the teacher's run dir; its labels are the student's targets |

`model.load(path, device)` rebuilds the model in eval mode.

Batch keys are documented in `MultiViewDiscriminator` and built by `data.Frames.inputs`:
- every model: `dino [B,3,391,1024]`, `centres [B,3,391,3]`, `centre_ok [B,3,391]`, `state [B,16]`, `state_xyz [B,2,3]`;
- the student also takes `track [B,391,16,3]` and `valid [B,391]`.

Inference runs under bf16 autocast, as in training.

## Provenance

This package is a cleaned port of two pieces of code in `/data/peilin/reward_model/DenseValue/`:
- `Distill/`: the student, run `abl_r3b_mv_s0`;
- `SVM_MultiView/`: the teacher, run `teacher_mv_state_aug`.

The old 3-view decision-frame cache is gone: the teacher reads the same head cache and wrist rows as the student, which give identical inputs on the decision frames.

Checked against the original code on the local data:
- train / eval frames, batches, the normalizer and the branch plan are bit-identical;
- the shipped student reproduces RMSE 0.736 and reverse acc 0.844 exactly;
- `label` reproduces the shipped labels exactly;
- `build_cache` reproduces the cache;
- a retrained teacher picks the same epoch (5), with eval AUC 0.928 vs 0.912;
- the student's first epochs match the original run's losses.
