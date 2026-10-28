# mv_distill — multi-view value distillation, cotrained over the SVM tasks

A dense per-chunk reward for GAP on RoboTwin (aloha-agilex), built in two stages:

1. **Teachers, one per task**: a state-only multi-view SVM discriminator `f(s)`. It sees the head and both wrist cameras plus the robot state, and is trained with BCE on that task's success vs failure rollouts. Its logit `T_t = V(s_t)` is a state value.
2. **One student, cotrained on all tasks**: `V(s_t, track)` reads the observation together with the **16-step 3D point track** the policy is about to execute (one track per head-camera patch). It predicts the frame's task teacher's value 16 frames ahead:
   - `V(s_t, 0) → T_t`
   - `V(s_t, track_t) → T_{t+16}`

   This gives a reward for a chunk *before* executing it:

```
r(s_t, chunk) = V(s_t, track_t) - V(s_t, 0)          ≈ T_{t+16} - T_t
```

Each task's targets are in its own teacher's logit units. There are two student architectures:

| `--arch` | student | task input |
|---|---|---|
| `dino` | the DinoV2 multi-view `track_fused` transformer, from scratch (`model.py`) | none: the scene tells the tasks apart |
| `qwen3d` | [Qwen-3D](https://github.com/ll220/qwen3d) (Qwen2.5-VL-3B pretrained on 3D grounding / QA) finetuned with a fresh LoRA (`qwen_model.py`) | the task's language instruction |

The `dino` recipe is the best single-task model of the scan_object ablations (commit `5b1b0d1` of this repo, the single-task code; its checkpoints are in `checkpoints/`). On scan_object it scored RMSE 0.719 (holding `T_t`: 1.28) and reverse acc 0.844. The first 10-task cotrain (commit `80cc7fd`: base data only, teachers `teacher_<task>`, student `runs/student`) scored pooled RMSE 1.076 (holding `T_t`: 1.01) and reverse acc 0.794.

## Data

vxiao's SVM rollouts (`/project_data/held/vxiao/Bimanual_Manipulation_Project/GAP-tte-flattenmlp-concatdit-actionrope/svm_rollout/`), staged by `/project_data/held/peilin/svm/scripts/fm_stage.py` to node-local `grogu-4-23:/scratch/pmeng/`, uncompressed `.npy`: RGB, depth, intrinsics / extrinsics of all three cameras, GT tracks, robot state, and DinoV2-L/14-reg4 patch features of all three cameras (fp32 compute, stored f16). One rollout per seed; failures truncated to the task's p90 success length.

`MV_DISTILL_DATA` lists the staged roots, base first:

| root | content | split |
|---|---|---|
| `svm_fm` (base) | 250 success + 250 failure episodes per task, 1.36 M frames | per task and class, the last 25 episodes (highest seeds) are eval; the rest train |
| `svm_fm_rest` | every other unique-seed rollout (`fm_stage.py plan --rest`), 0.91 M frames | all train |

The eval set is therefore the same as in the base-only cotrain, and it is **also** used for checkpoint selection, as in the single-task recipe.

| task | base frames | rest frames | | task | base frames | rest frames |
|---|---|---|---|---|---|---|
| handover_block | 142,992 | 96,592 | | place_cans_plasticbox | 120,128 | 71,392 |
| hanging_mug | 165,712 | 67,760 | | place_object_basket | 134,496 | 105,776 |
| pick_diverse_bottles | 61,424 | 58,048 | | stack_blocks_two | 155,104 | 87,424 |
| pick_dual_bottles | 61,536 | 42,192 | | stack_bowls_three | 245,008 | 204,784 |
| place_burger_fries | 117,056 | 81,696 | | stack_bowls_two | 155,616 | 93,312 |

The GT track `tracks_f32[:, p, j]` is head patch `p`'s displacement (m) after `j + 1` steps; only patches on objects and the robot have one (`valid`, 5–13% of the head patches). Patches at rest still carry the simulator's sub-0.1 mm jitter: exactly zero only in handover_block.

## The recipe

**Teacher** (`train_teacher`, per task, runs `teacher2_<task>`): BCE on decision frames (every 16th) of the task's train episodes; batches of 256, half success / half failure; 25 steps / epoch (the base-only teachers peaked within 3–29 k samples), lr 2e-4 (warmup + cosine over 60 epochs); checkpoint = lowest class-balanced eval BCE, patience 5.

**Student** (`train_student`, all tasks):
- Loss ("minus"): `MSE(V_s, T_t) + MSE(V_f - V_s, T_{t+16} - T_t)`.
- Data: every frame of the train episodes with a full chunk ahead.
- Batches: half success / half failure frames, each half drawn from that class's frames pooled over the tasks (so a task weighs by its frame count). 75% of each half comes from frames with `|T_{t+16} - T_t| > 0.5` (change-biased sampling).
- Normalizer (state min / max, one track RMS) fit on the train decision frames of all tasks.
- Checkpoint = lowest val loss (the same two terms) on every 16th eval frame.

| | `dino` | `qwen3d` |
|---|---|---|
| trainable | all 93 M | LoRA r16 / α32 (no dropout) on the LLM's q, k, v, o, gate, up, down + track encoder + value head (31.6 M); ViT and LLM frozen |
| batch / GPU | 192 frames (384 inputs) | 32 frames (64 inputs) |
| lr, schedule | 1e-4, 500 warmup steps, cosine, grad clip 1.0 (2e-4 without clip plateaued: train loss flat at ~5.2 for 4 k steps) | 1e-4, 500 warmup steps, cosine, grad clip 1.0 |
| epoch, stop | 1000 steps, ≤ 40 epochs, patience 10 | 500 steps, ≤ 50 epochs (~2 days on 2 GPUs), patience 5 |

**The `qwen3d` student** (`qwen_model.py`). One sequence per input:

1. **Observation tokens (t = 0)** from the frozen Qwen ViT, run online on the RGB:
   - head at 2x the DinoV2 input (476 x 644): 17 x 23 merged tokens, one per DinoV2 patch, at the patch centres;
   - wrists at native size (Qwen's smart_resize: 252 x 308): 9 x 11 tokens each, at their own centres (`build_centres` wrist_q);
   - only tokens above the tabletop are kept (`config.WORKSPACE`: x within +-0.6, y within +-0.35, z >= 0.73): the wall, the floor, the gap behind the table and the robot base are dropped;
   - voxel pooling as in Qwen-3D: tokens with a valid centre binned by world xyz into **2 cm** cells (`voxel_stats`: ~521 tokens / frame; at 3 cm 3% of the tracked patches would share a cell with the static background, at Qwen-3D's 5 cm 9%), features and xyz averaged.
2. **Future tokens**: for k = 4, 8, 12, 16, each tracked head patch in the workspace that has moved by step k (displacement > `config.MOVE_THRESH` = 1 mm at some step <= k; resting tracked points carry < 0.3 mm of sim jitter) re-enters at its moved position `centre + track[k-1]` and time t = k, as its ViT feature + `TrackEncoder(track up to step k)` (zero-initialized: it starts as a copy of the patch). A zero track moves nothing, so `V(s, 0)` sees the observation only.
3. **Language**: Qwen-3D's chat template around the points, `"Task: {full_description}. Estimate how likely the task is to succeed."` (RoboTwin's task_instruction). `V` = MLP(final hidden state of the last prompt token).

Positions are Qwen-3D's 4D RoPE `(t, x, y, z)` in raw world metres (text tokens `(t, 0, 0, 0)`, numbered as Qwen-3D does); attention is bidirectional. Rows are padded and masked, so unlike upstream (batch size 1) any batch size works; `qwen_check` verifies that against upstream.

Not carried over from scan_object: the noise branches (none were collected for these tasks).

## Layout

```
README.md
QWEN3D_PLAN.md      the plan of the Qwen-3D student
run.sbatch          run one CLI as a grogu-4-23 job (dheld partition); NPROC > 1: one process per GPU (DDP)
teachers.sbatch     train_teacher + label for each task given, one after another
checkpoints/        the single-task scan_object teacher / student (code: commit 5b1b0d1)
mv_distill/
  config.py         paths (env vars), tasks and their instructions, constants
  data.py           staged roots, splits, Normalizer, model batches (DinoV2 / Qwen)
  model.py          MultiViewDiscriminator (inputs "state" | "track_fused"); load() for both archs
  qwen_model.py     the Qwen-3D student
  qwen3d_vendor/    Qwen-3D's modified Qwen2.5-VL (github.com/ll220/qwen3d @ bf91573)
  student.py        student data: per-task teacher labels, train / eval frames, V(s, track)
  metrics.py        class-balanced RMSE, reverse acc
  batching.py       background prefetching
  build_centres.py  CLI  token centres from depth: DinoV2 patches (3 cameras), wrist Qwen tokens
  voxel_stats.py    CLI  token counts / merges per voxel size (chooses --voxel)
  qwen_check.py     CLI  the Qwen student vs upstream Qwen / Qwen-3D on real frames
  train_teacher.py  CLI  one task's teacher
  label.py          CLI  a teacher's logit on every frame of its task
  train_student.py  CLI
  infer.py          CLI  student on every eval frame -> eval_frames.npz
  evaluate.py       CLI  RMSE + reverse acc, pooled and per task -> metrics.json
  render.py         CLI  videos of one task
```

Every CLI runs from this folder as `python -m mv_distill.<name>`, with the `~/venvs/qwen3d` python (the `gap` env plus transformers 4.51.1 for the vendored Qwen-3D; `run.sbatch` uses it). Each module's docstring has the details.

Environment variables (`run.sbatch` sets them):

```bash
export MV_DISTILL_DATA=/scratch/pmeng/svm_fm:/scratch/pmeng/svm_fm_rest   # staged roots, base first (read only)
export MV_DISTILL_WORK=/scratch/pmeng/mv_distill/work                     # cache/<root>/<task>/, runs/<run>/
export WANDB_MODE=offline WANDB_DIR=/project_data/held/peilin/svm/logs    # wandb sync logs/wandb/offline-run-*
```

The Qwen weights are on `grogu-4-23:/scratch/pmeng/hf/`: `Qwen-3D/qwen3d_3b.pth` (HF `katefgroup/Qwen-3D`) and the config / tokenizer of `Qwen2.5-VL-3B-Instruct`.

**On grogu, data goes on node-local `/scratch`, never on `/project_data`.** Training reads the memmaps at random, which is exactly the I/O pattern the shared NAS must not see. Every job is pinned to `grogu-4-23`, the node holding the staged data, on the `dheld` partition.

## Pipeline

```bash
cd /project_data/held/peilin/svm/mv_distill
# 1. token centres (CPU): DinoV2 patches of all three cameras + wrist Qwen tokens, every root
sbatch -J centres --gres=gpu:0 -c 32 -t 6:00:00 run.sbatch mv_distill.build_centres build --workers 32
sbatch -J centres_check --gres=gpu:0 -c 4 -t 1:00:00 run.sbatch mv_distill.build_centres check
# 2. the 10 teachers and their labels (runs/teacher2_<task>/teacher.pt, labels_<task>.npz)
sbatch teachers.sbatch handover_block hanging_mug pick_diverse_bottles pick_dual_bottles place_burger_fries
sbatch teachers.sbatch place_cans_plasticbox place_object_basket stack_blocks_two stack_bowls_three stack_bowls_two
# 3. the cotrained students, then score them
sbatch -J student2 run.sbatch mv_distill.train_student --arch dino --run student2
NPROC=2 sbatch -J qwen3d --gres=gpu:2 -c 32 --mem=192G run.sbatch mv_distill.train_student --arch qwen3d \
    --run qwen3d --batch-size 32 --lr 1e-4 --clip-grad 1.0 --steps-per-epoch 500 --max-epochs 50 --patience 5
sbatch -J infer run.sbatch mv_distill.infer --run qwen3d --chunk 32
python -m mv_distill.evaluate --run qwen3d
sbatch -J render run.sbatch mv_distill.render --run qwen3d --task hanging_mug --n 5 --seed 1
```

- `train_teacher`, `train_student` have a `--smoke` flag: a few steps, a small eval subset, peak memory and s/step, no wandb.
- A single-task specialist student is `train_student --tasks <task> --run <name>`: same teacher, so it is scored on the same GT as the cotrained one.

## Outputs

**`runs/teacher2_<task>/`**: `teacher.pt`, `train_history.json`, `labels_<task>.npz` (`logit` on every frame of the task, its roots in `MV_DISTILL_DATA` order).

**`runs/<student>/eval_frames.npz`** holds every eval sample frame: each eval episode's frames with a full chunk ahead.

| key | meaning |
|---|---|
| `tasks` | task names; `task` indexes them |
| `frames`, `task`, `episode`, `t`, `label` | global frame, task, episode within the task (= its base episode), frame in episode, 1 = success / 0 = failure |
| `T`, `T_next` | the task teacher's logit `T_t` and `T_{t+16}` (the last frame's when the episode ends inside the chunk) |
| `static`, `forward`, `mirrored` | `V(s_t, 0)`, `V(s_t, track_t)`, `V(s_t, -track_t)` |

**`runs/<student>/metrics.json`**: `pooled` and per-task `tasks.<task>`, each with `rmse`, `reverse_acc`, two RMSE references (`balanced_std_T_next` = predicting the mean, `rmse_hold_T` = holding the teacher's `T_t`) and frame counts. All are on t ≥ 16, with success and failure frames weighing 1/2 each.
- **RMSE** is `sqrt(MSE(V(s_t, track_t), T_{t+16}))`, in the task teacher's logit units. Teachers differ in scale, so compare a task's RMSE with its own references.
- **reverse acc** is computed on the chunks the teacher says make progress (`g = T_{t+16} - T_t > 0.5`): the share where the forecast with the real track beats the forecast with the reversed track (same observation). 0.5 means blind to the direction of the motion.

**`runs/<student>/videos/<task>/{success|failure}_ep<ep>.mp4`**: head at 2x and both wrists, above a chart of the teacher's `T_{t+16}` (orange) vs the student's forecast `V(s_t, track_t)` (blue), every `--step` (4) frames; the title gives the episode's MSE (t ≥ 16).

## Checkpoint format

`teacher.pt` / `student.pt` are `torch.save` dicts:

| key | content |
|---|---|
| `model` | `dino`: the `state_dict`; `qwen3d`: only the trainable weights (LoRA, track encoder, value head) |
| `config` | `arch` and its kwargs: `dino` → `MultiViewDiscriminator` (`inputs`, `n_layers`, `dropout`, `rope_spatial_scale`); `qwen3d` → base weight paths, voxel, LoRA, tasks and prompt token ids |
| `normalizer` | `state_lo` / `state_hi` (robot state min / max) and `track_rms` |
| `epoch`, `args` | training metadata |
| `task` (teacher) | the task it was trained on |
| `teachers` (student) | task -> teacher run dir; their labels are the student's targets |

`model.load(path, device)` rebuilds either model in eval mode (`qwen3d`: the base from `config["qwen3d_ckpt"]`). Batch keys:
- `data.Frames.inputs` (DinoV2 models): `dino [B,3,391,1024]`, `centres [B,3,391,3]`, `centre_ok [B,3,391]`, `state [B,16]`, `state_xyz [B,2,3]`; the student also `track [B,391,16,3]`, `valid [B,391]`;
- `data.Frames.qwen_inputs`: `rgb [B,3,240,320,3]`, `centres [B,391,3]`, `centre_ok [B,391]`, `qcentres [B,2,99,3]`, `qcentre_ok [B,2,99]`, `task [B]`, `track` (normalized) / `track_m` (m) `[R,391,16,3]`, `valid [R,391]`, R a multiple of B.
