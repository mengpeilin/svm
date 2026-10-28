"""Checks of the Qwen-3D student against upstream on real frames (one GPU):

  patchify    `qwen_model.patchify` == Qwen2VLImageProcessor (do_resize=False)
  vit         `BatchedViT` (images as a batch dimension) vs Qwen's packed ViT forward
  positions   our 4D positions == Qwen-3D's get_xyz_rope_index (batch 1, no
              future tokens: the layout upstream supports)
  batching    V of each row alone == V in a padded batch (static + forward rows), in
              bf16 and in fp32
  future      the future tokens are exactly the tracked workspace patches whose
              displacement exceeds move_thresh by step k

    python -m mv_distill.qwen_check --teacher-pattern teacher_{task}
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .config import FAILURE, MOVE_THRESH, SUCCESS, TASKS, WORKSPACE
from .data import fit_normalizer
from .qwen_model import FUTURE_KS, HEAD_HW, WRIST_HW, build, make_config, patchify, preprocess
from .student import StudentData


def rel(a: torch.Tensor, b: torch.Tensor) -> str:
    a, b = a.float(), b.float()
    return f"max abs {float((a - b).abs().max()):.3g}, rel {float((a - b).norm() / b.norm()):.3g}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher-pattern", default="teacher2_{task}")
    p.add_argument("--qwen-dir", default="/scratch/pmeng/hf/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--qwen3d-ckpt", default="/scratch/pmeng/hf/Qwen-3D/qwen3d_3b.pth")
    p.add_argument("--voxel", type=float, default=0.02)
    p.add_argument("--frames", type=int, default=4)
    args = p.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    sd = StudentData({t: args.teacher_pattern.format(task=t) for t in TASKS}, "qwen3d")
    norm = fit_normalizer(sd.fr, np.concatenate([sd.fr.frames(label, "train", "decision")[::50]
                                                 for label in (SUCCESS, FAILURE)]))
    config = make_config(sd.fr.tasks, args.qwen_dir, args.qwen3d_ckpt, args.voxel, MOVE_THRESH, 16, 32, 0.0)
    model = build(config, device).eval()
    frames = np.sort(rng.choice(sd.train_frames(SUCCESS), args.frames, replace=False))
    b, _ = sd.pair_batch(frames, norm, device)
    print(f"prompt: prefix {len(config['prefix_ids'])} tokens, suffix {config['suffix_len']} tokens")

    # patchify vs the HF image processor, on our resized uint8 image
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(args.qwen_dir)
    for hw in (HEAD_HW, WRIST_HW):
        x = torch.nn.functional.interpolate(b["rgb"][:1, 0].permute(0, 3, 1, 2).float(), size=hw, mode="bicubic",
                                            align_corners=False, antialias=True).clamp(0, 255).round()
        ref = proc(images=[x[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()], do_resize=False,
                   return_tensors="pt")
        mean = x.new_tensor(proc.image_mean).view(1, 3, 1, 1)
        std = x.new_tensor(proc.image_std).view(1, 3, 1, 1)
        ours = patchify((x / 255.0 - mean) / std)[0].cpu()
        assert ref["image_grid_thw"].tolist() == [[1, hw[0] // 14, hw[1] // 14]], ref["image_grid_thw"]
        print(f"patchify {hw}: {rel(ours, ref['pixel_values'])}")

    # batched ViT vs the packed upstream forward
    with torch.no_grad():
        for vit, img, hw in ((model.head_vit, b["rgb"][:2, 0], HEAD_HW), (model.wrist_vit, b["rgb"][:2, 1], WRIST_HW)):
            x = preprocess(img, hw).to(torch.bfloat16)
            ours = vit(x)
            grid = torch.tensor([[1, hw[0] // 14, hw[1] // 14]] * 2, device=device)
            ref = model.qwen.visual(patchify(x).reshape(-1, 1176), grid).view(2, -1, ours.shape[-1])
            print(f"vit {hw}: {tuple(ours.shape)}, {rel(ours, ref)}")

    # positions vs get_xyz_rope_index (batch 1, zero track: no future tokens)
    one = {k: v[:1] for k, v in b.items() if k not in ("track", "track_m", "valid")}
    one.update(track=b["track"][:1] * 0, track_m=b["track_m"][:1] * 0, valid=b["valid"][:1])
    with torch.no_grad():
        rows_e, rows_p = model.sequences(one)
    P = len(config["prefix_ids"])
    task = int(one["task"][0])
    n_suf = config["suffix_len"][task]
    n_v = len(rows_p[0]) - P - n_suf
    pc_id = model.qwen.config.pointcloud_token_id
    ids = torch.tensor([config["prefix_ids"] + [pc_id] * n_v + config["suffix_ids"][task][:n_suf]], device=device)
    ref, _ = model.qwen.get_xyz_rope_index(ids, None, rows_p[0][P:P + n_v, 1:][None].float())
    assert torch.equal(ref[:, 0].T.float(), rows_p[0]), "positions differ from get_xyz_rope_index"
    print(f"positions: equal to get_xyz_rope_index ({n_v} voxels, {len(rows_p[0])} tokens)")

    # future tokens
    with torch.no_grad():
        head, _, _ = model._observation(b)
        fut_f, fut_x, fut_t = model._future(b, head)
    R = b["track"].shape[0]
    for r in range(R):
        o = r % len(frames)
        c = b["centres"][o].cpu().numpy()
        box = np.array(WORKSPACE)
        ok = (b["valid"][r] & b["centre_ok"][o]).cpu().numpy() & ((c >= box[:, 0]) & (c <= box[:, 1])).all(-1)
        disp = np.linalg.norm(b["track_m"][r].cpu().numpy(), axis=-1)
        want = sum(int((ok & (disp[:, :k] > MOVE_THRESH).any(1)).sum()) for k in FUTURE_KS)
        assert len(fut_f[r]) == want, (r, len(fut_f[r]), want)
    print(f"future tokens per row: {[len(f) for f in fut_f]} (first {len(frames)} rows static)")

    # batching: each row alone vs all rows padded together
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        together = model(b).float()
        alone = []
        for r in range(R):
            o = r % len(frames)
            br = {k: v[o:o + 1] for k, v in b.items() if k not in ("track", "track_m", "valid")}
            br.update({k: b[k][r:r + 1] for k in ("track", "track_m", "valid")})
            alone.append(model(br).float())
        alone = torch.cat(alone)
    print(f"batching (bf16): V together {together.cpu().numpy().round(4)}\n                 V alone    "
          f"{alone.cpu().numpy().round(4)}\n                 {rel(together, alone)}")
    model.float()
    model.llm.rotary_emb_4d.dtype = torch.float32
    with torch.no_grad():
        together = model(b)
        alone = torch.cat([model({**{k: v[r % len(frames):r % len(frames) + 1] for k, v in b.items()
                                     if k not in ("track", "track_m", "valid")},
                                  **{k: b[k][r:r + 1] for k in ("track", "track_m", "valid")}}) for r in range(R)])
    print(f"batching (fp32): {rel(together, alone)}")


if __name__ == "__main__":
    main()
