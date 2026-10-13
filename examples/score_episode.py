"""Score every decision step (every 16th frame) of the episodes in one
RoboTwin scan_object collector part (HDF5 with head + wrist RGB-D and GT tracks).

    python examples/score_episode.py --part scan_wdepth-w0-part0000.hdf5 \
        --ckpt Mercury007/mv-track-svm-scan-object
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mv_svm import MVTrackSVMReward, svm_reward  # noqa: E402

CHUNK = 16  # GAP executes 16 actions per decision; one reward per decision

# obs key -> HDF5 dataset
KEYS = {
    "head_rgb": "image", "head_depth": "depth", "head_K": "camera_intrinsic", "head_E": "camera_extrinsic",
    **{f"{w}_{k}": f"{w}_{h}" for w in ("left_wrist", "right_wrist")
       for k, h in (("rgb", "image"), ("depth", "depth"), ("K", "intrinsic"), ("E", "extrinsic"))},
    "state": "robot_state", "track": "3d_track", "track_valid": "valid_point_mask",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", type=Path, required=True)
    p.add_argument("--ckpt", required=True, help="HF repo id or local dir with model.safetensors + config.json")
    p.add_argument("--use-stored-head-dino", action="store_true",
                   help="read the head DINO features from the part instead of re-extracting them")
    args = p.parse_args()

    scorer = (MVTrackSVMReward(args.ckpt) if Path(args.ckpt).is_dir()
              else MVTrackSVMReward.from_pretrained(args.ckpt))
    with h5py.File(args.part, "r") as f:
        ends = f["episode_ends"][:]
        for ep, (s, e) in enumerate(zip(np.concatenate([[0], ends[:-1]]), ends)):
            steps = np.arange(s, e, CHUNK)
            obs = {k: f[h][steps] for k, h in KEYS.items()}
            head = f["dino_features"][steps][:, 0] if args.use_stored_head_dino else None
            logits = scorer.logits(obs, head_dino=head)
            print(f"ep{ep} seed {f['episode_seed'][ep]} success={f['episode_success'][ep]}: "
                  f"{len(steps)} steps, mean reward {svm_reward(logits).mean():+.3f}")
            print("  logits", np.array2string(logits, precision=2, max_line_width=120))


if __name__ == "__main__":
    main()
