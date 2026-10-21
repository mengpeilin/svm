"""Value videos of eval episodes: the cameras (head 2x, both wrists on the
right) above one chart of the ABSOLUTE value 16 frames ahead, every `--step`
frames:

  orange  teacher T_{t+16}
  blue    student forecast V(s_t, track_t)

Both are drawn at the decision frame t and held for `--step` frames, shown
once that step starts. The title gives the episode's MSE between the two over
t >= 16. Episodes: `--n` success + `--n` failure eval episodes drawn with
`--seed`, never one in `--exclude`; failures are cut at the failure cap. Needs
`infer.py`'s eval_frames.npz and the scan parts (camera images).

Writes runs/<run>/videos/{success|failure}_ep{ep}.mp4 (H.264 from RGB frames).

    python -m mv_distill.render --run student --n 5 --seed 1
"""

from __future__ import annotations

import argparse

import av
import cv2
import h5py
import numpy as np

from .config import CHUNK, FAILURE, FAILURE_CAP, SUCCESS, parts, run_dir
from .data import Cache
from .parts import episode_plan

VIEWS = ("image", "left_wrist_image", "right_wrist_image")
FONT = cv2.FONT_HERSHEY_SIMPLEX
BG, INK, MUTED = (24, 24, 24), (235, 235, 235), (150, 150, 150)   # RGB
TEACHER, STUDENT = (255, 170, 60), (90, 170, 255)
HEADER_H, CHART_H = 30, 260
PAD_L, PAD_R, PAD_T, PAD_B = 52, 12, 26, 22


def text(img, s, org, color=INK, scale=0.5, thick=1) -> None:
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def tile(head, left, right) -> np.ndarray:
    """Head at 2x on the left, the two wrist views stacked on the right, labelled."""
    h, w = head.shape[:2]
    out = np.concatenate([cv2.resize(head, (2 * w, 2 * h), interpolation=cv2.INTER_NEAREST),
                          np.concatenate([left, right], 0)], 1)
    for s, (x, y) in (("head", (0, 0)), ("left wrist", (2 * w, 0)), ("right wrist", (2 * w, h))):
        (tw, th), _ = cv2.getTextSize(s, FONT, 0.5, 1)
        cv2.rectangle(out, (x, y), (x + tw + 12, y + th + 10), BG, -1)
        text(out, s, (x + 6, y + th + 5), INK, 0.5)
    return out


def held(t: np.ndarray, y: np.ndarray, n: int, hold: int) -> tuple[np.ndarray, np.ndarray]:
    """Step curve: each value held from its frame for `hold` frames (clipped to n)."""
    return np.stack([t, np.minimum(t + hold, n) - 1], 1).ravel(), np.repeat(y, 2)


def render(video, curves, names, colors, step, out_path, title, fps, vrange) -> None:
    n = len(video[0])
    th, W = 2 * video[0].shape[1], 3 * video[0].shape[2]
    H = HEADER_H + th + CHART_H
    x0, x1, y0, y1 = PAD_L, W - PAD_R, PAD_T + 8, CHART_H - PAD_B
    fx = lambda t: x0 + (x1 - x0) * np.asarray(t, float) / max(n - 1, 1)  # noqa: E731
    fy = lambda v: y0 + (y1 - y0) * (1 - (np.clip(v, -vrange, vrange) + vrange) / (2 * vrange))  # noqa: E731

    base = np.full((CHART_H, W, 3), BG, np.uint8)
    cv2.rectangle(base, (x0, y0), (x1, y1), MUTED, 1)
    for v in (vrange, 0.0, -vrange):
        text(base, f"{v:+.1f}", (4, int(fy(v)) + 4), MUTED, 0.38)
    for t in range(0, n, 50):
        text(base, str(t), (int(fx(t)) - 8, CHART_H - 6), MUTED, 0.38)
    pts = []
    for (x, y), c in zip(curves, colors):
        p = np.stack([fx(x), fy(y)], -1)
        dim = tuple(int(0.35 * ch + 0.65 * bg) for ch, bg in zip(c, BG))
        cv2.polylines(base, [p.astype(np.int32)], False, dim, 1, cv2.LINE_AA)
        pts.append(p)

    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": "20"}
    for t in range(n):
        canvas = np.full((H, W, 3), BG, np.uint8)
        canvas[HEADER_H:HEADER_H + th] = tile(*(v[t] for v in video))
        text(canvas, f"{title}   frame {t}/{n - 1}", (10, 20), INK, 0.5)
        chart = base.copy()
        for j, ((x, y), p, c) in enumerate(zip(curves, pts, colors)):
            k = int(np.searchsorted(x, (t // step) * step + step - 1, side="right"))
            if k > 1:
                cv2.polylines(chart, [p[:k].astype(np.int32)], False, c, 2, cv2.LINE_AA)
            label = names[j]
            if k:
                cv2.circle(chart, (int(p[k - 1, 0]), int(p[k - 1, 1])), 4, c, -1, cv2.LINE_AA)
                label += f"  {float(y[k - 1]):+.2f}"
            text(chart, label, (PAD_L + 6 + j * (W - PAD_L) // len(curves), 17), c, 0.42, 1)
        xt = int(fx(t))
        cv2.line(chart, (xt, y0), (xt, y1), INK, 1)
        canvas[HEADER_H + th:] = chart
        for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"wrote {out_path} ({n} frames)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="student")
    p.add_argument("--n", type=int, default=5, help="eval episodes per class")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", type=int, nargs="*", default=[], help="episodes never drawn")
    p.add_argument("--step", type=int, default=4, help="frames between drawn values")
    p.add_argument("--fps", type=int, default=15, help="captured at 30 Hz")
    p.add_argument("--value-range", type=float, default=7.5, help="chart +- range (clipped)")
    args = p.parse_args()

    z = np.load(run_dir(args.run) / "eval_frames.npz")
    dec = z["t"] % args.step == 0
    ep_of, t_of, gt, forecast = z["episode"][dec], z["t"][dec], z["T_next"][dec], z["forward"][dec]
    cache = Cache("scan", wrist=False)
    srcs = parts("scan")
    plan, ends = episode_plan(srcs)
    if not np.array_equal(ends, cache.bounds[:, 1]):
        raise SystemExit("scan parts do not reproduce the scan cache's episodes")
    out_dir = run_dir(args.run) / "videos"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for label, cls in ((SUCCESS, "success"), (FAILURE, "failure")):
        eps = [e for e in cache.splits[(label, "eval")] if e not in args.exclude]
        for ep in sorted(rng.choice(eps, args.n, replace=False).tolist()):
            k, s, e = plan[ep]
            n = e - s if label == SUCCESS else min(e - s, FAILURE_CAP)
            with h5py.File(srcs[k], "r") as f:
                video = [f[view][s:s + n] for view in VIEWS]
            m = ep_of == ep
            ys, scored = [gt[m], forecast[m]], t_of[m] >= CHUNK
            mse = float(np.mean((ys[1][scored] - ys[0][scored]) ** 2))
            rows.append((cls, ep, mse))
            render(video, [held(t_of[m], y, n, args.step) for y in ys],
                   ["teacher T(t+16)", "student V(s_t, track_t)"], [TEACHER, STUDENT], args.step,
                   out_dir / f"{cls}_ep{ep}.mp4",
                   f"{cls} ep{ep} [value of s(t+16), every {args.step} frames]  MSE (t >= 16): {mse:.2f}",
                   args.fps, args.value_range)
    print("\n| class | ep | MSE (t >= 16) |\n|---|---|---|")
    for cls, ep, mse in rows:
        print(f"| {cls} | {ep} | {mse:.2f} |")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
