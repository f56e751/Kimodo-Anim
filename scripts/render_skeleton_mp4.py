"""Headless skeleton preview renderer for Kimodo NPZ outputs.

Renders posed_joints as a 3D stick figure mp4 using matplotlib + PyAV.
Two synchronized views (front + side) for easier inspection.

Usage:
    python scripts/render_skeleton_mp4.py outputs/spearman_idle.npz outputs/spearman_idle.mp4
"""

import argparse
from pathlib import Path

import av
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kimodo.skeleton.definitions import SOMASkeleton77


def render(npz_path: Path, out_path: Path, fps: int = 30, dpi: int = 100):
    data = np.load(npz_path)
    joints = data["posed_joints"].astype(np.float32)  # (T, J, 3)
    T, J, _ = joints.shape

    skel = SOMASkeleton77()
    parents = [int(p) for p in skel.joint_parents]

    bones = [(i, parents[i]) for i in range(J) if parents[i] >= 0]

    j_min = joints.reshape(-1, 3).min(0)
    j_max = joints.reshape(-1, 3).max(0)
    center = (j_min + j_max) / 2
    span = float((j_max - j_min).max()) * 0.6

    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    ax_front = fig.add_subplot(1, 2, 1, projection="3d")
    ax_side = fig.add_subplot(1, 2, 2, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0.05)

    def setup(ax, title, elev, azim):
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(title, fontsize=10)

    width, height = fig.canvas.get_width_height()
    width = (width // 2) * 2
    height = (height // 2) * 2

    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "medium"}

    for t in range(T):
        for ax, title, elev, azim in [
            (ax_front, "front", 10, 90),
            (ax_side, "side", 10, 0),
        ]:
            ax.cla()
            setup(ax, title, elev, azim)
            pts = joints[t]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#1f77b4", s=8)
            for i, p in bones:
                xs = [pts[i, 0], pts[p, 0]]
                ys = [pts[i, 1], pts[p, 1]]
                zs = [pts[i, 2], pts[p, 2]]
                ax.plot(xs, ys, zs, c="#333", linewidth=1.2)
        fig.suptitle(f"frame {t+1}/{T}", fontsize=9)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = rgba[:height, :width, :3]
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    plt.close(fig)
    print(f"wrote {out_path} ({T} frames @ {fps}fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("mp4", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    render(args.npz, args.mp4, fps=args.fps)
