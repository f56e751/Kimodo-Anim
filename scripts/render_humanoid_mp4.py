"""Headless humanoid mesh renderer for Kimodo NPZ outputs.

Uses kimodo's bundled SOMA skin (no py-soma-x required) to produce posed
mesh vertices, then renders front + 3/4 view with pyrender (EGL) and
encodes to mp4 via PyAV.

Usage:
    python scripts/render_humanoid_mp4.py outputs/spearman_idle.npz outputs/spearman_idle_mesh.mp4
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import av
import numpy as np
import pyrender
import torch
import trimesh

from kimodo.skeleton.definitions import SOMASkeleton77

import importlib.util
import sys
_soma_skin_path = Path(__file__).resolve().parent.parent / "kimodo" / "viz" / "soma_skin.py"
_spec = importlib.util.spec_from_file_location("kimodo_viz_soma_skin", _soma_skin_path)
_module = importlib.util.module_from_spec(_spec)
sys.modules["kimodo_viz_soma_skin"] = _module
_spec.loader.exec_module(_module)
SOMASkin = _module.SOMASkin


def compute_vertices(npz_path: Path, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    posed_joints = torch.from_numpy(data["posed_joints"]).to(device).float()  # (T, 77, 3)
    global_rot_mats = torch.from_numpy(data["global_rot_mats"]).to(device).float()  # (T, 77, 3, 3)

    skel = SOMASkeleton77().to(device)
    skin = SOMASkin(skel)
    with torch.no_grad():
        verts = skin.skin(global_rot_mats, posed_joints, rot_is_global=True)
    return verts.cpu().numpy(), skin.faces.cpu().numpy()


def make_camera(target_xyz: np.ndarray, distance: float, yaw_deg: float, pitch_deg: float = -10.0):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    eye = target_xyz + distance * np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    forward = target_xyz - eye
    forward /= np.linalg.norm(forward) + 1e-8
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)
    cam_pose = np.eye(4)
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = up
    cam_pose[:3, 2] = -forward
    cam_pose[:3, 3] = eye
    return cam_pose


def render(npz_path: Path, out_path: Path, fps: int = 30, width: int = 720, height: int = 540):
    print(f"loading {npz_path}")
    verts_seq, faces = compute_vertices(npz_path)  # (T, V, 3), (F, 3)
    T = verts_seq.shape[0]
    print(f"{T} frames, {verts_seq.shape[1]} verts, {faces.shape[0]} faces")

    pmin = verts_seq.reshape(-1, 3).min(0)
    pmax = verts_seq.reshape(-1, 3).max(0)
    center = (pmin + pmax) / 2
    char_size = float(np.linalg.norm(pmax - pmin))
    cam_dist = char_size * 0.9

    scene_w, scene_h = width, height
    panel_w = scene_w // 2

    renderer = pyrender.OffscreenRenderer(viewport_width=panel_w, viewport_height=scene_h)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=panel_w / scene_h)
    light_key = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    light_fill = pyrender.DirectionalLight(color=np.ones(3) * 0.7, intensity=2.5)

    light_key_pose = np.eye(4)
    light_key_pose[:3, 3] = [2, 3, 2]
    light_fill_pose = np.eye(4)
    light_fill_pose[:3, 3] = [-2, 2, -2]

    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = (scene_w // 2) * 2
    stream.height = (scene_h // 2) * 2
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "medium"}

    body_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.65, 0.55, 0.45, 1.0],
        metallicFactor=0.05,
        roughnessFactor=0.85,
    )
    floor_color = np.array([0.85, 0.85, 0.85, 1.0])

    floor_size = char_size * 2.0
    floor = trimesh.creation.box(extents=[floor_size, 0.02, floor_size])
    floor.visual.face_colors = (floor_color * 255).astype(np.uint8)
    floor_y = float(verts_seq[..., 1].min()) - 0.01
    floor.apply_translation([center[0], floor_y, center[2]])
    floor_mesh = pyrender.Mesh.from_trimesh(floor, smooth=False)

    for t in range(T):
        verts = verts_seq[t]
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        body_mesh = pyrender.Mesh.from_trimesh(tm, smooth=True, material=body_material)

        panels = []
        for yaw in (180.0, 135.0):  # front, 3/4
            scene = pyrender.Scene(bg_color=np.array([1.0, 1.0, 1.0, 1.0]), ambient_light=np.ones(3) * 0.4)
            scene.add(body_mesh)
            scene.add(floor_mesh)
            cam_pose = make_camera(center, cam_dist, yaw_deg=yaw, pitch_deg=-8.0)
            scene.add(cam, pose=cam_pose)
            scene.add(light_key, pose=light_key_pose)
            scene.add(light_fill, pose=light_fill_pose)
            color, _ = renderer.render(scene)
            panels.append(color)

        composite = np.concatenate(panels, axis=1)  # side-by-side
        # Pad to even dims if necessary
        h, w = composite.shape[:2]
        composite = composite[: (h // 2) * 2, : (w // 2) * 2]
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(composite), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

        if (t + 1) % 30 == 0 or t == T - 1:
            print(f"  rendered {t + 1}/{T}")

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    renderer.delete()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("mp4", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    render(args.npz, args.mp4, fps=args.fps, width=args.width, height=args.height)
