#!/usr/bin/env python3
"""Headless multi-view PNG renderer for ContactGen grasps and contact fields.

Reads grasp_<i>.npz saved by demo.py (after optimize_pose + final MANO forward pass)
and writes labeled PNGs:
  grasp_<i>_grasp.png       hand + object mesh
  grasp_<i>_contact.png     contacts_object heatmap on object samples
  grasp_<i>_partition.png   partition_object on object + hand part colors on MANO mesh
  grasp_<i>_uv.png          uv_object arrows + hand/object context

Each PNG title bar documents the color scheme used (mesh hex colors, colormaps,
finger-part palette). See grasp_color_legend(), contact_color_legend(),
partition_color_legend(), uv_color_legend() in this module.

Prerequisites (read before debugging "penetrating" grasps):
  1. Run the full demo first so npz files exist and match the current run:
       cd contactgen
       python demo.py --obj_path assets/toothpaste.ply --n_samples=10 --save_root exp/demo_results
  2. Always render from grasp_<i>.npz — it is the canonical bundle: optimized hand_verts,
     centered obj_mesh_verts, contact/partition/uv fields, and hand_frames from one pass.
  3. Do not pair grasp_*.obj with assets/<object>.ply. demo.py centers the object before
     optimization; hands and npz meshes live in that centered frame. The README vis_grasp.py
     example uses the unc centered asset mesh and will look misaligned / penetrating.
     Use exp/demo_results/<object>.ply or obj_mesh_verts from the npz instead.
  4. Prefer Open3D OffscreenRenderer (default). The matplotlib fallback has no real z-buffer;
     overlapping hand/object triangles can look like penetration even when geometry is correct.

Rendering backends (auto tries Open3D first):
  - Open3D OffscreenRenderer + EGL (default on this server): lit 3D, no DISPLAY/xvfb.
  - matplotlib Agg fallback: flat shaded 4-view panels if EGL is unavailable.

Open3D black-screen note:
  Legacy vis_grasp.py uses Visualizer.create_window() under xvfb/GLFW. On this stack
  create_window() succeeds but capture_screen_image() returns all zeros (GLFW+GLX path
  never binds a working context). OffscreenRenderer with EGL_PLATFORM=surfaceless and
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json works.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import trimesh
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

_CONTACTGEN_ROOT = os.path.dirname(os.path.abspath(__file__))
_HAND_PART_LABEL = os.path.join(_CONTACTGEN_ROOT, "assets", "hand_part_label.pkl")
_EGL_VENDORS = (
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
)

HAND_COLOR = np.array([0.85882353, 0.74117647, 0.65098039])
OBJ_COLOR = np.array([145 / 255, 191 / 255, 219 / 255])
UV_BG_COLOR = np.array([0.20, 0.22, 0.28])
UV_HAND_COLOR = np.array([0.96, 0.78, 0.60])  # warm peach — distinct from object blue
UV_OBJ_COLOR = np.array([0.48, 0.70, 0.88])   # steel blue object in UV view
UV_HAND_ALPHA = 0.42
UV_OBJ_ALPHA = 0.55
# plasma: dark purple (weak) → orange/pink (strong); readable on white (no white/yellow highlights)
CONTACT_CMAP = colormaps["plasma"]
CONTACT_GREY = np.array([0.72, 0.72, 0.72])

# MANO part_ids (ContactGen hand_part_label.pkl / partition_object), 16 rigid parts:
#   0=palm; 1-3=index; 4-6=middle; 7-9=ring; 10-12=little; 13-15=thumb (mcp/pip/dip).
PART_NAMES = {
    0: "palm",
    1: "index_mcp", 2: "index_pip", 3: "index_dip",
    4: "middle_mcp", 5: "middle_pip", 6: "middle_dip",
    7: "ring_mcp", 8: "ring_pip", 9: "ring_dip",
    10: "little_mcp", 11: "little_pip", 12: "little_dip",
    13: "thumb_mcp", 14: "thumb_pip", 15: "thumb_dip",
}
HAND_PART_COLORS = np.array([
    [1.00, 1.00, 1.00],  # 0 palm — white
    [1.00, 0.65, 0.65],  # 1 index mcp
    [0.95, 0.30, 0.30],  # 2 index pip
    [0.70, 0.08, 0.08],  # 3 index dip
    [0.65, 0.95, 0.65],  # 4 middle mcp
    [0.25, 0.78, 0.28],  # 5 middle pip
    [0.08, 0.48, 0.12],  # 6 middle dip
    [0.88, 0.72, 1.00],  # 7 ring mcp
    [0.62, 0.32, 0.88],  # 8 ring pip
    [0.38, 0.12, 0.58],  # 9 ring dip
    [1.00, 0.78, 0.88],  # 10 little mcp
    [0.96, 0.48, 0.72],  # 11 little pip
    [0.78, 0.18, 0.52],  # 12 little dip
    [0.62, 0.82, 1.00],  # 13 thumb mcp
    [0.22, 0.52, 0.96],  # 14 thumb pip
    [0.06, 0.22, 0.72],  # 15 thumb dip
], dtype=np.float64)

FINGER_PART_COLOR_LEGEND = (
    "palm=white | index=reds | middle=greens | ring=purples | "
    "little=pinks | thumb=blues (mcp/pip/dip = light→dark per finger)"
)

_O3D = None
_O3D_OK: bool | None = None


def _setup_egl_env() -> None:
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    if not os.environ.get("DISPLAY"):
        for vendor in _EGL_VENDORS:
            if os.path.isfile(vendor):
                os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", vendor)
                break


def _get_o3d():
    global _O3D, _O3D_OK
    if _O3D_OK is False:
        return None
    if _O3D is not None:
        return _O3D
    _setup_egl_env()
    try:
        import open3d as o3d  # noqa: WPS433 — must follow EGL env setup

        r = o3d.visualization.rendering.OffscreenRenderer(64, 64)
        r.scene.set_background([1.0, 1.0, 1.0, 1.0])
        _ = r.render_to_image()
        _O3D = o3d
        _O3D_OK = True
    except Exception:
        _O3D_OK = False
    return _O3D


def part_rgb(part_ids: np.ndarray) -> np.ndarray:
    part_ids = np.asarray(part_ids, dtype=np.int64).reshape(-1)
    colors = np.zeros((len(part_ids), 3), dtype=np.float64)
    for i, pid in enumerate(part_ids):
        colors[i] = HAND_PART_COLORS[int(pid) % 16]
    return colors


def load_hand_part_labels() -> np.ndarray:
    with open(_HAND_PART_LABEL, "rb") as f:
        return np.asarray(pickle.load(f), dtype=np.int64)


def hand_mesh_part_colors(hand_mesh: trimesh.Trimesh) -> np.ndarray:
    labels = load_hand_part_labels()
    if len(labels) != len(hand_mesh.vertices):
        return np.tile(HAND_COLOR, (len(hand_mesh.vertices), 1))
    return part_rgb(labels)


def uv_local_to_world(uv_local: np.ndarray, partition: np.ndarray, hand_frames: np.ndarray) -> np.ndarray:
    rot = hand_frames[partition, :3, :3]
    world = np.einsum("nij,nj->ni", rot, uv_local)
    norm = np.linalg.norm(world, axis=1, keepdims=True)
    return world / np.maximum(norm, 1e-8)


def contact_rgb(contacts: np.ndarray) -> np.ndarray:
    """Map contacts_object strengths to RGB using CONTACT_CMAP (plasma, white-bg safe)."""
    cnorm = np.clip(np.asarray(contacts, dtype=np.float64).reshape(-1), 0.0, 1.0)
    colors = CONTACT_CMAP(cnorm)[:, :3]
    colors[cnorm <= 0.1] = CONTACT_GREY
    return colors


def rgb_hex(rgb: np.ndarray | list[float]) -> str:
    c = np.asarray(rgb, dtype=np.float64).reshape(-1)[:3]
    return "#" + "".join(f"{int(np.clip(x * 255, 0, 255)):02x}" for x in c)


def grasp_color_legend(*, hand_part_colored: bool = False) -> list[str]:
    hand_note = (
        "hand mesh colored by MANO part id"
        if hand_part_colored
        else f"hand {rgb_hex(HAND_COLOR)}"
    )
    return [
        f"{hand_note} | object {rgb_hex(OBJ_COLOR)} | background white",
        FINGER_PART_COLOR_LEGEND if hand_part_colored else "optimized MANO hand mesh + centered object mesh",
    ]


def contact_color_legend(sample_note: str = "2048 object surface samples") -> list[str]:
    return [
        f"contact strength on {sample_note} (matplotlib plasma colormap)",
        f"grey {rgb_hex(CONTACT_GREY)} = weak/no contact (≤0.1) | purple → orange/pink = stronger",
    ]


def partition_color_legend(part_ids: np.ndarray) -> list[str]:
    uniq = sorted({int(x) for x in np.asarray(part_ids).reshape(-1)})
    names = [PART_NAMES.get(pid, f"part_{pid}") for pid in uniq]
    if len(names) > 10:
        part_summary = ", ".join(names[:10]) + f", … ({len(names)} parts on samples)"
    else:
        part_summary = ", ".join(names) if names else "none"
    return [
        "object dots + hand mesh vertices colored by assigned MANO part id",
        FINGER_PART_COLOR_LEGEND,
        f"parts present on samples: {part_summary}",
    ]


def uv_color_legend(
    n_arrows: int,
    max_arrows: int,
    contact_threshold: float,
    *,
    hand_part_colored: bool = False,
) -> list[str]:
    shown = min(n_arrows, max_arrows)
    hand_note = (
        "hand mesh colored by MANO part id (same palette as grasp/partition)"
        if hand_part_colored
        else f"peach hand {rgb_hex(UV_HAND_COLOR)}"
    )
    return [
        f"Lines: object contact point → hand-part center ({shown} of {n_arrows}, "
        f"threshold={contact_threshold}) | line color = assigned hand part",
        FINGER_PART_COLOR_LEGEND,
        f"Meshes: {hand_note} | steel-blue object {rgb_hex(UV_OBJ_COLOR)} "
        f"| background {rgb_hex(UV_BG_COLOR)}",
        "RGB triads at part centers: X=red, Y=green, Z=blue",
    ]


def part_centers_for_points(partition: np.ndarray, hand_frames: np.ndarray) -> np.ndarray:
    partition = np.asarray(partition, dtype=np.int64).reshape(-1)
    return np.asarray(hand_frames, dtype=np.float64)[partition, :3, 3]


def partition_legend(part_ids: np.ndarray) -> str:
    uniq = sorted({int(x) for x in np.asarray(part_ids).reshape(-1)})
    labels = [f"{PART_NAMES.get(pid, f'part {pid}')} ({pid})" for pid in uniq]
    return "partition_object | " + ", ".join(labels)


FINGER_GROUPS = (
    ("palm", (0,)),
    ("index", (1, 2, 3)),
    ("middle", (4, 5, 6)),
    ("ring", (7, 8, 9)),
    ("little", (10, 11, 12)),
    ("thumb", (13, 14, 15)),
)


def _pil_rgb(rgb: np.ndarray | list[float]) -> tuple[int, int, int]:
    c = np.asarray(rgb, dtype=np.float64).reshape(-1)[:3]
    return tuple(int(np.clip(x * 255, 0, 255)) for x in c)


def _legend_font(size: int = 12) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _fit_legend_width(legend: Image.Image, width: int) -> Image.Image:
    if legend.width == width:
        return legend
    padded = Image.new("RGB", (width, legend.height), "white")
    x0 = max(0, (width - legend.width) // 2)
    padded.paste(legend, (x0, 0))
    return padded


def contact_legend_image(width: int) -> Image.Image:
    """Visual plasma contact bar + grey no-contact swatch."""
    pad = 8
    bar_h = 14
    font = _legend_font(12)
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    _, _, _, text_h = draw_probe.textbbox((0, 0), "strong contact", font=font)
    height = pad + bar_h + 4 + text_h + pad
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    grey_w = 56
    grad_w = min(280, max(120, width // 6))
    x = pad
    y = pad
    draw.rectangle([x, y, x + grey_w, y + bar_h], fill=_pil_rgb(CONTACT_GREY), outline="black")
    draw.text((x + 4, y + bar_h + 2), "≤0.1 no contact", fill="black", font=font)
    x += grey_w + 12

    for i in range(grad_w):
        t = 0.1 + (i / max(grad_w - 1, 1)) * 0.9
        c = _pil_rgb(CONTACT_CMAP(t)[:3])
        draw.line([(x + i, y), (x + i, y + bar_h)], fill=c, width=1)
    draw.rectangle([x, y, x + grad_w, y + bar_h], outline="black")
    draw.text((x, y + bar_h + 2), "0.1", fill="black", font=font)
    draw.text((x + grad_w - 16, y + bar_h + 2), "1.0 strong", fill="black", font=font)
    draw.text((x + grad_w + 10, y + 1), "plasma on object samples", fill="black", font=font)
    return img


def partition_legend_image(width: int, part_ids: np.ndarray | None = None) -> Image.Image:
    """Swatches for MANO hand-part colors (optionally only parts on samples)."""
    present = None
    if part_ids is not None:
        present = {int(x) for x in np.asarray(part_ids).reshape(-1)}

    pad = 8
    swatch = 14
    gap = 4
    group_gap = 14
    font = _legend_font(11)
    small = _legend_font(10)
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    _, _, _, row_h = draw_probe.textbbox((0, 0), "middle", font=font)

    groups = []
    for finger, pids in FINGER_GROUPS:
        active = [pid for pid in pids if present is None or pid in present]
        if active:
            groups.append((finger, active))

    height = pad + row_h + 2 + swatch + 2 + row_h + row_h + pad
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y_name = pad
    y_sw = y_name + row_h + 2
    y_sub = y_sw + swatch + 2

    x = pad
    for finger, active in groups:
        draw.text((x, y_name), finger, fill="black", font=font)
        sx = x
        for j, pid in enumerate(active):
            draw.rectangle(
                [sx, y_sw, sx + swatch, y_sw + swatch],
                fill=_pil_rgb(HAND_PART_COLORS[pid]),
                outline="black",
            )
            suffix = PART_NAMES.get(pid, "").split("_")[-1] if pid != 0 else ""
            if suffix and suffix != PART_NAMES.get(pid, ""):
                draw.text((sx, y_sub), suffix, fill="black", font=small)
            sx += swatch + gap
        x = sx + group_gap

    draw.text((pad, y_sub + row_h if groups else y_sw), FINGER_PART_COLOR_LEGEND,
              fill="black", font=small)
    return img


def grasp_legend_image(width: int) -> Image.Image:
    pad = 8
    swatch = 16
    font = _legend_font(12)
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    _, _, _, text_h = draw_probe.textbbox((0, 0), "hand", font=font)
    height = pad + max(swatch, text_h) + pad
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = pad
    x = pad
    for label, color in (("hand mesh", HAND_COLOR), ("object mesh", OBJ_COLOR)):
        draw.rectangle([x, y, x + swatch, y + swatch], fill=_pil_rgb(color), outline="black")
        draw.text((x + swatch + 6, y + 1), label, fill="black", font=font)
        _, _, tw, _ = draw.textbbox((0, 0), label, font=font)
        x += swatch + 6 + tw + 20
    return img


def uv_legend_image(width: int) -> Image.Image:
    pad = 8
    swatch = 14
    font = _legend_font(11)
    small = _legend_font(10)
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    _, _, _, row_h = draw_probe.textbbox((0, 0), "axis", font=font)
    height = pad + swatch + 4 + row_h + pad
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = pad
    x = pad

    for label, color in (("hand", UV_HAND_COLOR), ("object", UV_OBJ_COLOR)):
        draw.rectangle([x, y, x + swatch, y + swatch], fill=_pil_rgb(color), outline="black")
        draw.text((x + swatch + 4, y + 1), label, fill="black", font=font)
        x += swatch + 4 + 44

    axis_len = 28
    ox = x + 8
    for ax, name, col in ((0, "X", (255, 0, 0)), (1, "Y", (0, 180, 0)), (2, "Z", (0, 0, 255))):
        draw.line([(ox, y + swatch // 2), (ox + axis_len, y + swatch // 2)], fill=col, width=2)
        draw.text((ox + axis_len + 4, y + 1), name, fill=col, font=small)
        ox += axis_len + 22

    draw.text((pad, y + swatch + 4),
              "Line color = assigned hand part (see partition palette) | "
              "line: contact point → part center",
              fill="black", font=small)
    return img


def add_label_bar(
    png_path: str,
    title: str,
    subtitle: str | list[str] = "",
    legend: Image.Image | None = None,
) -> None:
    img = Image.open(png_path).convert("RGB")
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=14)
    draw = ImageDraw.Draw(img)
    _, _, _, title_h = draw.textbbox((0, 0), title, font=font)
    pad = 8
    bar_h = title_h + 2 * pad
    sub_lines: list[str] = []
    if subtitle:
        if isinstance(subtitle, str):
            sub_lines = [subtitle]
        else:
            sub_lines = [line for line in subtitle if line]
    line_gap = 2
    for line in sub_lines:
        _, _, _, sub_h = draw.textbbox((0, 0), line, font=small)
        bar_h += sub_h + line_gap
    legend_h = 0
    if legend is not None:
        legend = _fit_legend_width(legend, img.width)
        legend_h = legend.height
    labeled = Image.new("RGB", (img.width, img.height + bar_h + legend_h), "white")
    labeled.paste(img, (0, bar_h))
    d = ImageDraw.Draw(labeled)
    y = pad
    d.text((pad, y), title, fill="black", font=font)
    y += title_h + line_gap
    for line in sub_lines:
        d.text((pad, y), line, fill="black", font=small)
        _, _, _, sub_h = draw.textbbox((0, 0), line, font=small)
        y += sub_h + line_gap

    if legend is not None:
        labeled.paste(legend, (0, bar_h + img.height))

    labeled.save(png_path)


def save_labeled_png(
    path: str,
    img_uint8: np.ndarray,
    title: str,
    subtitle: str | list[str] = "",
    legend: Image.Image | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(img_uint8).save(path)
    add_label_bar(path, title, subtitle, legend=legend)


# ---- Open3D OffscreenRenderer backend ----

def _o3d_mesh(
    mesh: trimesh.Trimesh,
    vertex_colors: np.ndarray | None = None,
    base_color: list[float] | None = None,
    alpha: float = 1.0,
):
    o3d = _get_o3d()
    g = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int64)),
    )
    if vertex_colors is not None:
        g.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors, dtype=np.float64))
    g.compute_vertex_normals()
    mat = o3d.visualization.rendering.MaterialRecord()
    if alpha < 1.0:
        mat.shader = "defaultLitTransparency"
        mat.has_alpha = True
    else:
        mat.shader = "defaultLit"
    if base_color is not None:
        mat.base_color = [float(base_color[0]), float(base_color[1]), float(base_color[2]), alpha]
    else:
        mat.base_color = [1.0, 1.0, 1.0, alpha]
    return g, mat


def _o3d_points(points: np.ndarray, colors: np.ndarray):
    o3d = _get_o3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 4.0
    return pcd, mat


def _o3d_lines(
    starts: np.ndarray,
    ends: np.ndarray,
    colors: np.ndarray,
    line_width: float = 2.0,
):
    o3d = _get_o3d()
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    n = len(starts)
    pts = np.vstack([starts, ends])
    lines = [[i, i + n] for i in range(n)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = float(line_width)
    return ls, mat


def _o3d_hand_part_frames(hand_frames: np.ndarray, axis_len: float = 0.012) -> tuple:
    """RGB axis triads at each MANO rigid-part origin (hand_frames[i, :3, :3])."""
    hand_frames = np.asarray(hand_frames, dtype=np.float64)
    starts, ends, colors = [], [], []
    axis_rgb = np.eye(3)
    for frame in hand_frames:
        origin = frame[:3, 3]
        rot = frame[:3, :3]
        for ax in range(3):
            starts.append(origin)
            ends.append(origin + rot[:, ax] * axis_len)
            colors.append(axis_rgb[ax])
    return _o3d_lines(np.asarray(starts), np.asarray(ends), np.asarray(colors), line_width=2.0)


def _bounds_from_layers(layers: list[tuple]) -> tuple[np.ndarray, float]:
    pts = []
    for geom, _ in layers:
        if hasattr(geom, "vertices"):
            pts.append(np.asarray(geom.vertices))
        elif hasattr(geom, "points"):
            pts.append(np.asarray(geom.points))
    stacked = np.vstack(pts)
    center = stacked.mean(axis=0)
    extent = float(np.max(np.ptp(stacked, axis=0)))
    return center, max(extent, 1e-3)


def _view_camera_angles(n_views: int) -> list[tuple[float, float]]:
    """Azimuth/elevation pairs: orbit views plus top and bottom when n_views >= 4."""
    if n_views <= 2:
        return [(np.pi / 6.0, np.radians(25.0)), (np.pi, np.radians(-25.0))][:n_views]

    angles: list[tuple[float, float]] = []
    n_orbit = max(n_views - 2, 1)
    for vi in range(n_orbit):
        az = 2.0 * np.pi * vi / n_orbit + np.pi / 6.0
        elev = np.radians(22.0 + 12.0 * (vi % 2))
        angles.append((az, elev))
    # Overhead and underside of the object-centric frame.
    angles.append((0.0, np.radians(78.0)))
    angles.append((0.0, np.radians(-78.0)))
    return angles[:n_views]


def _compose_view_grid(views: list[np.ndarray], n_cols: int = 4) -> np.ndarray:
    """Tile view images into rows of n_cols panels."""
    if not views:
        raise ValueError("views must be non-empty")
    n_cols = max(1, min(n_cols, len(views)))
    n_rows = int(np.ceil(len(views) / n_cols))
    h, w = views[0].shape[:2]
    canvas = np.zeros((n_rows * h, n_cols * w, views[0].shape[2]), dtype=views[0].dtype)
    for i, img in enumerate(views):
        r, c = divmod(i, n_cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return canvas


def render_o3d_layers(
    layers: list[tuple],
    width: int,
    height: int,
    n_views: int = 4,
    n_cols: int = 4,
    background: tuple[float, float, float] | None = None,
) -> np.ndarray:
    o3d = _get_o3d()
    if o3d is None:
        raise RuntimeError("Open3D OffscreenRenderer unavailable")
    bg = background if background is not None else (1.0, 1.0, 1.0)
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([float(bg[0]), float(bg[1]), float(bg[2]), 1.0])
    for i, (geom, mat) in enumerate(layers):
        renderer.scene.add_geometry(f"g{i}", geom, mat)
    center, extent = _bounds_from_layers(layers)
    dist = extent * 2.2
    views: list[np.ndarray] = []
    for az, elev in _view_camera_angles(n_views):
        eye = center + dist * np.array([
            np.cos(az) * np.cos(elev),
            np.sin(az) * np.cos(elev),
            np.sin(elev),
        ])
        renderer.setup_camera(60.0, center, eye, [0.0, 0.0, 1.0])
        views.append(np.asarray(renderer.render_to_image()))
    return _compose_view_grid(views, n_cols=n_cols)


# ---- matplotlib fallback backend ----

def _mpl_render_layers(
    layers: list[dict],
    width: int,
    height: int,
    n_views: int = 4,
    n_cols: int = 4,
    background: tuple[float, float, float] | None = None,
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    all_pts = []
    mesh_data = []
    point_data = []
    arrow_data = []
    for layer in layers:
        kind = layer["kind"]
        if kind == "mesh":
            verts = np.asarray(layer["verts"], dtype=np.float64)
            faces = np.asarray(layer["faces"], dtype=np.int64)
            colors = np.asarray(layer["colors"], dtype=np.float64)
            all_pts.append(verts)
            mesh_data.append({"verts": verts, "faces": faces, "colors": colors, "alpha": layer["alpha"]})
        elif kind == "points":
            all_pts.append(layer["points"])
            point_data.append(layer)
        elif kind == "arrows":
            for start, end, color in layer["segments"]:
                all_pts.append(np.vstack([start, end]))
                arrow_data.append((start, end, color))

    stacked = np.vstack(all_pts)
    half_range = max(float(np.max(np.ptp(stacked, axis=0))), 1e-3) * 0.55
    bg = background if background is not None else (1.0, 1.0, 1.0)
    views = []
    for az, elev in _view_camera_angles(n_views):
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=bg)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=np.degrees(elev), azim=np.degrees(az))
        for mesh in mesh_data:
            collection = Poly3DCollection(
                mesh["verts"][mesh["faces"]],
                facecolors=mesh["colors"][mesh["faces"]].mean(axis=1),
                edgecolors="none",
                alpha=mesh["alpha"],
                linewidths=0,
            )
            ax.add_collection3d(collection)
        for pc in point_data:
            ax.scatter(pc["points"][:, 0], pc["points"][:, 1], pc["points"][:, 2],
                       c=pc["colors"], s=pc["sizes"], alpha=pc["alpha"], depthshade=True)
        for start, end, color in arrow_data:
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, lw=1.2)
        ax.set_xlim(-half_range, half_range)
        ax.set_ylim(-half_range, half_range)
        ax.set_zlim(-half_range, half_range)
        ax.set_box_aspect([1, 1, 1])
        ax.set_facecolor(bg)
        ax.grid(False)
        ax.set_axis_off()
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        views.append(rgba[:, :, :3])
        plt.close(fig)
    return _compose_view_grid(views, n_cols=n_cols)


def _to_mpl_layers(o3d_layers: list[tuple]) -> list[dict]:
    mpl = []
    for geom, mat in o3d_layers:
        alpha = mat.base_color[3] if hasattr(mat, "base_color") else 1.0
        if hasattr(geom, "triangles") and len(np.asarray(geom.triangles)) > 0:
            verts = np.asarray(geom.vertices)
            faces = np.asarray(geom.triangles)
            if geom.has_vertex_colors():
                colors = np.asarray(geom.vertex_colors)
            else:
                bc = mat.base_color[:3]
                colors = np.tile(bc, (len(verts), 1))
            mpl.append({"kind": "mesh", "verts": verts, "faces": faces, "colors": colors, "alpha": alpha})
        elif hasattr(geom, "points") and hasattr(geom, "colors") and len(np.asarray(geom.lines)) == 0:
            mpl.append({
                "kind": "points", "points": np.asarray(geom.points), "colors": np.asarray(geom.colors),
                "sizes": 6.0, "alpha": 0.95,
            })
        elif hasattr(geom, "lines") and len(np.asarray(geom.lines)) > 0:
            pts = np.asarray(geom.points)
            lines = np.asarray(geom.lines)
            colors = np.asarray(geom.colors)
            n = len(lines)
            segments = [(pts[a], pts[b], colors[i % len(colors)]) for i, (a, b) in enumerate(lines)]
            mpl.append({"kind": "arrows", "segments": segments})
    return mpl


def render_scene(
    o3d_layers: list[tuple],
    width: int,
    height: int,
    backend: str,
    background: tuple[float, float, float] | None = None,
    n_views: int = 4,
    n_cols: int = 4,
) -> np.ndarray:
    use_o3d = backend in ("auto", "o3d")
    if use_o3d and _get_o3d() is not None:
        try:
            return render_o3d_layers(
                o3d_layers, width, height, n_views=n_views, n_cols=n_cols, background=background)
        except Exception as exc:
            if backend == "o3d":
                raise
            print(f"Open3D render failed ({exc}); falling back to matplotlib.")
    return _mpl_render_layers(
        _to_mpl_layers(o3d_layers), width, height, n_views=n_views, n_cols=n_cols, background=background)


def _layers_grasp(hand_mesh: trimesh.Trimesh, obj_mesh: trimesh.Trimesh,
                    hand_part_colors: np.ndarray | None = None) -> list[tuple]:
    h_colors = hand_part_colors if hand_part_colors is not None else None
    return [
        _o3d_mesh(hand_mesh, vertex_colors=h_colors, base_color=list(HAND_COLOR), alpha=1.0),
        _o3d_mesh(obj_mesh, base_color=list(OBJ_COLOR), alpha=1.0),
    ]


def _layers_uv_context(
    hand_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    hand_part_colors: np.ndarray | None = None,
) -> list[tuple]:
    """Hand + object backdrop for uv_object arrows."""
    return [
        _o3d_mesh(obj_mesh, base_color=list(UV_OBJ_COLOR), alpha=UV_OBJ_ALPHA),
        _o3d_mesh(
            hand_mesh,
            vertex_colors=hand_part_colors,
            base_color=list(UV_HAND_COLOR),
            alpha=UV_HAND_ALPHA,
        ),
    ]


def render_grasp_npz(
    npz_path: str,
    out_dir: str,
    width: int,
    height: int,
    contact_threshold: float,
    arrow_scale: float,
    max_arrows: int,
    backend: str,
    n_views: int = 8,
    n_cols: int = 4,
) -> dict[str, str]:
    data = np.load(npz_path)
    idx = os.path.splitext(os.path.basename(npz_path))[0]
    obj_mesh = trimesh.Trimesh(data["obj_mesh_verts"], data["obj_mesh_faces"], process=False)
    hand_mesh = trimesh.Trimesh(data["hand_verts"], data["hand_faces"], process=False)
    hand_part_colors = hand_mesh_part_colors(hand_mesh)

    obj_pts = np.asarray(data["obj_verts"], dtype=np.float64)
    contacts = np.asarray(data["contacts_object"], dtype=np.float64).reshape(-1)
    partition = np.asarray(data["partition_object"], dtype=np.int64).reshape(-1)
    uv_local = np.asarray(data["uv_object"], dtype=np.float64).reshape(-1, 3)
    hand_frames = np.asarray(data["hand_frames"], dtype=np.float64)

    outputs: dict[str, str] = {}

    scene_kw = dict(n_views=n_views, n_cols=n_cols)

    grasp_path = os.path.join(out_dir, f"{idx}_grasp.png")
    img = render_scene(
        _layers_grasp(hand_mesh, obj_mesh, hand_part_colors), width, height, backend, **scene_kw)
    save_labeled_png(
        grasp_path, img, "grasp", grasp_color_legend(hand_part_colored=True),
        legend=grasp_legend_image(img.shape[1]),
    )
    outputs["grasp"] = grasp_path

    ccolors = contact_rgb(contacts)
    contact_layers = [
        _o3d_mesh(obj_mesh, base_color=list(OBJ_COLOR), alpha=0.45),
        _o3d_points(obj_pts, ccolors),
    ]
    contact_path = os.path.join(out_dir, f"{idx}_contact.png")
    contact_img = render_scene(contact_layers, width, height, backend, **scene_kw)
    save_labeled_png(
        contact_path, contact_img,
        "contacts_object",
        contact_color_legend(),
        legend=contact_legend_image(contact_img.shape[1]),
    )
    outputs["contact"] = contact_path

    pcolors = part_rgb(partition)
    part_layers = [
        _o3d_mesh(hand_mesh, vertex_colors=hand_part_colors),
        _o3d_mesh(obj_mesh, base_color=list(OBJ_COLOR), alpha=0.35),
        _o3d_points(obj_pts, pcolors),
    ]
    part_path = os.path.join(out_dir, f"{idx}_partition.png")
    part_img = render_scene(part_layers, width, height, backend, **scene_kw)
    save_labeled_png(
        part_path, part_img,
        "partition_object",
        partition_color_legend(partition),
        legend=partition_legend_image(part_img.shape[1], partition),
    )
    outputs["partition"] = part_path

    mask = contacts >= contact_threshold
    n_arrows = int(mask.sum())
    uv_layers = _layers_uv_context(hand_mesh, obj_mesh, hand_part_colors)
    uv_layers.append(_o3d_hand_part_frames(hand_frames))
    if n_arrows > 0:
        idxs = np.flatnonzero(mask)
        if len(idxs) > max_arrows:
            idxs = idxs[np.linspace(0, len(idxs) - 1, max_arrows, dtype=int)]
        # uv_object is the unit vector from part center → object contact (local frame).
        # Draw the geometric link: object sample → assigned part center (world frame).
        starts = obj_pts[idxs]
        ends = part_centers_for_points(partition[idxs], hand_frames)
        acolors = part_rgb(partition[idxs])
        uv_layers.append(_o3d_lines(starts, ends, acolors, line_width=3.0))
    uv_path = os.path.join(out_dir, f"{idx}_uv.png")
    uv_img = render_scene(
        uv_layers, width, height, backend, background=tuple(UV_BG_COLOR), **scene_kw)
    save_labeled_png(
        uv_path, uv_img,
        "uv_object",
        uv_color_legend(n_arrows, max_arrows, contact_threshold, hand_part_colored=True),
        legend=uv_legend_image(uv_img.shape[1]),
    )
    outputs["uv"] = uv_path
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-root", default="exp/demo_results")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--npz-path", default=None)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--backend", choices=("auto", "o3d", "matplotlib"), default="auto")
    parser.add_argument("--contact-threshold", type=float, default=0.25)
    parser.add_argument("--arrow-scale", type=float, default=0.03)
    parser.add_argument("--max-arrows", type=int, default=120)
    parser.add_argument("--n-views", type=int, default=8,
                        help="Number of camera angles tiled into each PNG (default 8).")
    parser.add_argument("--view-cols", type=int, default=4,
                        help="Panels per row when tiling multi-view renders (default 4).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or args.save_root
    os.makedirs(output_dir, exist_ok=True)
    npz_files = [args.npz_path] if args.npz_path else sorted(
        glob.glob(os.path.join(args.save_root, "grasp_*.npz")))
    if not npz_files:
        raise SystemExit(f"No grasp_*.npz in {args.save_root}. Re-run demo.py first.")
    backend_used = "matplotlib"
    if args.backend in ("auto", "o3d") and _get_o3d() is not None:
        backend_used = "open3d-egl"
    elif args.backend == "o3d":
        raise SystemExit("Open3D OffscreenRenderer unavailable; install open3d and EGL vendor libs.")
    print(f"Render backend: {backend_used}")
    for npz_path in npz_files:
        outputs = render_grasp_npz(
            npz_path, output_dir, args.width, args.height,
            args.contact_threshold, args.arrow_scale, args.max_arrows, args.backend,
            n_views=args.n_views, n_cols=args.view_cols,
        )
        print(f"{npz_path}:")
        for kind, path in outputs.items():
            print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
