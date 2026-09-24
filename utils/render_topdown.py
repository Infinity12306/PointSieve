import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def load_points_and_colors(pcd_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from spatiallm.pcd import load_o3d_pcd

    pcd = load_o3d_pcd(str(pcd_path))
    points = np.asarray(pcd.points)
    if points.size == 0:
        return points, np.empty((0, 3), dtype=np.uint8)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if colors.shape[1] > 3:
            colors = colors[:, :3]
        if np.issubdtype(colors.dtype, np.floating) and colors.max(initial=0) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        z = points[:, 2]
        z_span = max(float(z.max() - z.min()), 1e-6)
        gray = ((z - z.min()) / z_span * 255.0).astype(np.uint8)
        colors = np.repeat(gray[:, None], 3, axis=1)

    return points, colors


def save_points_and_colors(ply_path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertex_data = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ],
    )
    vertex_data["x"] = points[:, 0].astype(np.float32)
    vertex_data["y"] = points[:, 1].astype(np.float32)
    vertex_data["z"] = points[:, 2].astype(np.float32)
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    vertex_data["alpha"] = 255

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by render_topdown_arkitscenes.py\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property uchar alpha\n"
        "end_header\n"
    )
    with ply_path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertex_data.tofile(f)


def remove_top_slice(
    points: np.ndarray,
    colors: np.ndarray,
    remove_height_abs: float,
    remove_height_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    z = points[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    room_height = z_max - z_min
    remove_height = min(remove_height_abs, remove_height_ratio * room_height)
    keep_max_z = z_max - remove_height

    keep_mask = z < keep_max_z if remove_height > 0 else np.ones_like(z, dtype=bool)
    stats = {
        "z_min": z_min,
        "z_max": z_max,
        "room_height": room_height,
        "remove_height": remove_height,
        "keep_max_z": keep_max_z,
        "points_before": int(points.shape[0]),
        "points_after": int(keep_mask.sum()),
    }
    return points[keep_mask], colors[keep_mask], stats


def render_topdown(
    points: np.ndarray,
    colors: np.ndarray,
    resolution: int,
    point_size: int,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    image = np.full((resolution, resolution, 3), background, dtype=np.uint8)
    if points.size == 0:
        return Image.fromarray(image)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    x_min = float(x.min())
    x_max = float(x.max())
    y_min = float(y.min())
    y_max = float(y.max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    scale = (resolution - 1) / max(x_span, y_span)
    x_pad = ((resolution - 1) - x_span * scale) * 0.5
    y_pad = ((resolution - 1) - y_span * scale) * 0.5

    px = np.rint((x - x_min) * scale + x_pad).astype(np.int64)
    py = np.rint((y_max - y) * scale + y_pad).astype(np.int64)
    px = np.clip(px, 0, resolution - 1)
    py = np.clip(py, 0, resolution - 1)

    top_z = np.full(resolution * resolution, -np.inf, dtype=z.dtype)
    offset_start = -(point_size // 2)
    offset_stop = offset_start + point_size

    for dy in range(offset_start, offset_stop):
        sy = py + dy
        valid_y = (sy >= 0) & (sy < resolution)
        for dx in range(offset_start, offset_stop):
            sx = px + dx
            valid = valid_y & (sx >= 0) & (sx < resolution)
            linear = sy[valid] * resolution + sx[valid]
            np.maximum.at(top_z, linear, z[valid])

    flat_image = image.reshape(-1, 3)
    for dy in range(offset_start, offset_stop):
        sy = py + dy
        valid_y = (sy >= 0) & (sy < resolution)
        for dx in range(offset_start, offset_stop):
            sx = px + dx
            valid = valid_y & (sx >= 0) & (sx < resolution)
            point_indices = np.flatnonzero(valid)
            linear = sy[point_indices] * resolution + sx[point_indices]
            top_mask = z[point_indices] == top_z[linear]
            flat_image[linear[top_mask]] = colors[point_indices[top_mask]]

    return Image.fromarray(image)


def normalize_scene_ids(scene_ids: list[str] | None) -> list[str] | None:
    if scene_ids is None:
        return None

    normalized = []
    for raw_scene_id in scene_ids:
        for scene_id in raw_scene_id.split(","):
            scene_id = Path(scene_id.strip()).stem
            if scene_id:
                normalized.append(scene_id)
    return normalized


def choose_scenes(
    pcd_dir: Path,
    num_scenes: int,
    seed: int,
    scene_ids: list[str] | None,
) -> list[Path]:
    scene_paths = sorted(pcd_dir.glob("*.ply"))
    if not scene_paths:
        raise FileNotFoundError(f"No .ply files found in {pcd_dir}")

    scene_ids = normalize_scene_ids(scene_ids)
    if scene_ids:
        selected_paths = []
        for scene_id in scene_ids:
            pcd_path = pcd_dir / f"{scene_id}.ply"
            if not pcd_path.exists():
                raise FileNotFoundError(f"Scene {scene_id} not found at {pcd_path}")
            selected_paths.append(pcd_path)
        return selected_paths

    if num_scenes >= len(scene_paths):
        return scene_paths

    rng = random.Random(seed)
    return sorted(rng.sample(scene_paths, num_scenes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render top-down images for ARKitScenes SpatialLM point clouds after "
            "removing the top min(abs_threshold, ratio * room_height) height slice."
        )
    )
    parser.add_argument(
        "--pcd_dir",
        "--pcd-dir",
        type=Path,
        default=Path("arkitscenes-spatiallm/pcd"),
        help="Directory containing ARKitScenes SpatialLM .ply point clouds.",
    )
    parser.add_argument(
        "--resolution",
        type=positive_int,
        default=1024,
        help="Square output image resolution in pixels.",
    )
    parser.add_argument(
        "--point_size",
        "--point-size",
        type=positive_int,
        default=1,
        help="Square point splat size in pixels for the top-down render.",
    )
    parser.add_argument(
        "--remove_height_abs",
        "--remove-height-abs",
        "--threshold_abs",
        "--threshold-abs",
        dest="remove_height_abs",
        type=nonnegative_float,
        default=0.3,
        help="Absolute top height range to remove in meters.",
    )
    parser.add_argument(
        "--remove_height_ratio",
        "--remove-height-ratio",
        "--threshold_ratio",
        "--threshold-ratio",
        dest="remove_height_ratio",
        type=nonnegative_float,
        default=0.1,
        help="Room-height ratio for the top height range to remove.",
    )
    parser.add_argument(
        "--num_scenes",
        "--num-scenes",
        type=nonnegative_int,
        default=5,
        help="Number of random scenes to render.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to randomly select scenes.",
    )
    parser.add_argument(
        "--scene_ids",
        "--scene-ids",
        nargs="*",
        default=None,
        help=(
            "Optional scene ids to render exactly. Accepts space-separated ids, "
            "comma-separated ids, or paths. Overrides --num_scenes and --seed."
        ),
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=Path,
        default=Path("render/arkitscenes-spatiallm"),
        help="Directory where rendered PNGs and manifest.json are written.",
    )
    parser.add_argument(
        "--save_debug_artifacts",
        "--save-debug-artifacts",
        action="store_true",
        help=(
            "Save before/after clipping PNG and PLY files for each scene under "
            "--debug_dir or OUTPUT_DIR/debug."
        ),
    )
    parser.add_argument(
        "--debug_dir",
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory for debug artifacts when --save_debug_artifacts is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = args.debug_dir or args.output_dir / "debug"
    if args.save_debug_artifacts:
        debug_dir.mkdir(parents=True, exist_ok=True)

    selected_paths = choose_scenes(
        args.pcd_dir,
        args.num_scenes,
        args.seed,
        args.scene_ids,
    )

    manifest = {
        "pcd_dir": str(args.pcd_dir),
        "resolution": args.resolution,
        "point_size": args.point_size,
        "remove_height_abs": args.remove_height_abs,
        "remove_height_ratio": args.remove_height_ratio,
        "num_scenes": len(selected_paths),
        "seed": args.seed,
        "scene_ids": normalize_scene_ids(args.scene_ids),
        "save_debug_artifacts": args.save_debug_artifacts,
        "debug_dir": str(debug_dir) if args.save_debug_artifacts else None,
        "scenes": [],
    }

    for scene_idx, pcd_path in enumerate(selected_paths, start=1):
        scene_id = pcd_path.stem
        print(f"[{scene_idx}/{len(selected_paths)}] Rendering {scene_id}")

        points, colors = load_points_and_colors(pcd_path)
        if points.size == 0:
            print(f"  Skipping {scene_id}: empty point cloud")
            continue

        clipped_points, clipped_colors, stats = remove_top_slice(
            points,
            colors,
            args.remove_height_abs,
            args.remove_height_ratio,
        )
        image = render_topdown(
            clipped_points,
            clipped_colors,
            args.resolution,
            args.point_size,
        )

        output_path = args.output_dir / f"{scene_id}.png"
        image.save(output_path)

        scene_entry = {
            "scene_id": scene_id,
            "point_cloud": str(pcd_path),
            "output_image": str(output_path),
            **stats,
        }

        if args.save_debug_artifacts:
            before_png = debug_dir / f"{scene_id}_before_clip.png"
            after_png = debug_dir / f"{scene_id}_after_clip.png"
            before_ply = debug_dir / f"{scene_id}_before_clip.ply"
            after_ply = debug_dir / f"{scene_id}_after_clip.ply"

            render_topdown(points, colors, args.resolution, args.point_size).save(
                before_png
            )
            image.save(after_png)
            save_points_and_colors(before_ply, points, colors)
            save_points_and_colors(after_ply, clipped_points, clipped_colors)

            scene_entry["debug_artifacts"] = {
                "before_clip_image": str(before_png),
                "after_clip_image": str(after_png),
                "before_clip_ply": str(before_ply),
                "after_clip_ply": str(after_ply),
            }

        manifest["scenes"].append(scene_entry)

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(manifest['scenes'])} renders to {args.output_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
