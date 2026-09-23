import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from render_topdown_arkitscenes import load_points_and_colors


@dataclass(frozen=True)
class ObjectBbox:
    class_name: str
    position_x: float
    position_y: float
    position_z: float
    angle_z: float
    scale_x: float
    scale_y: float
    scale_z: float


@dataclass(frozen=True)
class Region:
    position_x: float
    position_y: float
    position_z: float
    scale_x: float
    scale_y: float
    scale_z: float


@dataclass(frozen=True)
class CanvasTransform:
    x_min: float
    y_max: float
    scale: float
    x_pad: float
    y_pad: float
    resolution: int


def parse_bboxes(layout_text: str) -> list[ObjectBbox]:
    bboxes: list[ObjectBbox] = []
    layout_text = layout_text.replace("<|layout_s|>", "").replace("<|layout_e|>", "")

    for line in layout_text.splitlines():
        if "=Bbox(" not in line:
            continue

        bbox_start = line.index("=Bbox(") + len("=Bbox(")
        bbox_body = line[bbox_start:].split(")", 1)[0]
        parts = [part.strip() for part in bbox_body.rsplit(",", 7)]
        if len(parts) != 8:
            continue

        class_name = parts[0]
        values = [float(value) for value in parts[1:]]
        bboxes.append(ObjectBbox(class_name, *values))

    return bboxes


def scene_id_from_item(item: dict) -> str:
    point_clouds = item.get("point_clouds") or []
    if not point_clouds:
        raise ValueError("Test item has no point_clouds entry")
    return Path(point_clouds[0]).stem


def layout_text_from_item(item: dict) -> str:
    for conversation in item.get("conversations", []):
        if conversation.get("from") == "gpt":
            return conversation.get("value", "")
    return ""


def bbox_corners(box: ObjectBbox | Region) -> np.ndarray:
    hx = box.scale_x * 0.5
    hy = box.scale_y * 0.5
    hz = box.scale_z * 0.5
    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )

    angle_z = getattr(box, "angle_z", 0.0)
    c = np.cos(angle_z)
    s = np.sin(angle_z)
    rotation = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    center = np.array(
        [box.position_x, box.position_y, box.position_z], dtype=np.float64
    )
    return local @ rotation.T + center


def footprint_corners(box: ObjectBbox | Region) -> np.ndarray:
    hx = box.scale_x * 0.5
    hy = box.scale_y * 0.5
    local = np.array(
        [
            [-hx, -hy],
            [hx, -hy],
            [hx, hy],
            [-hx, hy],
        ],
        dtype=np.float64,
    )
    angle_z = getattr(box, "angle_z", 0.0)
    c = np.cos(angle_z)
    s = np.sin(angle_z)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    center = np.array([box.position_x, box.position_y], dtype=np.float64)
    return local @ rotation.T + center


def deterministic_kmeans(points: np.ndarray, k: int, max_iters: int = 100) -> np.ndarray:
    point_count = points.shape[0]
    if point_count == 0:
        return np.empty((0,), dtype=np.int64)
    if point_count <= k:
        return np.arange(point_count, dtype=np.int64)

    centers = np.empty((k, points.shape[1]), dtype=np.float64)
    mean = points.mean(axis=0, keepdims=True)
    first_index = int(np.argmin(np.sum((points - mean) ** 2, axis=1)))
    centers[0] = points[first_index]

    closest_dist = np.sum((points - centers[0]) ** 2, axis=1)
    for center_index in range(1, k):
        next_index = int(np.argmax(closest_dist))
        centers[center_index] = points[next_index]
        closest_dist = np.minimum(
            closest_dist, np.sum((points - centers[center_index]) ** 2, axis=1)
        )

    labels = np.zeros(point_count, dtype=np.int64)
    for _ in range(max_iters):
        distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()
        for center_index in range(k):
            member_mask = new_labels == center_index
            if np.any(member_mask):
                new_centers[center_index] = points[member_mask].mean(axis=0)
            else:
                assigned_dist = distances[np.arange(point_count), new_labels]
                farthest_index = int(np.argmax(assigned_dist))
                new_centers[center_index] = points[farthest_index]
                new_labels[farthest_index] = center_index

        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            break

        labels = new_labels
        centers = new_centers

    unique_labels = sorted(int(label) for label in np.unique(labels))
    remap = {label: index for index, label in enumerate(unique_labels)}
    return np.array([remap[int(label)] for label in labels], dtype=np.int64)


def region_from_boxes(boxes: list[ObjectBbox]) -> Region:
    corners = np.concatenate([bbox_corners(box) for box in boxes], axis=0)
    mins = corners.min(axis=0)
    maxs = corners.max(axis=0)
    center = (mins + maxs) * 0.5
    scales = np.maximum(maxs - mins, 1e-6)
    return Region(
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(scales[0]),
        float(scales[1]),
        float(scales[2]),
    )


def make_regions(bboxes: list[ObjectBbox], k: int) -> list[Region]:
    if not bboxes:
        return []

    features = np.array(
        [[bbox.position_x, bbox.position_y] for bbox in bboxes], dtype=np.float64
    )
    labels = deterministic_kmeans(features, min(k, len(bboxes)))

    regions: list[Region] = []
    for label in sorted(int(label) for label in np.unique(labels)):
        members = [bbox for bbox, bbox_label in zip(bboxes, labels) if bbox_label == label]
        regions.append(region_from_boxes(members))
    return regions


def expand_region(region: Region, fraction_per_side: float) -> Region:
    scale_multiplier = 1.0 + 2.0 * fraction_per_side
    return Region(
        region.position_x,
        region.position_y,
        region.position_z,
        region.scale_x * scale_multiplier,
        region.scale_y * scale_multiplier,
        region.scale_z * scale_multiplier,
    )


def format_float(value: float) -> str:
    return repr(float(value))


def write_regions(path: Path, regions: list[Region]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for index, region in enumerate(regions):
            values = [
                region.position_x,
                region.position_y,
                region.position_z,
                region.scale_x,
                region.scale_y,
                region.scale_z,
            ]
            value_text = ",".join(format_float(value) for value in values)
            f.write(f"region_{index}=Region({value_text})\n")


def remove_top_point_fraction(
    points: np.ndarray, colors: np.ndarray, fraction: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    if points.size == 0 or fraction <= 0.0:
        return points, colors, {
            "points_before": int(points.shape[0]),
            "points_after": int(points.shape[0]),
            "z_keep_threshold": None,
        }

    if fraction >= 1.0:
        return points[:0], colors[:0], {
            "points_before": int(points.shape[0]),
            "points_after": 0,
            "z_keep_threshold": None,
        }

    keep_threshold = float(np.quantile(points[:, 2], 1.0 - fraction))
    keep_mask = points[:, 2] <= keep_threshold
    return points[keep_mask], colors[keep_mask], {
        "points_before": int(points.shape[0]),
        "points_after": int(keep_mask.sum()),
        "z_keep_threshold": keep_threshold,
    }


def make_canvas_transform(
    points: np.ndarray,
    boxes: list[ObjectBbox],
    regions: list[Region],
    resolution: int,
) -> CanvasTransform:
    xy_parts = []
    if points.size:
        xy_parts.append(points[:, :2])
    xy_parts.extend(footprint_corners(box) for box in boxes)
    xy_parts.extend(footprint_corners(region) for region in regions)

    if xy_parts:
        xy = np.concatenate(xy_parts, axis=0)
        x_min = float(xy[:, 0].min())
        x_max = float(xy[:, 0].max())
        y_min = float(xy[:, 1].min())
        y_max = float(xy[:, 1].max())
    else:
        x_min, x_max, y_min, y_max = -1.0, 1.0, -1.0, 1.0

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    pad = 0.03 * max(x_span, y_span)
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    scale = (resolution - 1) / max(x_span, y_span)
    x_pad = ((resolution - 1) - x_span * scale) * 0.5
    y_pad = ((resolution - 1) - y_span * scale) * 0.5
    return CanvasTransform(x_min, y_max, scale, x_pad, y_pad, resolution)


def world_to_pixel(xy: np.ndarray, transform: CanvasTransform) -> list[tuple[int, int]]:
    px = np.rint((xy[:, 0] - transform.x_min) * transform.scale + transform.x_pad)
    py = np.rint((transform.y_max - xy[:, 1]) * transform.scale + transform.y_pad)
    px = np.clip(px, 0, transform.resolution - 1).astype(np.int64)
    py = np.clip(py, 0, transform.resolution - 1).astype(np.int64)
    return [(int(x), int(y)) for x, y in zip(px, py)]


def render_points(
    points: np.ndarray,
    colors: np.ndarray,
    transform: CanvasTransform,
    point_size: int,
) -> Image.Image:
    resolution = transform.resolution
    image = np.full((resolution, resolution, 3), 255, dtype=np.uint8)
    if points.size == 0:
        return Image.fromarray(image)

    px = np.rint((points[:, 0] - transform.x_min) * transform.scale + transform.x_pad)
    py = np.rint((transform.y_max - points[:, 1]) * transform.scale + transform.y_pad)
    px = np.clip(px, 0, resolution - 1).astype(np.int64)
    py = np.clip(py, 0, resolution - 1).astype(np.int64)
    z = points[:, 2]

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


def draw_footprint(
    draw: ImageDraw.ImageDraw,
    corners_xy: np.ndarray,
    transform: CanvasTransform,
    color: tuple[int, int, int],
    width: int,
) -> None:
    points = world_to_pixel(corners_xy, transform)
    draw.line(points + [points[0]], fill=color, width=width, joint="curve")


def render_scene(
    points: np.ndarray,
    colors: np.ndarray,
    object_bboxes: list[ObjectBbox],
    regions: list[Region],
    output_path: Path,
    resolution: int,
    point_size: int,
    region_color: tuple[int, int, int],
) -> None:
    transform = make_canvas_transform(points, object_bboxes, regions, resolution)
    image = render_points(points, colors, transform, point_size)
    draw = ImageDraw.Draw(image)

    for bbox in object_bboxes:
        draw_footprint(
            draw,
            footprint_corners(bbox),
            transform,
            color=(0, 150, 70),
            width=max(1, resolution // 512),
        )

    for region in regions:
        draw_footprint(
            draw,
            footprint_corners(region),
            transform,
            color=region_color,
            width=max(3, resolution // 220),
        )

    image.save(output_path)


def parse_scene_filter(scene_ids: list[str] | None) -> set[str] | None:
    if not scene_ids:
        return None
    normalized = set()
    for raw in scene_ids:
        for scene_id in raw.split(","):
            scene_id = Path(scene_id.strip()).stem
            if scene_id:
                normalized.add(scene_id)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate K-Means region bboxes from SpatialLM test GT boxes and "
            "render GT/region top-down overlays."
        )
    )
    parser.add_argument(
        "--test_json",
        "--test-json",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "spatiallm" / "spatiallm_test.json",
    )
    parser.add_argument(
        "--output_root",
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "region_bbox",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--expand_fraction",
        "--expand-fraction",
        type=float,
        default=0.25,
        help="Fraction of the original scale added on each side.",
    )
    parser.add_argument(
        "--top_point_fraction",
        "--top-point-fraction",
        type=float,
        default=0.2,
        help="Highest-z fraction of points to remove before rendering.",
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--point_size", "--point-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scene_ids", "--scene-ids", nargs="*", default=None)
    parser.add_argument("--skip_render", "--skip-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.test_json.resolve().parent
    raw_dir = args.output_root / "raw"
    expanded_dir = args.output_root / "expanded"
    raw_render_dir = raw_dir / "render"
    expanded_render_dir = expanded_dir / "render"

    raw_dir.mkdir(parents=True, exist_ok=True)
    expanded_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_render:
        raw_render_dir.mkdir(parents=True, exist_ok=True)
        expanded_render_dir.mkdir(parents=True, exist_ok=True)

    with args.test_json.open("r", encoding="utf-8") as f:
        test_items = json.load(f)

    scene_filter = parse_scene_filter(args.scene_ids)
    if scene_filter is not None:
        test_items = [
            item for item in test_items if scene_id_from_item(item) in scene_filter
        ]
    if args.limit is not None:
        test_items = test_items[: args.limit]

    manifest = {
        "test_json": str(args.test_json),
        "k": args.k,
        "expand_fraction_per_side": args.expand_fraction,
        "top_point_fraction": args.top_point_fraction,
        "resolution": args.resolution,
        "point_size": args.point_size,
        "scenes": [],
    }

    for item_index, item in enumerate(test_items, start=1):
        scene_id = scene_id_from_item(item)
        layout_text = layout_text_from_item(item)
        object_bboxes = parse_bboxes(layout_text)
        raw_regions = make_regions(object_bboxes, args.k)
        expanded_regions = [
            expand_region(region, args.expand_fraction) for region in raw_regions
        ]

        raw_path = raw_dir / f"{scene_id}.txt"
        expanded_path = expanded_dir / f"{scene_id}.txt"
        write_regions(raw_path, raw_regions)
        write_regions(expanded_path, expanded_regions)

        scene_entry = {
            "scene_id": scene_id,
            "object_bbox_count": len(object_bboxes),
            "region_count": len(raw_regions),
            "raw_region_path": str(raw_path),
            "expanded_region_path": str(expanded_path),
        }

        if not args.skip_render:
            pcd_relative = Path((item.get("point_clouds") or [""])[0])
            pcd_path = dataset_root / pcd_relative
            points, colors = load_points_and_colors(pcd_path)
            clipped_points, clipped_colors, clip_stats = remove_top_point_fraction(
                points, colors, args.top_point_fraction
            )

            raw_render_path = raw_render_dir / f"{scene_id}.png"
            expanded_render_path = expanded_render_dir / f"{scene_id}.png"
            render_scene(
                clipped_points,
                clipped_colors,
                object_bboxes,
                raw_regions,
                raw_render_path,
                args.resolution,
                args.point_size,
                region_color=(220, 40, 40),
            )
            render_scene(
                clipped_points,
                clipped_colors,
                object_bboxes,
                expanded_regions,
                expanded_render_path,
                args.resolution,
                args.point_size,
                region_color=(40, 80, 230),
            )
            scene_entry.update(
                {
                    "point_cloud": str(pcd_path),
                    "raw_render_path": str(raw_render_path),
                    "expanded_render_path": str(expanded_render_path),
                    **clip_stats,
                }
            )

        manifest["scenes"].append(scene_entry)
        print(
            f"[{item_index}/{len(test_items)}] {scene_id}: "
            f"{len(object_bboxes)} GT boxes -> {len(raw_regions)} regions"
        )

    manifest_path = args.output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote raw regions to {raw_dir}")
    print(f"Wrote expanded regions to {expanded_dir}")
    if not args.skip_render:
        print(f"Wrote raw renders to {raw_render_dir}")
        print(f"Wrote expanded renders to {expanded_render_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
