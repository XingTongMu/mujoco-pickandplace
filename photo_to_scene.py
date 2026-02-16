from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedObject:
    name: str
    geom_type: str
    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    mass: float


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def detect_objects_from_photo(
    image_path: str | Path,
    *,
    meters_per_px: float = 0.0015,
    world_center_xy: tuple[float, float] = (0.5, 0.0),
    table_z: float = 0.0,
    min_area_px: int = 500,
) -> list[DetectedObject]:
    image_path = Path(image_path)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    fg = ((sat > 35) & (val > 35)).astype(np.uint8) * 255
    fg = cv2.medianBlur(fg, 5)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs: list[DetectedObject] = []
    idx = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        cx = x + bw / 2.0
        cy = y + bh / 2.0

        wx = world_center_xy[0] + (cx - w / 2.0) * meters_per_px
        wy = world_center_xy[1] + (h / 2.0 - cy) * meters_per_px

        bw_m = bw * meters_per_px
        bh_m = bh * meters_per_px
        thickness = 0.04

        perimeter = cv2.arcLength(cnt, True)
        circularity = 0.0 if perimeter <= 1e-6 else float(4.0 * np.pi * area / (perimeter * perimeter))

        roi = bgr[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]
        if roi.size == 0:
            continue
        mean_bgr = roi.reshape(-1, 3).mean(axis=0)
        rgba = (float(mean_bgr[2] / 255.0), float(mean_bgr[1] / 255.0), float(mean_bgr[0] / 255.0), 1.0)

        if circularity > 0.78:
            geom_type = "cylinder"
            r = _clamp(0.5 * max(bw_m, bh_m), 0.01, 0.08)
            hz = _clamp(0.5 * thickness, 0.01, 0.05)
            size = (r, hz, 0.0)
            z = table_z + hz
        else:
            geom_type = "box"
            sx = _clamp(0.5 * bw_m, 0.01, 0.10)
            sy = _clamp(0.5 * bh_m, 0.01, 0.10)
            sz = _clamp(0.5 * thickness, 0.01, 0.05)
            size = (sx, sy, sz)
            z = table_z + sz

        mass = float(_clamp(area / (w * h) * 1.0, 0.02, 0.25))
        objs.append(
            DetectedObject(
                name=f"obj_{idx}",
                geom_type=geom_type,
                size=size,
                pos=(float(wx), float(wy), float(z)),
                rgba=rgba,
                mass=mass,
            )
        )
        idx += 1

    objs.sort(key=lambda o: (o.pos[0], o.pos[1]))
    return objs


def build_mjcf_objects_xml(objs: Iterable[DetectedObject]) -> str:
    parts: list[str] = []
    for obj in objs:
        r, g, b, a = obj.rgba
        if obj.geom_type == "box":
            sx, sy, sz = obj.size
            geom = f'<geom type="box" size="{sx:.5f} {sy:.5f} {sz:.5f}" rgba="{r:.3f} {g:.3f} {b:.3f} {a:.3f}" mass="{obj.mass:.5f}"/>'
            site = '<site name="{name}_site" pos="0 0 0" size="0.01" rgba="0 1 0 0.3"/>'.format(name=obj.name)
        elif obj.geom_type == "cylinder":
            rad, hz, _ = obj.size
            geom = f'<geom type="cylinder" size="{rad:.5f} {hz:.5f}" rgba="{r:.3f} {g:.3f} {b:.3f} {a:.3f}" mass="{obj.mass:.5f}"/>'
            site = '<site name="{name}_site" pos="0 0 0" size="0.01" rgba="0 1 0 0.3"/>'.format(name=obj.name)
        else:
            continue

        x, y, z = obj.pos
        parts.append(f'<body name="{obj.name}" pos="{x:.5f} {y:.5f} {z:.5f}">')
        parts.append("<freejoint/>")
        parts.append(geom)
        parts.append(site)
        parts.append("</body>")
    return "\n    ".join(parts)


def generate_scene_from_photo(
    *,
    template_xml_path: str | Path,
    image_path: str | Path,
    output_xml_path: str | Path,
    meters_per_px: float = 0.0015,
    world_center_xy: tuple[float, float] = (0.5, 0.0),
    table_z: float = 0.0,
) -> tuple[list[DetectedObject], Path]:
    template_xml_path = Path(template_xml_path)
    output_xml_path = Path(output_xml_path)

    template = template_xml_path.read_text(encoding="utf-8")
    objs = detect_objects_from_photo(
        image_path,
        meters_per_px=meters_per_px,
        world_center_xy=world_center_xy,
        table_z=table_z,
    )
    objects_xml = build_mjcf_objects_xml(objs) or ""

    placeholder = '<body name="photo_objects_placeholder" pos="0 0 0"/>'
    if placeholder not in template:
        raise ValueError("模板缺少占位符 body: photo_objects_placeholder")

    updated = template.replace(placeholder, objects_xml)
    output_xml_path.write_text(updated, encoding="utf-8")
    return objs, output_xml_path

