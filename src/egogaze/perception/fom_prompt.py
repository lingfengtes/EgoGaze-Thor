"""Foveated Set-of-Mark (F-SoM) visual prompt generator for System 2 Edge VLM.

Renders:
1. Context image with clean numbered badges ([1], [2], ...), highlighted grounded target,
   and a subtle foveated gaze focus indicator (reducing VLM spatial grounding errors).
2. Optional high-resolution foveated ROI crop of the grounded/fixated object so the VLM
   can read small text/labels without encoding a massive full-resolution frame.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from egogaze.core.schemas import GroundingResult, SceneSnapshot


@dataclass
class FoveatedPromptPack:
    fsom_bgr: np.ndarray
    roi_bgr: Optional[np.ndarray]
    fsom_b64: str
    roi_b64: Optional[str]
    scene_legend: str


def _encode_jpeg_b64(img_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _resize_max_width(img_bgr: np.ndarray, max_width: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w <= max_width:
        return img_bgr
    scale = max_width / float(w)
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def build_foveated_som_prompt(
    frame_bgr: np.ndarray,
    scene: Optional[SceneSnapshot],
    grounding: Optional[GroundingResult],
    full_frame_max_width: int = 768,
    roi_size: int = 448,
) -> FoveatedPromptPack:
    """Build Foveated Set-of-Mark annotated full-frame + optional high-res ROI crop."""
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]
    selected_tid = grounding.selected_track_id if grounding else None
    gaze_xy = grounding.gaze_centroid if grounding else None

    legend_lines = []
    roi_bgr: Optional[np.ndarray] = None

    if scene and scene.objects:
        for obj in scene.objects:
            x1, y1, x2, y2 = obj.bbox_xyxy
            is_sel = (selected_tid is not None and obj.track_id == selected_tid)
            box_color = (0, 215, 255) if is_sel else (80, 220, 120)
            thickness = 3 if is_sel else 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)

            badge = f"[{obj.som_id}] {obj.class_name}"
            (tw, th), baseline = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            by1 = max(0, y1 - th - 10)
            cv2.rectangle(canvas, (x1, by1), (min(w - 1, x1 + tw + 10), by1 + th + 10), box_color, -1)
            cv2.putText(
                canvas,
                badge,
                (x1 + 5, by1 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (15, 15, 15),
                2,
                cv2.LINE_AA,
            )

            marker = " <-- USER TARGET" if is_sel else ""
            legend_lines.append(
                f"[{obj.som_id}] {obj.class_name} (color={obj.color}, pos=({obj.center_norm[0]:.2f},{obj.center_norm[1]:.2f})){marker}"
            )

            if is_sel:
                # Extract padded foveated ROI around the grounded target
                bw = x2 - x1
                bh = y2 - y1
                pad_x = int(bw * 0.18)
                pad_y = int(bh * 0.18)
                rx1 = max(0, x1 - pad_x)
                ry1 = max(0, y1 - pad_y)
                rx2 = min(w, x2 + pad_x)
                ry2 = min(h, y2 + pad_y)
                crop = frame_bgr[ry1:ry2, rx1:rx2]
                if crop.size > 0:
                    roi_bgr = _resize_max_width(crop, roi_size)

    # Overlay foveated gaze crosshair if valid gaze centroid exists
    if gaze_xy is not None:
        gx = int(np.clip(gaze_xy[0], 0.0, 1.0) * w)
        gy = int(np.clip(gaze_xy[1], 0.0, 1.0) * h)
        r_outer = max(18, int(min(w, h) * 0.045))
        cv2.circle(canvas, (gx, gy), r_outer, (0, 60, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (gx, gy), 5, (0, 60, 255), -1, cv2.LINE_AA)
        cv2.line(canvas, (gx - r_outer - 6, gy), (gx - 8, gy), (0, 60, 255), 2)
        cv2.line(canvas, (gx + 8, gy), (gx + r_outer + 6, gy), (0, 60, 255), 2)
        cv2.line(canvas, (gx, gy - r_outer - 6), (gx, gy - 8), (0, 60, 255), 2)
        cv2.line(canvas, (gx, gy + 8), (gx, gy + r_outer + 6), (0, 60, 255), 2)

        # Open-world gaze target: no YOLO box matched the user's fixation -> crop fovea centered on gaze crosshair!
        if roi_bgr is None:
            half = max(110, int(min(w, h) * 0.28))
            rx1, ry1 = max(0, gx - half), max(0, gy - half)
            rx2, ry2 = min(w, gx + half), min(h, gy + half)
            crop = frame_bgr[ry1:ry2, rx1:rx2].copy()
            if crop.size > 0:
                # Draw subtle gaze crosshair inside ROI crop too so VLM locks onto the exact fixated item
                cx_r = gx - rx1
                cy_r = gy - ry1
                cv2.circle(crop, (cx_r, cy_r), 14, (0, 60, 255), 2, cv2.LINE_AA)
                cv2.circle(crop, (cx_r, cy_r), 3, (0, 60, 255), -1, cv2.LINE_AA)
                roi_bgr = _resize_max_width(crop, roi_size)
            legend_lines.insert(
                0,
                f"[OPEN-WORLD GAZE TARGET] 红色十字注视点位于 ({gaze_xy[0]:.2f}, {gaze_xy[1]:.2f})，"
                "不在任何绿框物体内！第二张图是红色十字注视点周围的高清局部放大图，"
                "请严格回答红色十字正指着的物体/内容，切勿误答旁边绿框里的物体！",
            )

    fsom_resized = _resize_max_width(canvas, full_frame_max_width)
    return FoveatedPromptPack(
        fsom_bgr=fsom_resized,
        roi_bgr=roi_bgr,
        fsom_b64=_encode_jpeg_b64(fsom_resized),
        roi_b64=_encode_jpeg_b64(roi_bgr) if roi_bgr is not None else None,
        scene_legend="\n".join(legend_lines) if legend_lines else "(No detected objects in System 1)",
    )
