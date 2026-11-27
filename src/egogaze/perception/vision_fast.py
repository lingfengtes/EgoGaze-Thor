"""System 1 Fast Open-Vocabulary Desktop/Indoor Vision, Egocentric Tracker & Appearance Re-ID.

Key tracking improvements over naive IoU:
1. Hybrid association (IoU + normalized center proximity + box scale + HSV color histogram)
   so fast head/ego-motion (IoU = 0) does NOT cause track_id churn.
2. Compatible class groups prevent 1-frame YOLO class flicker (e.g. cup <-> bottle, laptop <-> tv)
   from breaking an active track.
3. LockedTargetSignature with HSV appearance Re-ID automatically re-acquires a locked object
   after hand occlusion, fast pan, or temporary out-of-frame departure.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from egogaze.core.schemas import DetectedObject, GazeSample, GazeState, SceneSnapshot

logger = logging.getLogger("egogaze.vision")

_COMPAT_GROUPS: List[set[str]] = [
    {"apple", "orange", "sports ball", "donut", "cake", "bowl"},
    {"cup", "bottle", "wine glass", "bowl", "vase"},
    {"laptop", "tv", "monitor"},
    {"cell phone", "remote"},
    {"book", "notebook", "handbag", "backpack", "box"},
    {"backpack", "handbag", "suitcase"},
    {"chair", "couch"},
    {"fork", "knife", "spoon", "toothbrush", "pen"},
    {"banana", "carrot"},
]


def _classes_compatible(c1: str, c2: str) -> bool:
    a, b = c1.lower(), c2.lower()
    if a == b:
        return True
    return any(a in grp and b in grp for grp in _COMPAT_GROUPS)


def _class_mismatch_penalty(c1: str, c2: str) -> float:
    """Graduated class penalty: exact=0.0, confusion-group=0.06, cross-class=0.24 (never hard-drops strong visual matches)."""
    a, b = c1.lower(), c2.lower()
    if a == b:
        return 0.0
    if any(a in grp and b in grp for grp in _COMPAT_GROUPS):
        return 0.06
    return 0.24


def compute_hsv_histogram(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Compute L2-normalized 16x8 Hue-Saturation histogram for fast appearance Re-ID."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    try:
        small = cv2.resize(crop_bgr, (48, 48), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32)
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L2)
        return hist
    except Exception:
        return None


def _hist_similarity(h1: Optional[np.ndarray], h2: Optional[np.ndarray]) -> float:
    if h1 is None or h2 is None:
        return 0.5
    corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, (corr + 1.0) * 0.5))


@dataclass
class LockedTargetSignature:
    track_id: int
    class_name: str
    color: str
    hsv_hist: Optional[np.ndarray]
    last_center_norm: Tuple[float, float]
    last_area_norm: float
    last_bbox: Tuple[int, int, int, int]
    miss_frames: int = 0
    alias_label: Optional[str] = None     # User-facing semantic label (e.g. "苹果")
    hud_alias: Optional[str] = None       # ASCII label for OpenCV HUD (e.g. "apple")


class _OpenWorldPatchTracker:
    """Fast local-window NCC + HSV appearance tracker for VLM-localized targets missed by YOLO (<1.5ms/frame)."""

    SYNTHETIC_TRACK_ID = 90001

    def __init__(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
        alias_label: str,
        hud_alias: str,
    ) -> None:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy
        x1, y1 = max(0, min(w - 2, x1)), max(0, min(h - 2, y1))
        x2, y2 = max(x1 + 8, min(w - 1, x2)), max(y1 + 8, min(h - 1, y2))
        self.bbox: Tuple[int, int, int, int] = (x1, y1, x2, y2)
        self.alias_label = alias_label
        self.hud_alias = hud_alias
        crop = frame_bgr[y1:y2, x1:x2]
        self.color = detect_dominant_color(crop)
        self.hsv_hist = compute_hsv_histogram(crop)
        self.scale = 0.5
        self.template = self._make_template(frame_bgr, self.bbox)
        self.miss_frames = 0
        self.confidence = 0.85

    def _make_template(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        crop = frame_bgr[y1:y2, x1:x2]
        tw = max(12, int((x2 - x1) * self.scale))
        th = max(12, int((y2 - y1) * self.scale))
        return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

    def update(self, frame_bgr: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self.bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 8 or self.template is None or self.template.size == 0:
            return None

        # Adaptive search window grows slightly when target moves fast or briefly occluded
        pad_factor = min(2.2, 0.85 + 0.08 * self.miss_frames)
        pad_x = max(60, int(bw * pad_factor))
        pad_y = max(60, int(bh * pad_factor))
        sx1, sy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        sx2, sy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        roi = frame_bgr[sy1:sy2, sx1:sx2]
        if roi.size == 0:
            self.miss_frames += 1
            return None

        rw = max(self.template.shape[1] + 2, int(roi.shape[1] * self.scale))
        rh = max(self.template.shape[0] + 2, int(roi.shape[0] * self.scale))
        roi_s = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_AREA)
        if roi_s.shape[0] < self.template.shape[0] or roi_s.shape[1] < self.template.shape[1]:
            self.miss_frames += 1
            return None

        res = cv2.matchTemplate(roi_s, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        nx1 = int(sx1 + max_loc[0] / self.scale)
        ny1 = int(sy1 + max_loc[1] / self.scale)
        nx2 = min(w - 1, nx1 + bw)
        ny2 = min(h - 1, ny1 + bh)
        nx1 = max(0, nx2 - bw)
        ny1 = max(0, ny2 - bh)

        cand_crop = frame_bgr[ny1:ny2, nx1:nx2]
        cand_hist = compute_hsv_histogram(cand_crop)
        app_sim = _hist_similarity(self.hsv_hist, cand_hist)
        combined = 0.65 * max(0.0, float(max_val)) + 0.35 * app_sim

        if combined < 0.42:
            self.miss_frames += 1
            return None

        self.bbox = (nx1, ny1, nx2, ny2)
        self.miss_frames = 0
        self.confidence = round(min(0.95, max(0.45, combined)), 3)
        # Slowly adapt template and color histogram when match is crisp
        if combined >= 0.68:
            new_tpl = self._make_template(frame_bgr, self.bbox)
            if new_tpl.shape == self.template.shape:
                self.template = cv2.addWeighted(self.template, 0.90, new_tpl, 0.10, 0)
            if self.hsv_hist is not None and cand_hist is not None:
                blended = 0.90 * self.hsv_hist + 0.10 * cand_hist
                cv2.normalize(blended, blended, alpha=1.0, norm_type=cv2.NORM_L2)
                self.hsv_hist = blended
        return self.bbox, self.confidence


class _EgocentricTracker:
    """Motion-, scale-, and appearance-aware multi-object tracker resilient to egocentric head motion."""

    def __init__(self, match_thresh: float = 0.28, max_age: int = 60) -> None:
        self.match_thresh = match_thresh
        self.max_age = max_age
        self._next_id = 1
        self._tracks: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / float(area_a + area_b - inter)

    def update(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], str, Optional[np.ndarray]]],
        img_w: int,
        img_h: int,
    ) -> List[int]:
        if not detections:
            for tid in list(self._tracks.keys()):
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    self._tracks.pop(tid, None)
            return []

        track_ids = list(self._tracks.keys())
        assigned: List[Optional[int]] = [None] * len(detections)
        matched_tids = set()
        w_f, h_f = float(max(img_w, 1)), float(max(img_h, 1))

        if track_ids:
            cost = np.ones((len(track_ids), len(detections)), dtype=np.float32)
            for r, tid in enumerate(track_ids):
                tr = self._tracks[tid]
                tb = tr["bbox"]
                tc = tr["cls"]
                tcx = 0.5 * (tb[0] + tb[2]) / w_f
                tcy = 0.5 * (tb[1] + tb[3]) / h_f
                ta = max(1.0, float((tb[2] - tb[0]) * (tb[3] - tb[1])))
                for c, (db, dc, dhist) in enumerate(detections):
                    iou = self._iou(tb, db)
                    dcx = 0.5 * (db[0] + db[2]) / w_f
                    dcy = 0.5 * (db[1] + db[3]) / h_f
                    dist = math.hypot(dcx - tcx, dcy - tcy)
                    # Survives fast ego-motion up to ~25% screen displacement per frame
                    prox = math.exp(-dist / 0.18)
                    da = max(1.0, float((db[2] - db[0]) * (db[3] - db[1])))
                    size_sim = min(ta, da) / max(ta, da)
                    app_sim = _hist_similarity(tr["hist"], dhist)
                    cls_penalty = _class_mismatch_penalty(tc, dc)
                    # Cross-class matches require strong visual + spatial continuity
                    if cls_penalty >= 0.20 and iou < 0.25 and (prox < 0.70 or app_sim < 0.65):
                        continue
                    score = 0.40 * iou + 0.32 * prox + 0.12 * size_sim + 0.16 * app_sim - cls_penalty
                    cost[r, c] = max(0.0, 1.0 - score)

            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                sim = 1.0 - float(cost[r, c])
                if sim >= self.match_thresh:
                    tid = track_ids[r]
                    assigned[c] = tid
                    matched_tids.add(tid)
                    prev_hist = self._tracks[tid]["hist"]
                    new_hist = detections[c][2]
                    if prev_hist is not None and new_hist is not None:
                        blended = 0.8 * prev_hist + 0.2 * new_hist
                        cv2.normalize(blended, blended, alpha=1.0, norm_type=cv2.NORM_L2)
                    else:
                        blended = new_hist if new_hist is not None else prev_hist
                    self._tracks[tid] = {
                        "bbox": detections[c][0],
                        "cls": detections[c][1],
                        "hist": blended,
                        "age": 0,
                    }

        for tid in track_ids:
            if tid not in matched_tids:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    self._tracks.pop(tid, None)

        out_ids: List[int] = []
        for idx, tid in enumerate(assigned):
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "bbox": detections[idx][0],
                    "cls": detections[idx][1],
                    "hist": detections[idx][2],
                    "age": 0,
                }
            out_ids.append(tid)
        return out_ids


def detect_dominant_color(crop_bgr: np.ndarray) -> str:
    """Fast HSV dominant color classification."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "Unknown"
    try:
        small = cv2.resize(crop_bgr, (48, 48), interpolation=cv2.INTER_NEAREST)
    except Exception:
        return "Unknown"

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    valid = v > 25
    total = int(np.count_nonzero(valid))
    if total < 10:
        return "Black"

    ranges = {
        "Red": [((0, 60, 50), (10, 255, 255)), ((168, 60, 50), (180, 255, 255))],
        "Orange": [((11, 60, 60), (22, 255, 255))],
        "Yellow": [((23, 60, 60), (36, 255, 255))],
        "Green": [((37, 45, 50), (82, 255, 255))],
        "Blue": [((90, 55, 50), (132, 255, 255))],
        "Purple": [((133, 45, 50), (167, 255, 255))],
        "White": [((0, 0, 175), (180, 42, 255))],
        "Gray": [((0, 0, 65), (180, 42, 174))],
        "Black": [((0, 0, 0), (180, 255, 64))],
    }

    best_color = "Gray"
    best_ratio = 0.0
    for name, bounds in ranges.items():
        mask = np.zeros_like(h, dtype=np.uint8)
        for lo, hi in bounds:
            mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        ratio = float(np.count_nonzero((mask > 0) & valid)) / total
        bonus = 0.05 if name not in ("Gray", "Black", "White") else 0.0
        if ratio + bonus > best_ratio:
            best_ratio = ratio + bonus
            best_color = name
    return best_color


class FastVisionEngine:
    """Open-vocabulary/COCO detector + Egocentric Tracker + Appearance Re-ID Target Lock."""

    def __init__(self, cfg: Dict[str, Any], demo_mode: bool = False) -> None:
        self.cfg = cfg
        self.demo_mode = demo_mode
        self.model_name = cfg.get("model", "data/models/yolo11s.pt")
        self.fallback_model = cfg.get("fallback_model", "data/models/yolo11n.pt")
        self.device = str(cfg.get("device", "0"))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("conf", 0.25))
        self.iou = float(cfg.get("iou", 0.45))
        self.tracker = cfg.get("tracker", "bytetrack.yaml")
        self.vocab: List[str] = list(cfg.get("vocab", []))

        self._model = None
        self._is_open_vocab = False
        self._backend_name = "demo_synthetic"
        self._locked_track_id: Optional[int] = None
        self._locked_sig: Optional[LockedTargetSignature] = None
        self._patch_tracker: Optional[_OpenWorldPatchTracker] = None
        self._latest_hists: Dict[int, Optional[np.ndarray]] = {}
        self._latest_objs: Dict[int, DetectedObject] = {}
        self._track_state: Dict[int, Dict[str, Any]] = {}
        self._last_mono_ns: Optional[int] = None
        self._ego_tracker = _EgocentricTracker(match_thresh=0.28, max_age=60)

        if not self.demo_mode:
            self._init_model()

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def locked_track_id(self) -> Optional[int]:
        return self._locked_track_id

    @property
    def locked_signature(self) -> Optional[LockedTargetSignature]:
        return self._locked_sig

    def set_locked_track(
        self,
        track_id: Optional[int],
        alias_label: Optional[str] = None,
        hud_alias: Optional[str] = None,
    ) -> None:
        self._locked_track_id = track_id
        self._patch_tracker = None
        if track_id is None:
            self._locked_sig = None
            return
        obj = self._latest_objs.get(track_id)
        if obj is not None:
            self._locked_sig = LockedTargetSignature(
                track_id=track_id,
                class_name=obj.class_name,
                color=obj.color,
                hsv_hist=self._latest_hists.get(track_id),
                last_center_norm=obj.center_norm,
                last_area_norm=obj.area_norm,
                last_bbox=obj.bbox_xyxy,
                miss_frames=0,
                alias_label=alias_label,
                hud_alias=hud_alias,
            )

    def set_locked_bbox(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
        alias_label: str = "target",
        hud_alias: str = "target",
    ) -> Tuple[int, Optional[DetectedObject]]:
        """Lock onto a VLM-localized bbox. Snaps to an overlapping YOLO object if IoU>=0.35, else starts OpenWorldPatchTracker."""
        # Check if an existing YOLO detection already strongly overlaps this bbox
        best_obj: Optional[DetectedObject] = None
        best_iou = 0.0
        for obj in self._latest_objs.values():
            iou = _EgocentricTracker._iou(bbox_xyxy, obj.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_obj = obj
        if best_obj is not None and best_iou >= 0.35:
            self.set_locked_track(best_obj.track_id, alias_label=alias_label, hud_alias=hud_alias)
            return best_obj.track_id, best_obj

        # Otherwise launch lightweight visual patch tracker for open-world object missed by YOLO
        h, w = frame_bgr.shape[:2]
        tracker = _OpenWorldPatchTracker(frame_bgr, bbox_xyxy, alias_label=alias_label, hud_alias=hud_alias)
        self._patch_tracker = tracker
        tid = _OpenWorldPatchTracker.SYNTHETIC_TRACK_ID
        self._locked_track_id = tid
        x1, y1, x2, y2 = tracker.bbox
        cx = ((x1 + x2) / 2.0) / float(max(w, 1))
        cy = ((y1 + y2) / 2.0) / float(max(h, 1))
        area = max(1, (x2 - x1) * (y2 - y1)) / float(max(w * h, 1))
        self._locked_sig = LockedTargetSignature(
            track_id=tid,
            class_name=hud_alias,
            color=tracker.color,
            hsv_hist=tracker.hsv_hist,
            last_center_norm=(round(cx, 4), round(cy, 4)),
            last_area_norm=round(area, 4),
            last_bbox=tracker.bbox,
            miss_frames=0,
            alias_label=alias_label,
            hud_alias=hud_alias,
        )
        return tid, None

    def inject_vocab_terms(self, terms: List[str]) -> bool:
        if not self._is_open_vocab or self._model is None:
            return False
        added = False
        for term in terms:
            t = term.strip().lower()
            if t and t not in self.vocab:
                self.vocab.append(t)
                added = True
        if added:
            try:
                self._model.set_classes(self.vocab)
                return True
            except Exception as exc:
                logger.warning("Failed to update open-vocab classes: %s", exc)
        return False

    def _init_model(self) -> None:
        try:
            import torch
            import torchvision

            if torch.cuda.is_available():
                try:
                    _b = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device="cuda")
                    _s = torch.tensor([0.9], device="cuda")
                    torchvision.ops.nms(_b, _s, 0.5)
                except NotImplementedError:
                    _orig_nms = torchvision.ops.nms

                    def _safe_nms(boxes, scores, iou_threshold):
                        if boxes.is_cuda:
                            return _orig_nms(boxes.cpu(), scores.cpu(), iou_threshold).to(boxes.device)
                        return _orig_nms(boxes, scores, iou_threshold)

                    torchvision.ops.nms = _safe_nms
                    logger.info("Applied Jetson CPU-NMS shim for torchvision::nms (backbone remains 100%% CUDA)")
        except Exception:
            pass

        try:
            from ultralytics import YOLO, YOLOWorld
        except ImportError:
            logger.warning("ultralytics not installed; running FastVisionEngine in demo mode")
            self.demo_mode = True
            return

        try:
            if "world" in self.model_name.lower():
                self._model = YOLOWorld(self.model_name)
                if self.vocab:
                    self._model.set_classes(self.vocab)
                self._is_open_vocab = True
                self._backend_name = f"yoloworld:{self.model_name}"
            else:
                self._model = YOLO(self.model_name)
                if hasattr(self._model, "set_classes") and self.vocab and "yoloe" in self.model_name.lower():
                    self._model.set_classes(self.vocab)
                    self._is_open_vocab = True
                self._backend_name = f"yolo:{self.model_name}"
            logger.info("FastVisionEngine loaded %s (open_vocab=%s)", self._backend_name, self._is_open_vocab)
        except Exception as exc:
            logger.warning("Primary vision model %s failed (%s); falling back to %s", self.model_name, exc, self.fallback_model)
            try:
                self._model = YOLO(self.fallback_model)
                self._is_open_vocab = False
                self._backend_name = f"yolo_fallback:{self.fallback_model}"
            except Exception as exc2:
                logger.error("Fallback vision model failed (%s); switching to demo mode", exc2)
                self.demo_mode = True

    def _reacquire_locked_target(
        self,
        objs: List[DetectedObject],
        hists: Dict[int, Optional[np.ndarray]],
    ) -> None:
        """Keep locked target alive across ID switches, class flickers, and brief occlusions via appearance Re-ID."""
        if self._locked_track_id is None:
            return

        # 1. Direct track_id match
        direct = next((o for o in objs if o.track_id == self._locked_track_id), None)
        if direct is not None:
            new_h = hists.get(direct.track_id)
            if self._locked_sig is None:
                self._locked_sig = LockedTargetSignature(
                    track_id=direct.track_id,
                    class_name=direct.class_name,
                    color=direct.color,
                    hsv_hist=new_h,
                    last_center_norm=direct.center_norm,
                    last_area_norm=direct.area_norm,
                    last_bbox=direct.bbox_xyxy,
                    miss_frames=0,
                )
            else:
                if self._locked_sig.hsv_hist is not None and new_h is not None:
                    blended = 0.88 * self._locked_sig.hsv_hist + 0.12 * new_h
                    cv2.normalize(blended, blended, alpha=1.0, norm_type=cv2.NORM_L2)
                    self._locked_sig.hsv_hist = blended
                elif new_h is not None:
                    self._locked_sig.hsv_hist = new_h
                self._locked_sig.last_center_norm = direct.center_norm
                self._locked_sig.last_area_norm = direct.area_norm
                self._locked_sig.last_bbox = direct.bbox_xyxy
                self._locked_sig.miss_frames = 0
            return

        # 2. Track ID temporarily missing -> try Appearance + Spatial + Class Re-ID!
        if self._locked_sig is None:
            return
        self._locked_sig.miss_frames += 1

        best_obj: Optional[DetectedObject] = None
        best_score = 0.0
        sig = self._locked_sig
        # Allow wider spatial search radius the longer the target has been absent
        sigma_pos = min(0.55, 0.22 + 0.008 * sig.miss_frames)

        for o in objs:
            cls_pen = _class_mismatch_penalty(sig.class_name, o.class_name)
            app_sim = _hist_similarity(sig.hsv_hist, hists.get(o.track_id))
            dist = math.hypot(o.center_norm[0] - sig.last_center_norm[0], o.center_norm[1] - sig.last_center_norm[1])
            pos_sim = math.exp(-dist / sigma_pos)
            size_sim = min(o.area_norm, sig.last_area_norm) / max(o.area_norm, sig.last_area_norm, 1e-4)
            color_bonus = 0.08 if o.color.lower() == sig.color.lower() else 0.0
            # Cross-class Re-ID (e.g. apple flickering to mouse/clock) requires strong visual + spatial match
            if cls_pen >= 0.20 and (app_sim < 0.65 or pos_sim < 0.55):
                continue
            score = 0.46 * app_sim + 0.34 * pos_sim + 0.14 * size_sim + color_bonus - cls_pen
            if score > best_score:
                best_score = score
                best_obj = o

        if best_obj is not None and best_score >= 0.48:
            logger.info(
                "Re-acquired locked target %s (alias=%s): old_tid=%s -> new_tid=%s (det=%s, reid_score=%.2f, missed=%df)",
                sig.class_name,
                sig.alias_label,
                self._locked_track_id,
                best_obj.track_id,
                best_obj.class_name,
                best_score,
                sig.miss_frames,
            )
            self._locked_track_id = best_obj.track_id
            sig.track_id = best_obj.track_id
            sig.last_center_norm = best_obj.center_norm
            sig.last_area_norm = best_obj.area_norm
            sig.last_bbox = best_obj.bbox_xyxy
            sig.miss_frames = 0

    def _step_patch_tracker(
        self,
        frame_bgr: np.ndarray,
        objs: List[DetectedObject],
        hists: Dict[int, Optional[np.ndarray]],
    ) -> None:
        """Update OpenWorldPatchTracker if active, handing off to YOLO automatically when a matching box appears."""
        pt = self._patch_tracker
        if pt is None:
            return
        h, w = frame_bgr.shape[:2]
        res = pt.update(frame_bgr)
        if res is None:
            return
        bbox, conf = res
        # Check seamless handoff to a newly appeared YOLO detection overlapping the tracked patch
        best_obj: Optional[DetectedObject] = None
        best_iou = 0.0
        for o in objs:
            iou = _EgocentricTracker._iou(bbox, o.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_obj = o
        if best_obj is not None and best_iou >= 0.38:
            app_sim = _hist_similarity(pt.hsv_hist, hists.get(best_obj.track_id))
            if app_sim >= 0.55:
                logger.info(
                    "OpenWorldPatchTracker handed off '%s' to YOLO track_id=%d (%s, iou=%.2f, app=%.2f)",
                    pt.alias_label,
                    best_obj.track_id,
                    best_obj.class_name,
                    best_iou,
                    app_sim,
                )
                alias_lbl, hud_lbl = pt.alias_label, pt.hud_alias
                self._patch_tracker = None
                self._locked_track_id = best_obj.track_id
                self._locked_sig = LockedTargetSignature(
                    track_id=best_obj.track_id,
                    class_name=best_obj.class_name,
                    color=best_obj.color,
                    hsv_hist=hists.get(best_obj.track_id) or pt.hsv_hist,
                    last_center_norm=best_obj.center_norm,
                    last_area_norm=best_obj.area_norm,
                    last_bbox=best_obj.bbox_xyxy,
                    miss_frames=0,
                    alias_label=alias_lbl,
                    hud_alias=hud_lbl,
                )
                return

        x1, y1, x2, y2 = bbox
        cx = ((x1 + x2) / 2.0) / float(max(w, 1))
        cy = ((y1 + y2) / 2.0) / float(max(h, 1))
        area = max(1, (x2 - x1) * (y2 - y1)) / float(max(w * h, 1))
        tid = _OpenWorldPatchTracker.SYNTHETIC_TRACK_ID
        objs.append(
            DetectedObject(
                track_id=tid,
                class_name=pt.hud_alias,
                confidence=conf,
                bbox_xyxy=bbox,
                center_norm=(round(cx, 4), round(cy, 4)),
                area_norm=round(area, 4),
                color=pt.color,
            )
        )
        hists[tid] = pt.hsv_hist

    def _update_track_dynamics(
        self,
        objs: List[DetectedObject],
        hists: Dict[int, Optional[np.ndarray]],
        gaze: Optional[GazeSample],
        capture_mono_ns: int,
    ) -> None:
        dt_ms = 50.0 if self._last_mono_ns is None else max((capture_mono_ns - self._last_mono_ns) / 1e6, 1.0)
        dt_s = dt_ms / 1000.0
        self._last_mono_ns = capture_mono_ns

        self._latest_hists = hists
        self._latest_objs = {o.track_id: o for o in objs}
        self._reacquire_locked_target(objs, hists)

        active_ids = set()
        for idx, obj in enumerate(objs, start=1):
            obj.som_id = idx
            obj.is_locked = (self._locked_track_id is not None and obj.track_id == self._locked_track_id)
            active_ids.add(obj.track_id)

            prev = self._track_state.get(obj.track_id, {"dwell_ms": 0.0, "center": obj.center_norm})
            vel = math.hypot(obj.center_norm[0] - prev["center"][0], obj.center_norm[1] - prev["center"][1]) / dt_s

            dwell = prev["dwell_ms"]
            if gaze is not None and gaze.state == GazeState.FIXATION:
                gx, gy = gaze.filtered_xy
                cx, cy = obj.center_norm
                if math.hypot(gx - cx, gy - cy) < max(0.08, math.sqrt(obj.area_norm) * 0.75):
                    dwell = min(dwell + dt_ms, 5000.0)
                else:
                    dwell = max(0.0, dwell - dt_ms * 0.5)
            else:
                dwell = max(0.0, dwell - dt_ms * 0.25)

            obj.gaze_dwell_ms = round(dwell, 1)
            obj.velocity_norm = round(vel, 4)
            self._track_state[obj.track_id] = {"dwell_ms": dwell, "center": obj.center_norm}

        for tid in list(self._track_state.keys()):
            if tid not in active_ids:
                self._track_state.pop(tid, None)

    def _synthetic_demo_scene(
        self,
        frame_bgr: np.ndarray,
        gaze: Optional[GazeSample],
        capture_mono_ns: int,
        frame_id: int,
    ) -> SceneSnapshot:
        h, w = frame_bgr.shape[:2]
        t = time.monotonic()
        dx = int(12 * math.sin(t * 0.8))
        raw_objs = [
            (1, "cup", "Blue", (int(w * 0.14), int(h * 0.42), int(w * 0.27), int(h * 0.68)), 0.93),
            (2, "laptop", "Gray", (int(w * 0.34), int(h * 0.28), int(w * 0.65), int(h * 0.66)), 0.96),
            (3, "cup", "Red", (int(w * 0.70) + dx, int(h * 0.44), int(w * 0.83) + dx, int(h * 0.70)), 0.91),
            (4, "book", "Yellow", (int(w * 0.18), int(h * 0.71), int(w * 0.44), int(h * 0.90)), 0.88),
            (5, "cell phone", "Black", (int(w * 0.68), int(h * 0.72), int(w * 0.79), int(h * 0.89)), 0.90),
        ]
        objs: List[DetectedObject] = []
        hists: Dict[int, Optional[np.ndarray]] = {}
        for tid, cls_name, color, (x1, y1, x2, y2), conf in raw_objs:
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            area = max((x2 - x1) * (y2 - y1), 1) / float(w * h)
            objs.append(
                DetectedObject(
                    track_id=tid,
                    class_name=cls_name,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_norm=(round(cx, 4), round(cy, 4)),
                    area_norm=round(area, 4),
                    color=color,
                )
            )
            hists[tid] = compute_hsv_histogram(frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)])
        self._step_patch_tracker(frame_bgr, objs, hists)
        self._update_track_dynamics(objs, hists, gaze, capture_mono_ns)
        return SceneSnapshot(
            frame_id=frame_id,
            capture_mono_ns=capture_mono_ns,
            timestamp=time.time(),
            width=w,
            height=h,
            objects=objs,
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        gaze: Optional[GazeSample],
        capture_mono_ns: int,
        frame_id: int,
    ) -> SceneSnapshot:
        if self.demo_mode or self._model is None:
            return self._synthetic_demo_scene(frame_bgr, gaze, capture_mono_ns, frame_id)

        h, w = frame_bgr.shape[:2]
        results = self._model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        r = results[0]
        raw_items: List[Tuple[Tuple[int, int, int, int], str, float, Optional[np.ndarray], np.ndarray]] = []
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = str(r.names.get(cls_id, f"cls_{cls_id}"))
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame_bgr[y1:y2, x1:x2]
                hist = compute_hsv_histogram(crop)
                raw_items.append(((x1, y1, x2, y2), cls_name, conf, hist, crop))

        assigned_ids = self._ego_tracker.update([(b, c, hs) for b, c, _, hs, _ in raw_items], w, h)

        objs: List[DetectedObject] = []
        hists: Dict[int, Optional[np.ndarray]] = {}
        for ((x1, y1, x2, y2), cls_name, conf, hist, crop), tid in zip(raw_items, assigned_ids):
            color = detect_dominant_color(crop)
            cx = ((x1 + x2) / 2.0) / float(w)
            cy = ((y1 + y2) / 2.0) / float(h)
            area = ((x2 - x1) * (y2 - y1)) / float(w * h)
            objs.append(
                DetectedObject(
                    track_id=tid,
                    class_name=cls_name,
                    confidence=round(conf, 3),
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_norm=(round(cx, 4), round(cy, 4)),
                    area_norm=round(area, 4),
                    color=color,
                )
            )
            hists[tid] = hist

        self._step_patch_tracker(frame_bgr, objs, hists)
        objs.sort(key=lambda o: (o.center_norm[0], o.center_norm[1]))
        self._update_track_dynamics(objs, hists, gaze, capture_mono_ns)
        return SceneSnapshot(
            frame_id=frame_id,
            capture_mono_ns=capture_mono_ns,
            timestamp=time.time(),
            width=w,
            height=h,
            objects=objs,
        )
