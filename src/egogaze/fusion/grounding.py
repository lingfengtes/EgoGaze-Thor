"""Uncertainty-Aware Probabilistic Spatio-Temporal Referential Grounding Engine.

Computes posterior probability over candidate objects O = {o_1, ..., o_K} given:
- Gaze trajectory G_{t_s:t_e} (with I-VT FIXATION/SACCADE states & dispersion)
- Speech utterance U (parsed into semantic class, color, spatial cue, deictic flag)
- Temporal Scene Graph V (object bboxes, normalized coords, dwell times, track stability)

Log-linear posterior:
    log P(o_i | G, U, V) = w_g(sigma_g, fix_ratio, deictic) * S_gaze(o_i)
                         + w_p * S_spatial(o_i)
                         + w_s * S_semantic(o_i)
                         + w_t * S_temporal(o_i) - log Z

Supports `mode="probabilistic" | "speech_only" | "gaze_only" | "heuristic_baseline"`
directly for reproducible ablation benchmarks.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

from egogaze.core.schemas import (
    DetectedObject,
    GazeSample,
    GazeState,
    GroundingCandidateScore,
    GroundingResult,
    ParsedLexicalCues,
    SceneSnapshot,
    SpeechUtterance,
)


# Bilingual desktop/indoor lexical mappings
CLASS_SYNONYMS: Dict[str, List[str]] = {
    "cup": ["杯子", "水杯", "茶杯", "咖啡杯", "马克杯", "cup", "mug"],
    "bottle": ["瓶子", "水瓶", "饮料瓶", "矿泉水", "bottle"],
    "laptop": ["电脑", "笔记本", "笔记本电脑", "laptop", "macbook"],
    "keyboard": ["键盘", "keyboard"],
    "mouse": ["鼠标", "mouse"],
    "cell phone": ["手机", "电话", "phone", "cell phone", "smartphone"],
    "book": ["书", "书本", "课本", "本子", "笔记本子", "book", "notebook"],
    "pen": ["笔", "钢笔", "圆珠笔", "中性笔", "pen", "pencil"],
    "glasses": ["眼镜", "墨镜", "glasses"],
    "headphones": ["耳机", "头戴式耳机", "headphones", "earphones"],
    "backpack": ["书包", "背包", "包", "backpack", "bag"],
    "monitor": ["显示器", "屏幕", "monitor", "screen"],
    "chair": ["椅子", "凳子", "chair"],
    "bowl": ["碗", "bowl"],
    "remote": ["遥控器", "remote"],
    "box": ["盒子", "箱子", "纸盒", "box"],
    "person": ["人", "人物", "person"],
    "hand": ["手", "hand"],
}

COLOR_SYNONYMS: Dict[str, List[str]] = {
    "red": ["红色", "红的", "红", "red"],
    "orange": ["橙色", "橘色", "橙", "orange"],
    "yellow": ["黄色", "黄的", "黄", "yellow"],
    "green": ["绿色", "绿的", "绿", "green"],
    "blue": ["蓝色", "蓝的", "蓝", "blue"],
    "purple": ["紫色", "紫的", "紫", "purple"],
    "white": ["白色", "白的", "白", "white"],
    "gray": ["灰色", "灰的", "银色", "gray", "grey", "silver"],
    "black": ["黑色", "黑的", "黑", "black"],
}

SPATIAL_SYNONYMS: List[Tuple[str, List[str]]] = [
    ("left", ["左边", "左侧", "左面", "最左", "left"]),
    ("right", ["右边", "右侧", "右面", "最右", "right"]),
    ("center", ["中间", "中央", "正中", "center", "middle"]),
    ("top", ["上边", "上方", "上面", "顶部", "top", "upper"]),
    ("bottom", ["下边", "下方", "下面", "底部", "bottom", "lower"]),
    ("largest", ["最大", "大个", "largest", "biggest"]),
]

DEICTIC_WORDS = ["这个", "那个", "这一个", "那一个", "它", "这边", "那边", "我看着", "我盯着", "视线", "this", "that", "it", "looking at"]
TRACK_WORDS = ["追踪", "跟踪", "锁定", "盯住", "锁住", "标出", "高亮", "track", "lock", "follow", "highlight"]
STOP_WORDS = ["停止", "取消", "结束", "解除", "别追", "stop", "cancel", "unlock", "clear"]
MEMORIZE_WORDS = ["记住桌面", "记住当前", "记住场景", "保存桌面", "拍个快照", "remember desk", "memorize"]
DIFF_WORDS = ["有什么变化", "少了什么", "多了什么", "变了什么", "动了什么", "桌面变化", "what changed", "scene diff"]
VLM_TRIGGER_WORDS = [
    "是什么", "写的什么", "什么字", "上面写", "描述", "介绍一下", "看看", "干嘛", "内容",
    "标题", "品牌", "颜色", "文字", "读一下", "总结", "周围有什么", "分析",
    "手", "手指", "手势", "数字", "比划", "比了", "拿着", "手里", "举着",
    "看到", "看见", "眼前", "画面", "镜头", "面前", "前面", "桌面有什么", "桌上有什么",
    "穿", "戴", "表情", "姿势", "是谁", "在哪", "什么东西", "什么物体", "怎么了",
    "what is", "describe", "read", "explain", "tell me about", "summarize",
    "what am i looking", "finger", "hand", "gesture", "holding", "see", "in front",
]


def parse_lexical_cues(text: str) -> ParsedLexicalCues:
    """Extract structured linguistic cues from Chinese/English utterances."""
    raw = (text or "").strip()
    norm = raw.lower()
    compact = norm.replace(" ", "")

    target_classes: List[str] = []
    for canonical, kws in CLASS_SYNONYMS.items():
        if any(kw in norm or kw in compact for kw in kws):
            target_classes.append(canonical)

    color_cue: Optional[str] = None
    for canonical, kws in COLOR_SYNONYMS.items():
        if any(kw in norm or kw in compact for kw in kws):
            color_cue = canonical
            break

    spatial_cue: Optional[str] = None
    for canonical, kws in SPATIAL_SYNONYMS:
        if any(kw in norm or kw in compact for kw in kws):
            spatial_cue = canonical
            break

    is_deictic = any(w in norm or w in compact for w in DEICTIC_WORDS)
    has_track = any(w in norm or w in compact for w in TRACK_WORDS)
    has_stop = any(w in norm or w in compact for w in STOP_WORDS)
    has_memorize = any(w in norm or w in compact for w in MEMORIZE_WORDS)
    has_diff = any(w in norm or w in compact for w in DIFF_WORDS)
    needs_vlm = any(w in norm or w in compact for w in VLM_TRIGGER_WORDS)

    if has_memorize:
        intent_hint = "memorize_scene"
        needs_vlm = False
    elif has_diff:
        intent_hint = "diff_scene"
        needs_vlm = False
    elif has_stop and has_track:
        intent_hint = "stop_track"
        needs_vlm = False
    elif has_track:
        intent_hint = "track"
        needs_vlm = False
    elif needs_vlm:
        intent_hint = "describe_vlm"
    elif "几个" in compact or "多少" in compact or "how many" in norm:
        intent_hint = "count"
    else:
        intent_hint = "chat"

    return ParsedLexicalCues(
        raw_text=raw,
        intent_hint=intent_hint,
        target_classes=target_classes,
        spatial_cue=spatial_cue,
        color_cue=color_cue,
        is_deictic=is_deictic,
        needs_vlm=needs_vlm,
    )


def gaussian_box_integral_2d(
    gx: float,
    gy: float,
    sigma: float,
    box_norm: Tuple[float, float, float, float],
) -> float:
    """Exact closed-form integral of isotropic 2D Gaussian N((gx, gy), sigma^2 I) over [x1, x2] x [y1, y2].

    Fair to both small objects (pens, cups) and large objects (laptops, monitors) because
    it integrates probability mass inside the box while retaining smooth distance decay outside.
    """
    x1, y1, x2, y2 = box_norm
    inv_s = 1.0 / (max(sigma, 1e-4) * math.sqrt(2.0))
    px = 0.5 * (math.erf((x2 - gx) * inv_s) - math.erf((x1 - gx) * inv_s))
    py = 0.5 * (math.erf((y2 - gy) * inv_s) - math.erf((y1 - gy) * inv_s))
    mass = max(0.0, px * py)
    # Blend exact box probability mass with peak-aligned center proximity so tiny objects aren't starved
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    d2 = (cx - gx) ** 2 + (cy - gy) ** 2
    center_gauss = math.exp(-0.5 * d2 / (max(sigma, 1e-4) ** 2))
    # Area-normalized probability density bonus
    area = max((x2 - x1) * (y2 - y1), 1e-4)
    density = min(1.0, mass / (area * 8.0))
    return float(min(1.0, 0.55 * center_gauss + 0.45 * max(mass, density)))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def compute_spatial_score(obj: DetectedObject, spatial_cue: Optional[str], all_objs: List[DetectedObject]) -> float:
    """Continuous soft spatial prior in [0, 1] instead of brittle hard x < 0.5 cutoffs."""
    if not spatial_cue:
        return 0.5
    cx, cy = obj.center_norm
    if spatial_cue == "left":
        return _sigmoid((0.5 - cx) * 7.0)
    if spatial_cue == "right":
        return _sigmoid((cx - 0.5) * 7.0)
    if spatial_cue == "top":
        return _sigmoid((0.5 - cy) * 7.0)
    if spatial_cue == "bottom":
        return _sigmoid((cy - 0.5) * 7.0)
    if spatial_cue == "center":
        dist = math.hypot(cx - 0.5, cy - 0.5)
        return math.exp(-0.5 * (dist / 0.22) ** 2)
    if spatial_cue == "largest":
        max_area = max((o.area_norm for o in all_objs), default=1e-4)
        return float(obj.area_norm / max(max_area, 1e-4))
    return 0.5


def compute_semantic_score(obj: DetectedObject, cues: ParsedLexicalCues) -> float:
    """Semantic similarity score combining class match and color attribute match."""
    cls_score = 0.5
    if cues.target_classes:
        obj_cls = obj.class_name.lower()
        exact = any(tc == obj_cls or tc in obj_cls or obj_cls in tc for tc in cues.target_classes)
        cls_score = 1.0 if exact else 0.05

    color_score = 0.5
    if cues.color_cue:
        color_score = 1.0 if obj.color.lower() == cues.color_cue.lower() else 0.1

    if cues.target_classes and cues.color_cue:
        return 0.65 * cls_score + 0.35 * color_score
    if cues.target_classes:
        return cls_score
    if cues.color_cue:
        return color_score
    return 0.5


class ProbabilisticGroundingEngine:
    """Uncertainty-aware spatio-temporal multimodal referential grounding."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        cfg = cfg or {}
        self.pre_window_ns = int(float(cfg.get("pre_window_ms", 350.0)) * 1e6)
        self.post_window_ns = int(float(cfg.get("post_window_ms", 200.0)) * 1e6)
        self.gaze_sigma = float(cfg.get("gaze_sigma", 0.14))
        self.time_decay_tau_ns = float(cfg.get("time_decay_tau_ms", 450.0)) * 1e6
        weights = cfg.get("base_weights", {})
        self.w_gaze_base = float(weights.get("gaze", 2.2))
        self.w_spatial = float(weights.get("spatial", 1.8))
        self.w_semantic = float(weights.get("semantic", 2.4))
        self.w_temporal = float(weights.get("temporal", 0.8))
        self.deictic_gaze_boost = float(cfg.get("deictic_gaze_boost", 1.45))
        self.max_disp_full = float(cfg.get("max_dispersion_for_full_trust", 0.08))
        self.disp_cutoff = float(cfg.get("dispersion_cutoff", 0.24))
        self.min_posterior_lock = float(cfg.get("min_posterior_lock", 0.35))

    def _aggregate_gaze_window(
        self,
        gaze_samples: List[GazeSample],
        anchor_mono_ns: int,
    ) -> Tuple[List[Tuple[Tuple[float, float], float]], Optional[Tuple[float, float]], float, int, float]:
        """Filter gaze window to valid/fixation points, compute time weights, centroid, dispersion, and reliability."""
        if not gaze_samples:
            return [], None, 0.0, 0, 1.0

        valid = [g for g in gaze_samples if g.state != GazeState.INVALID]
        if not valid:
            return [], None, 0.0, 0, 1.0

        fixations = [g for g in valid if g.state == GazeState.FIXATION]
        active = fixations if fixations else valid
        fix_ratio = len(fixations) / float(len(gaze_samples))

        weighted_pts: List[Tuple[Tuple[float, float], float]] = []
        w_sum = 0.0
        cx_acc, cy_acc = 0.0, 0.0
        for g in active:
            dt = abs(anchor_mono_ns - g.capture_mono_ns)
            tw = math.exp(-dt / max(self.time_decay_tau_ns, 1.0)) * (1.0 if g.state == GazeState.FIXATION else 0.35)
            weighted_pts.append((g.filtered_xy, tw))
            w_sum += tw
            cx_acc += g.filtered_xy[0] * tw
            cy_acc += g.filtered_xy[1] * tw

        if w_sum <= 1e-9:
            return [], None, 0.0, 0, 1.0

        centroid = (cx_acc / w_sum, cy_acc / w_sum)
        # Spatial dispersion around weighted centroid
        dispersion = math.sqrt(
            sum(tw * ((pt[0] - centroid[0]) ** 2 + (pt[1] - centroid[1]) ** 2) for pt, tw in weighted_pts) / w_sum
        )

        # Reliability gate in [0, 1]: high when fixating tightly, decays smoothly during saccades / drift
        if dispersion <= self.max_disp_full:
            disp_trust = 1.0
        elif dispersion >= self.disp_cutoff:
            disp_trust = 0.0
        else:
            disp_trust = 1.0 - (dispersion - self.max_disp_full) / max(self.disp_cutoff - self.max_disp_full, 1e-6)

        state_trust = 0.25 + 0.75 * fix_ratio if fixations else 0.20
        reliability = max(0.0, min(1.0, disp_trust * state_trust))
        norm_pts = [(pt, tw / w_sum) for pt, tw in weighted_pts]
        return norm_pts, centroid, reliability, len(fixations), dispersion

    def ground(
        self,
        utterance: SpeechUtterance,
        scene: Optional[SceneSnapshot],
        gaze_window: List[GazeSample],
        mode: str = "probabilistic",
    ) -> GroundingResult:
        """Run multimodal referential grounding and return full explainable candidate breakdown."""
        t0 = time.perf_counter_ns()
        cues = parse_lexical_cues(utterance.text)
        anchor_ns = utterance.speech_end_mono_ns or utterance.final_mono_ns

        norm_pts, centroid, reliability, fix_count, dispersion = self._aggregate_gaze_window(
            gaze_window, anchor_ns
        )

        if not scene or not scene.objects:
            return GroundingResult(
                utterance_id=utterance.utterance_id,
                utterance_text=utterance.text,
                cues=cues,
                gaze_reliability=round(reliability, 4),
                fixation_count=fix_count,
                gaze_dispersion=round(dispersion, 4),
                gaze_centroid=centroid,
                fusion_ms=round((time.perf_counter_ns() - t0) / 1e6, 3),
                timestamp=time.time(),
            )

        # Modality weight selection (also enables clean ablation runs!)
        if mode == "speech_only":
            w_g, w_p, w_s, w_t = 0.0, self.w_spatial, self.w_semantic, 0.2
        elif mode == "gaze_only":
            w_g, w_p, w_s, w_t = self.w_gaze_base, 0.0, 0.0, 0.3
        elif mode == "heuristic_baseline":
            # Rule-based baseline: hard spatial conflict drop + nearest Euclidean gaze
            return self._ground_heuristic_baseline(
                utterance, cues, scene, centroid, reliability, fix_count, dispersion, t0
            )
        else:
            # Uncertainty-aware adaptive gaze weight
            boost = self.deictic_gaze_boost if (cues.is_deictic and not cues.spatial_cue) else 1.0
            # When explicit spatial cue conflicts strongly with gaze centroid, gently downweight gaze
            conflict_damp = 1.0
            if centroid is not None and cues.spatial_cue in ("left", "right", "top", "bottom"):
                gx, gy = centroid
                if (
                    (cues.spatial_cue == "left" and gx > 0.62)
                    or (cues.spatial_cue == "right" and gx < 0.38)
                    or (cues.spatial_cue == "top" and gy > 0.62)
                    or (cues.spatial_cue == "bottom" and gy < 0.38)
                ):
                    conflict_damp = 0.25
            w_g = self.w_gaze_base * reliability * boost * conflict_damp
            w_p = self.w_spatial if cues.spatial_cue else 0.0
            w_s = self.w_semantic if (cues.target_classes or cues.color_cue) else 0.0
            w_t = self.w_temporal

        raw_scores: List[Tuple[DetectedObject, float, float, float, float, float]] = []
        w_img, h_img = float(max(scene.width, 1)), float(max(scene.height, 1))

        for obj in scene.objects:
            x1, y1, x2, y2 = obj.bbox_xyxy
            box_norm = (x1 / w_img, y1 / h_img, x2 / w_img, y2 / h_img)

            if norm_pts:
                s_gaze = sum(
                    tw * gaussian_box_integral_2d(pt[0], pt[1], self.gaze_sigma, box_norm)
                    for pt, tw in norm_pts
                )
            else:
                s_gaze = 0.0

            s_spatial = compute_spatial_score(obj, cues.spatial_cue, scene.objects)
            s_semantic = compute_semantic_score(obj, cues)
            # Temporal prior: normalized dwell time + detector confidence + lock continuity
            dwell_norm = min(1.0, obj.gaze_dwell_ms / 800.0)
            s_temporal = 0.55 * dwell_norm + 0.35 * obj.confidence + (0.10 if obj.is_locked else 0.0)

            # Mixture-with-uniform-lapse likelihood prevents extreme log(0) vetoes during conflicts
            p_gaze = 0.92 * s_gaze + 0.08
            p_spatial = 0.95 * s_spatial + 0.05
            p_semantic = 0.95 * s_semantic + 0.05
            p_temporal = 0.90 * s_temporal + 0.10

            log_score = (
                w_g * math.log(p_gaze)
                + w_p * math.log(p_spatial)
                + w_s * math.log(p_semantic)
                + w_t * math.log(p_temporal)
            )
            raw_scores.append((obj, s_gaze, s_spatial, s_semantic, s_temporal, log_score))

        # Numerically stable softmax over candidates
        max_log = max(item[5] for item in raw_scores)
        exps = [math.exp(item[5] - max_log) for item in raw_scores]
        z = sum(exps) or 1.0

        candidates: List[GroundingCandidateScore] = []
        for (obj, sg, sp, ss, st, ls), ex in zip(raw_scores, exps):
            candidates.append(
                GroundingCandidateScore(
                    track_id=obj.track_id,
                    som_id=obj.som_id,
                    class_name=obj.class_name,
                    color=obj.color,
                    bbox_xyxy=obj.bbox_xyxy,
                    score_gaze=round(sg, 4),
                    score_spatial=round(sp, 4),
                    score_semantic=round(ss, 4),
                    score_temporal=round(st, 4),
                    weighted_log_score=round(ls, 4),
                    posterior_prob=round(ex / z, 4),
                )
            )

        candidates.sort(key=lambda c: c.posterior_prob, reverse=True)
        top = candidates[0] if candidates else None
        selected = top if (top and top.posterior_prob >= self.min_posterior_lock) else None

        # Open-World Null-Object Rejection:
        # When the user asks a pure deictic/visual question ("我正看着的是什么？" / "这个是什么") without naming
        # a class/spatial/color cue, and their valid gaze centroid falls OUTSIDE every detected YOLO box,
        # reject forcing a false closed-set box so System 2 VLM crops directly around the red gaze crosshair!
        if (
            selected is not None
            and mode == "probabilistic"
            and centroid is not None
            and reliability >= 0.30
            and not cues.target_classes
            and not cues.spatial_cue
            and not cues.color_cue
        ):
            gx, gy = centroid
            x1, y1, x2, y2 = selected.bbox_xyxy
            nx1, ny1, nx2, ny2 = x1 / w_img, y1 / h_img, x2 / w_img, y2 / h_img
            margin = 0.05
            gaze_near_box = (nx1 - margin <= gx <= nx2 + margin) and (ny1 - margin <= gy <= ny2 + margin)
            if not gaze_near_box and selected.score_gaze < 0.48:
                selected = None

        return GroundingResult(
            utterance_id=utterance.utterance_id,
            utterance_text=utterance.text,
            cues=cues,
            selected_track_id=selected.track_id if selected else None,
            selected_som_id=selected.som_id if selected else None,
            selected_class=selected.class_name if selected else None,
            selected_prob=selected.posterior_prob if selected else 0.0,
            gaze_reliability=round(reliability, 4),
            fixation_count=fix_count,
            gaze_dispersion=round(dispersion, 4),
            gaze_centroid=(round(centroid[0], 4), round(centroid[1], 4)) if centroid else None,
            candidates=candidates,
            fusion_ms=round((time.perf_counter_ns() - t0) / 1e6, 3),
            timestamp=time.time(),
        )

    def _ground_heuristic_baseline(
        self,
        utterance: SpeechUtterance,
        cues: ParsedLexicalCues,
        scene: SceneSnapshot,
        centroid: Optional[Tuple[float, float]],
        reliability: float,
        fix_count: int,
        dispersion: float,
        t0_ns: int,
    ) -> GroundingResult:
        """Rule-based nearest-gaze baseline for direct head-to-head ablation comparison."""
        pool = list(scene.objects)
        if cues.target_classes:
            filtered = [o for o in pool if o.class_name.lower() in cues.target_classes]
            if filtered:
                pool = filtered
        if cues.spatial_cue == "left":
            filtered = [o for o in pool if o.center_norm[0] < 0.5]
            if filtered:
                pool = filtered
        elif cues.spatial_cue == "right":
            filtered = [o for o in pool if o.center_norm[0] >= 0.5]
            if filtered:
                pool = filtered

        if centroid is not None:
            best = min(pool, key=lambda o: math.hypot(o.center_norm[0] - centroid[0], o.center_norm[1] - centroid[1]))
        else:
            best = max(pool, key=lambda o: o.confidence)

        cands = [
            GroundingCandidateScore(
                track_id=o.track_id,
                som_id=o.som_id,
                class_name=o.class_name,
                color=o.color,
                bbox_xyxy=o.bbox_xyxy,
                score_gaze=1.0 if o.track_id == best.track_id else 0.0,
                score_spatial=0.5,
                score_semantic=0.5,
                score_temporal=o.confidence,
                weighted_log_score=0.0,
                posterior_prob=1.0 if o.track_id == best.track_id else 0.0,
            )
            for o in scene.objects
        ]
        cands.sort(key=lambda c: c.posterior_prob, reverse=True)
        return GroundingResult(
            utterance_id=utterance.utterance_id,
            utterance_text=utterance.text,
            cues=cues,
            selected_track_id=best.track_id,
            selected_som_id=best.som_id,
            selected_class=best.class_name,
            selected_prob=1.0,
            gaze_reliability=round(reliability, 4),
            fixation_count=fix_count,
            gaze_dispersion=round(dispersion, 4),
            gaze_centroid=centroid,
            candidates=cands,
            fusion_ms=round((time.perf_counter_ns() - t0_ns) / 1e6, 3),
            timestamp=time.time(),
        )
