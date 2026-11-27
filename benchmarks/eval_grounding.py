"""Synthetic & Recorded Ablation Benchmark for Probabilistic Referential Grounding.

Compares 4 methods on desktop referring scenarios:
1. speech_only
2. gaze_only
3. heuristic_baseline (rule-based nearest-gaze + hard-filter baseline)
4. probabilistic (Ours: uncertainty-aware spatio-temporal fusion)
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from egogaze.core.schemas import DetectedObject, GazeSample, GazeState, SceneSnapshot, SpeechUtterance
from egogaze.fusion.grounding import ProbabilisticGroundingEngine


def _make_desktop_scene() -> SceneSnapshot:
    now_ns = time.monotonic_ns()
    objs = [
        DetectedObject(track_id=1, som_id=1, class_name="cup", confidence=0.92, bbox_xyxy=(120, 220, 260, 380), center_norm=(0.19, 0.55), area_norm=0.04, color="Blue", gaze_dwell_ms=50.0),
        DetectedObject(track_id=2, som_id=2, class_name="laptop", confidence=0.96, bbox_xyxy=(340, 150, 640, 370), center_norm=(0.49, 0.48), area_norm=0.12, color="Gray", gaze_dwell_ms=50.0),
        DetectedObject(track_id=3, som_id=3, class_name="cup", confidence=0.90, bbox_xyxy=(700, 230, 830, 390), center_norm=(0.77, 0.57), area_norm=0.04, color="Red", gaze_dwell_ms=50.0),
        DetectedObject(track_id=4, som_id=4, class_name="book", confidence=0.88, bbox_xyxy=(180, 390, 440, 495), center_norm=(0.31, 0.82), area_norm=0.05, color="Yellow", gaze_dwell_ms=50.0),
        DetectedObject(track_id=5, som_id=5, class_name="cell phone", confidence=0.91, bbox_xyxy=(680, 395, 790, 490), center_norm=(0.74, 0.82), area_norm=0.02, color="Black", gaze_dwell_ms=50.0),
    ]
    return SceneSnapshot(frame_id=1, capture_mono_ns=now_ns, timestamp=time.time(), width=1000, height=540, objects=objs)


def _gaze_series(points: List[Tuple[float, float, GazeState]], base_ns: int) -> List[GazeSample]:
    return [
        GazeSample(
            capture_mono_ns=base_ns - (len(points) - i) * 40_000_000,
            timestamp=time.time(),
            raw_xy=(x, y),
            calibrated_xy=(x, y),
            filtered_xy=(x, y),
            state=st,
            dwell_ms=300.0 if st == GazeState.FIXATION else 0.0,
        )
        for i, (x, y, st) in enumerate(points)
    ]


def run_benchmark() -> Dict[str, float]:
    engine = ProbabilisticGroundingEngine()
    scene = _make_desktop_scene()
    base = scene.capture_mono_ns

    # Cases: (category, utterance_text, gaze_points, ground_truth_track_id)
    cases = [
        # 1. Pure deictic pronoun (gaze resolves ambiguity)
        ("deictic", "追踪这个", [(0.19, 0.55, GazeState.FIXATION)] * 6, 1),
        ("deictic", "帮我看看那个是什么", [(0.74, 0.82, GazeState.FIXATION)] * 6, 5),
        ("deictic", "锁定它", [(0.31, 0.82, GazeState.FIXATION)] * 6, 4),
        # 2. Same-class disambiguation by gaze
        ("same_class_gaze", "追踪那个杯子", [(0.77, 0.57, GazeState.FIXATION)] * 6, 3),
        ("same_class_gaze", "锁定水杯", [(0.19, 0.55, GazeState.FIXATION)] * 6, 1),
        # 3. Spatial + class disambiguation even with gaze looking elsewhere (conflict)
        ("gaze_speech_conflict", "追踪左边的杯子", [(0.77, 0.57, GazeState.FIXATION)] * 6, 1),
        ("gaze_speech_conflict", "锁定右边的杯子", [(0.19, 0.55, GazeState.FIXATION)] * 6, 3),
        # 4. Saccade / noisy gaze where speech color/class should win cleanly
        ("saccade_noise", "追踪红色的杯子", [(0.15, 0.2, GazeState.SACCADE), (0.20, 0.55, GazeState.SACCADE), (0.18, 0.50, GazeState.SACCADE)], 3),
        ("saccade_noise", "看看黄色的书", [(0.75, 0.80, GazeState.SACCADE), (0.72, 0.82, GazeState.SACCADE)], 4),
        # 5. Subtle off-center gaze + color synergy (gaze drifted closer to laptop/red cup, color specifies blue cup)
        ("synergy", "看看那个蓝色的杯子", [(0.55, 0.52, GazeState.FIXATION)] * 5, 1),
        # 6. Center spatial cue + noisy edge gaze (heuristic ignores "center" and picks gaze-nearest)
        ("center_spatial", "锁定中间的电脑", [(0.19, 0.55, GazeState.FIXATION)] * 5, 2),
        # 7. Deictic with mixed saccade + final fixation (time-weighted fixation wins over earlier saccade)
        (
            "temporal_fixation",
            "追踪这个",
            [
                (0.19, 0.55, GazeState.SACCADE),
                (0.35, 0.55, GazeState.SACCADE),
                (0.77, 0.57, GazeState.FIXATION),
                (0.76, 0.57, GazeState.FIXATION),
                (0.77, 0.56, GazeState.FIXATION),
            ],
            3,
        ),
    ]

    methods = ["speech_only", "gaze_only", "heuristic_baseline", "probabilistic"]
    results: Dict[str, float] = {}
    print(f"{'Method':<22} | {'Accuracy':>8} | {'Correct/Total':>13} | {'Avg ms':>8}")
    print("-" * 62)
    for m in methods:
        correct = 0
        total_ms = 0.0
        for idx, (_cat, text, gpts, gt_id) in enumerate(cases):
            utt = SpeechUtterance(
                utterance_id=f"c{idx}",
                text=text,
                speech_start_mono_ns=base - 400_000_000,
                speech_end_mono_ns=base,
                final_mono_ns=base,
            )
            res = engine.ground(utt, scene, _gaze_series(gpts, base), mode=m)
            total_ms += res.fusion_ms
            if res.selected_track_id == gt_id:
                correct += 1
        acc = correct / len(cases)
        results[m] = acc
        print(f"{m:<22} | {acc*100:7.1f}% | {correct:>6}/{len(cases):<6} | {total_ms/len(cases):7.3f}")
    return results


if __name__ == "__main__":
    run_benchmark()
