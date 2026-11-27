"""Unit tests for GazeEngine filters/calibration, Probabilistic Grounding, F-SoM, and TTS chunker."""

from __future__ import annotations

import time
import numpy as np

from egogaze.core.schemas import DetectedObject, GazeSample, GazeState, SceneSnapshot, SpeechUtterance
from egogaze.fusion.grounding import ProbabilisticGroundingEngine, gaussian_box_integral_2d, parse_lexical_cues
from egogaze.perception.fom_prompt import build_foveated_som_prompt
from egogaze.perception.gaze_engine import IVTFixationClassifier, OneEuroFilter2D, PolyGazeCalibrator
from egogaze.speech.tts_stream import StreamingSentenceChunker


def test_poly_calibrator_recovers_affine_and_nonlinear_warp():
    cal = PolyGazeCalibrator()
    # Synthetic nonlinear distortion: raw -> true screen target
    targets = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
    ]
    for tx, ty in targets:
        raw_x = 0.25 + 0.5 * tx + 0.04 * (tx - 0.5) ** 2
        raw_y = 0.30 + 0.4 * ty - 0.03 * (ty - 0.5) ** 2
        cal.add_sample((raw_x, raw_y), (tx, ty))
    rmse = cal.fit(alpha=1e-6)
    assert rmse < 0.015
    cx, cy = cal.transform((0.5, 0.5))
    assert abs(cx - 0.5) < 0.05 and abs(cy - 0.5) < 0.05


def test_one_euro_and_ivt_fixation_vs_saccade():
    f = OneEuroFilter2D()
    ivt = IVTFixationClassifier(saccade_velocity_thresh=1.2, min_fixation_ms=100.0)
    t0 = 1_000_000_000
    state = GazeState.INVALID
    for i in range(8):
        t_ns = t0 + i * 40_000_000  # 25Hz
        pt = f.filter((0.5 + (0.002 if i % 2 else -0.002), 0.5), t_ns / 1e9)
        state, _vel, dwell, _disp = ivt.update(pt, t_ns)
    assert state == GazeState.FIXATION
    assert dwell >= 100.0

    # Large sudden jump -> SACCADE
    jump_ns = t0 + 9 * 40_000_000
    s_state, vel, dwell2, _ = ivt.update((0.85, 0.5), jump_ns)
    assert s_state == GazeState.SACCADE
    assert vel > 1.2
    assert dwell2 == 0.0


def test_gaussian_box_integral_and_probabilistic_grounding():
    # Box containing gaze point should score much higher than distant box
    near = gaussian_box_integral_2d(0.2, 0.5, 0.14, (0.12, 0.40, 0.28, 0.62))
    far = gaussian_box_integral_2d(0.2, 0.5, 0.14, (0.72, 0.40, 0.88, 0.62))
    assert near > 0.7 and far < 0.05

    now_ns = time.monotonic_ns()
    scene = SceneSnapshot(
        frame_id=1,
        capture_mono_ns=now_ns,
        timestamp=time.time(),
        width=1000,
        height=500,
        objects=[
            DetectedObject(track_id=1, som_id=1, class_name="cup", confidence=0.9, bbox_xyxy=(120, 200, 260, 350), center_norm=(0.19, 0.55), area_norm=0.04, color="Blue"),
            DetectedObject(track_id=2, som_id=2, class_name="cup", confidence=0.9, bbox_xyxy=(720, 200, 860, 350), center_norm=(0.79, 0.55), area_norm=0.04, color="Red"),
        ],
    )
    engine = ProbabilisticGroundingEngine()

    # Case A: Pure deictic "追踪这个" while fixating on right red cup -> track_id=2
    gaze_right = [
        GazeSample(capture_mono_ns=now_ns - i * 40_000_000, timestamp=time.time(), raw_xy=(0.79, 0.55), calibrated_xy=(0.79, 0.55), filtered_xy=(0.79, 0.55), state=GazeState.FIXATION, dwell_ms=300.0)
        for i in range(5)
    ]
    utt_deictic = SpeechUtterance(utterance_id="u1", text="追踪这个", speech_start_mono_ns=now_ns - 300_000_000, speech_end_mono_ns=now_ns, final_mono_ns=now_ns)
    res_a = engine.ground(utt_deictic, scene, gaze_right)
    assert res_a.selected_track_id == 2
    assert res_a.gaze_reliability > 0.8

    # Case B: Explicit spatial speech "追踪左边的杯子" overrides conflicting gaze on right -> track_id=1
    utt_conflict = SpeechUtterance(utterance_id="u2", text="追踪左边的杯子", speech_start_mono_ns=now_ns - 300_000_000, speech_end_mono_ns=now_ns, final_mono_ns=now_ns)
    res_b = engine.ground(utt_conflict, scene, gaze_right)
    assert res_b.selected_track_id == 1

    # Case C: Foveated Set-of-Mark prompt generation
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    pack = build_foveated_som_prompt(frame, scene, res_a)
    assert len(pack.fsom_b64) > 100
    assert pack.roi_b64 is not None and len(pack.roi_b64) > 100
    assert "[2] cup" in pack.scene_legend


def test_lexical_parser_and_tts_chunker():
    cues = parse_lexical_cues("帮我看看左边那个红色的杯子上写的什么")
    assert "cup" in cues.target_classes
    assert cues.spatial_cue == "left"
    assert cues.color_cue == "red"
    assert cues.needs_vlm is True

    chunker = StreamingSentenceChunker(min_first_chars=6)
    out = []
    for tok in ["好的，", "我看到桌面上", "有一个蓝色的马克杯。", "旁边是笔记本电脑。"]:
        out.extend(chunker.push(tok))
    tail = chunker.flush()
    if tail:
        out.append(tail)
    assert len(out) >= 2
    assert "马克杯。" in "".join(out)
