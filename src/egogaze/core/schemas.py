"""Core typed schemas with unified monotonic nanosecond timestamps."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class GazeState(str, Enum):
    FIXATION = "FIXATION"
    SACCADE = "SACCADE"
    INVALID = "INVALID"


class RouteType(str, Enum):
    SYSTEM1_FAST = "SYSTEM1_FAST"
    SYSTEM2_VLM_FSOM = "SYSTEM2_VLM_FSOM"
    DIALOGUE_LLM = "DIALOGUE_LLM"


class GazeSample(BaseModel):
    capture_mono_ns: int
    timestamp: float
    raw_xy: Tuple[float, float]
    calibrated_xy: Tuple[float, float]
    filtered_xy: Tuple[float, float]
    velocity: float = 0.0
    state: GazeState = GazeState.INVALID
    dwell_ms: float = 0.0
    dispersion: float = 0.0
    confidence: float = 1.0
    frame_id: int = 0


class DetectedObject(BaseModel):
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    center_norm: Tuple[float, float]
    area_norm: float
    color: str = "Unknown"
    gaze_dwell_ms: float = 0.0
    velocity_norm: float = 0.0
    som_id: int = 0
    is_locked: bool = False


class SceneSnapshot(BaseModel):
    frame_id: int
    capture_mono_ns: int
    timestamp: float
    width: int
    height: int
    objects: List[DetectedObject] = Field(default_factory=list)


class SpeechUtterance(BaseModel):
    utterance_id: str
    text: str
    is_final: bool = True
    speech_start_mono_ns: int
    speech_end_mono_ns: int
    final_mono_ns: int
    source: str = "voice"  # voice | text | replay | audio_file
    vad_ms: float = 0.0
    asr_ms: float = 0.0


class ParsedLexicalCues(BaseModel):
    raw_text: str
    intent_hint: str = "query"  # track | stop_track | describe_vlm | count | chat
    target_classes: List[str] = Field(default_factory=list)
    requested_target: Optional[str] = None  # Free-text noun phrase (e.g. "苹果", "魔方") for open-world/VLM verification
    spatial_cue: Optional[str] = None  # left | right | center | top | bottom | closest | largest
    color_cue: Optional[str] = None
    is_deictic: bool = False
    needs_vlm: bool = False


class GroundingCandidateScore(BaseModel):
    track_id: int
    som_id: int
    class_name: str
    color: str
    bbox_xyxy: Tuple[int, int, int, int]
    score_gaze: float
    score_spatial: float
    score_semantic: float
    score_temporal: float
    weighted_log_score: float
    posterior_prob: float


class GroundingResult(BaseModel):
    utterance_id: str
    utterance_text: str
    cues: ParsedLexicalCues
    selected_track_id: Optional[int] = None
    selected_som_id: Optional[int] = None
    selected_class: Optional[str] = None
    selected_prob: float = 0.0
    gaze_reliability: float = 0.0
    fixation_count: int = 0
    gaze_dispersion: float = 0.0
    gaze_centroid: Optional[Tuple[float, float]] = None
    candidates: List[GroundingCandidateScore] = Field(default_factory=list)
    fusion_ms: float = 0.0
    timestamp: float = 0.0


class PipelineLatencyTrace(BaseModel):
    utterance_id: str
    utterance_text: str
    route: RouteType
    backend: str
    vad_ms: float = 0.0
    asr_ms: float = 0.0
    fusion_ms: float = 0.0
    ttft_ms: float = 0.0
    ttfa_ms: float = 0.0
    total_ms: float = 0.0
    timestamp: float = 0.0


class DialogueEvent(BaseModel):
    utterance_id: str
    role: str  # user | assistant | system
    text: str
    route: Optional[RouteType] = None
    backend: Optional[str] = None
    is_partial: bool = False
    fsom_image_b64: Optional[str] = None
    roi_image_b64: Optional[str] = None
    timestamp: float = 0.0
    meta: Dict[str, Any] = Field(default_factory=dict)
