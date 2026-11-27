"""Fast-Slow Intent Router deciding System 1 Fast Action vs System 2 F-SoM VLM vs LLM Dialogue."""

from __future__ import annotations

from egogaze.core.schemas import GroundingResult, RouteType


class FastSlowRouter:
    """Routes each grounded utterance to the optimal execution tier for latency vs reasoning depth."""

    def route(self, grounding: GroundingResult) -> RouteType:
        hint = grounding.cues.intent_hint
        if hint in ("track", "stop_track", "count", "memorize_scene", "diff_scene"):
            return RouteType.SYSTEM1_FAST
        if hint == "describe_vlm" or grounding.cues.needs_vlm:
            return RouteType.SYSTEM2_VLM_FSOM
        # If user uses a deictic pronoun while fixating on a grounded object, invoke System 2 VLM
        if grounding.cues.is_deictic and grounding.selected_track_id is not None:
            return RouteType.SYSTEM2_VLM_FSOM
        return RouteType.DIALOGUE_LLM
