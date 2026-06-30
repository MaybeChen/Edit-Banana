"""Backward-compatible import path for the VLM prompt planner."""

from .vlm.prompt_planner import PROMPT_GROUP_KEYS, VLMPromptPlanner

__all__ = ["PROMPT_GROUP_KEYS", "VLMPromptPlanner"]
