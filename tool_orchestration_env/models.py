# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Tool Orchestration Environment.

Defines Action, Observation, and State types for a multi-tool orchestration
RL environment where agents chain together 6 simulated business tools
(database, email, filestore, calculator, calendar, validator) to complete
increasingly complex workflows.
"""

from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator

# Evaluation requires all scores/rewards strictly in (0, 1) — never exactly 0.0 or 1.0
SCORE_MIN = 0.01
SCORE_MAX = 0.99


def clamp_score(v: float) -> float:
    """Clamp a score to (0, 1) exclusive."""
    return round(max(SCORE_MIN, min(SCORE_MAX, float(v))), 4)


AVAILABLE_TOOLS = [
    "database",
    "email",
    "filestore",
    "calculator",
    "calendar",
    "validator",
]


class ToolOrchestrationAction(Action):
    """Action for the Tool Orchestration environment.

    Each action represents a single tool call. The agent selects which tool
    to use, which method to invoke, and provides the method-specific parameters.

    Example:
        ToolOrchestrationAction(
            tool_name="database",
            method="query",
            parameters={"sql": "SELECT * FROM employees WHERE hire_date > '2026-03-01'"}
        )
    """

    tool_name: str = Field(
        ...,
        description="Tool to invoke. One of: database, email, filestore, calculator, calendar, validator",
    )
    method: str = Field(
        ...,
        description="Method name to call on the selected tool",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific parameters as key-value pairs",
    )


class ToolOrchestrationObservation(Observation):
    """Observation returned by the Tool Orchestration environment.

    Contains the tool response, current task context, accumulated workspace
    of previous outputs, and progress indicators.
    """

    tool_response: Dict[str, Any] = Field(
        default_factory=dict,
        description="Result from the tool call, or error dict if call failed",
    )
    task_description: str = Field(
        default="",
        description="Text description of the current task",
    )
    available_tools: List[str] = Field(
        default_factory=lambda: list(AVAILABLE_TOOLS),
        description="Tool names the agent can invoke",
    )
    workspace: Dict[str, Any] = Field(
        default_factory=dict,
        description="Accumulated outputs from previous steps, keyed as 'step_0', 'step_1', etc.",
    )
    step_number: int = Field(
        default=0,
        description="Current step number in the episode",
    )
    max_steps: int = Field(
        default=10,
        description="Maximum steps allowed for the current task",
    )

    @field_validator("reward", mode="before")
    @classmethod
    def clamp_reward_value(cls, v):
        """Ensure reward is always strictly between 0 and 1."""
        if v is None:
            return SCORE_MIN
        return clamp_score(float(v))


class ToolOrchestrationState(State):
    """Extended state for the Tool Orchestration environment.

    Tracks episode metadata beyond the base State fields (episode_id, step_count).
    State.extra='allow' permits these additional fields.
    """

    task_id: str = Field(
        default="",
        description="ID of the current task: 'easy', 'medium', or 'hard'",
    )
    total_reward: float = Field(
        default=0.01,
        description="Accumulated reward for the current episode",
    )
    tools_called: List[str] = Field(
        default_factory=list,
        description="Ordered list of tool names called so far in this episode",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has terminated",
    )
