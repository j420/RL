# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tool Orchestration Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    ToolOrchestrationAction,
    ToolOrchestrationObservation,
    ToolOrchestrationState,
)


class ToolOrchestrationEnv(
    EnvClient[ToolOrchestrationAction, ToolOrchestrationObservation, ToolOrchestrationState]
):
    """
    Client for the Tool Orchestration Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with ToolOrchestrationEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_description)
        ...
        ...     result = client.step(ToolOrchestrationAction(
        ...         tool_name="database",
        ...         method="query",
        ...         parameters={"sql": "SELECT * FROM employees"}
        ...     ))
        ...     print(result.observation.tool_response)
    """

    def _step_payload(self, action: ToolOrchestrationAction) -> Dict:
        """Convert action to JSON payload for step message."""
        return {
            "tool_name": action.tool_name,
            "method": action.method,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ToolOrchestrationObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = ToolOrchestrationObservation(
            tool_response=obs_data.get("tool_response", {}),
            task_description=obs_data.get("task_description", ""),
            available_tools=obs_data.get("available_tools", []),
            workspace=obs_data.get("workspace", {}),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 10),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ToolOrchestrationState:
        """Parse server response into State object."""
        return ToolOrchestrationState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            total_reward=payload.get("total_reward", 0.0),
            tools_called=payload.get("tools_called", []),
            done=payload.get("done", False),
        )
