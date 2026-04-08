# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tool Orchestration Environment Implementation.

An RL environment where agents chain 6 simulated business tools
(database, email, filestore, calculator, calendar, validator) to
complete multi-step workflows with per-step partial reward signals.
"""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import ToolOrchestrationAction, ToolOrchestrationObservation, ToolOrchestrationState
except ImportError:
    from models import ToolOrchestrationAction, ToolOrchestrationObservation, ToolOrchestrationState

from .grader import Grader
from .tasks import TASKS, Task, get_task
from .tools.calculator import CalculatorTool
from .tools.calendar import CalendarTool
from .tools.database import DatabaseTool
from .tools.email import EmailTool
from .tools.filestore import FileStoreTool
from .tools.validator import ValidatorTool


TOOL_NAMES = ["database", "email", "filestore", "calculator", "calendar", "validator"]


class ToolOrchestrationEnvironment(Environment):
    """
    Multi-tool orchestration RL environment.

    Agents receive a business task and must chain together tool calls
    to complete it. Provides per-step partial reward signals for RL training.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._tools: Dict[str, Any] = {
            "database": DatabaseTool(),
            "email": EmailTool(),
            "filestore": FileStoreTool(),
            "calculator": CalculatorTool(),
            "calendar": CalendarTool(),
            "validator": ValidatorTool(),
        }
        self._grader = Grader()
        self._task: Optional[Task] = None
        self._episode_id: str = ""
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False
        self._episode_history: List[Dict[str, Any]] = []
        self._workspace: Dict[str, Any] = {}
        self._tools_called: List[str] = []
        self._graded: bool = False
        self._last_grader_result: Optional[Dict[str, Any]] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolOrchestrationObservation:
        """Reset the environment for a new episode."""
        task_id = kwargs.get("task_id", None)

        if seed is not None:
            random.seed(seed)

        # Pick task
        if task_id and task_id in TASKS:
            self._task = get_task(task_id)
        else:
            self._task = random.choice(list(TASKS.values()))

        # Generate episode ID
        self._episode_id = episode_id or str(uuid4())

        # Reset all tools
        for tool in self._tools.values():
            tool.reset(task_id=self._task.task_id)

        # Clear episode state
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._episode_history = []
        self._workspace = {}
        self._tools_called = []
        self._graded = False
        self._last_grader_result = None

        return ToolOrchestrationObservation(
            tool_response={"message": "Environment ready. Complete the task described below."},
            task_description=self._task.description,
            available_tools=list(TOOL_NAMES),
            workspace={},
            step_number=0,
            max_steps=self._task.max_steps,
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: ToolOrchestrationAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ToolOrchestrationObservation:
        """Execute one tool call and return the result with reward."""
        if self._task is None:
            return ToolOrchestrationObservation(
                tool_response={"error": "Environment not initialized. Call reset() first."},
                done=True,
                reward=0.0,
            )

        if self._done:
            return ToolOrchestrationObservation(
                tool_response={"error": "Episode is done. Call reset() to start a new one."},
                task_description=self._task.description,
                available_tools=list(TOOL_NAMES),
                workspace=self._workspace,
                step_number=self._step_count,
                max_steps=self._task.max_steps,
                done=True,
                reward=self._normalize_reward(),
            )

        # Validate tool name
        if action.tool_name not in TOOL_NAMES:
            self._step_count += 1
            self._total_reward -= 0.1
            self._tools_called.append(action.tool_name)

            if self._step_count >= self._task.max_steps:
                self._done = True
                self._run_grader()

            return ToolOrchestrationObservation(
                tool_response={"error": f"Unknown tool '{action.tool_name}'. Available: {TOOL_NAMES}"},
                task_description=self._task.description,
                available_tools=list(TOOL_NAMES),
                workspace=self._workspace,
                step_number=self._step_count,
                max_steps=self._task.max_steps,
                done=self._done,
                reward=self._normalize_reward(),
            )

        # Execute the tool
        tool = self._tools[action.tool_name]
        result = tool.execute(action.method, action.parameters)

        # Store in workspace
        self._workspace[f"step_{self._step_count}"] = {
            "tool": action.tool_name,
            "method": action.method,
            "result": result,
        }

        # Record history
        history_entry = {
            "action": {
                "tool_name": action.tool_name,
                "method": action.method,
                "parameters": action.parameters,
            },
            "result": result,
        }
        self._episode_history.append(history_entry)
        self._tools_called.append(action.tool_name)

        # Calculate step reward
        step_reward = self._calculate_step_reward(action, result)
        self._total_reward += step_reward

        # Increment step
        self._step_count += 1

        # Check completion (heuristic)
        if self._check_completion():
            self._done = True
            self._run_grader()

        # Check max steps
        if self._step_count >= self._task.max_steps:
            self._done = True
            if not self._graded:
                self._run_grader()

        return ToolOrchestrationObservation(
            tool_response=result,
            task_description=self._task.description,
            available_tools=list(TOOL_NAMES),
            workspace=self._workspace,
            step_number=self._step_count,
            max_steps=self._task.max_steps,
            done=self._done,
            reward=self._normalize_reward(),
        )

    @property
    def state(self) -> ToolOrchestrationState:
        """Get current environment state."""
        return ToolOrchestrationState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task.task_id if self._task else "",
            total_reward=self._normalize_reward(),
            tools_called=list(self._tools_called),
            done=self._done,
        )

    def get_last_grader_result(self) -> Optional[Dict[str, Any]]:
        """Expose grader result for the /grader endpoint."""
        return self._last_grader_result

    def get_tool_states(self) -> Dict[str, Any]:
        """Snapshot tool states for grader use."""
        states: Dict[str, Any] = {}
        email_tool = self._tools["email"]
        if hasattr(email_tool, "get_outbox"):
            states["email_outbox"] = email_tool.get_outbox()
        filestore_tool = self._tools["filestore"]
        if hasattr(filestore_tool, "get_files"):
            states["files"] = filestore_tool.get_files()
        return states

    # ----- Private helpers -----

    def _calculate_step_reward(self, action: ToolOrchestrationAction, result: Dict[str, Any]) -> float:
        """Per-step partial reward signal."""
        step_reward = 0.0
        optimal = self._task.optimal_tool_sequence

        # Reward for correct tool at correct sequence position
        if self._step_count < len(optimal):
            expected = optimal[self._step_count]
            actual = f"{action.tool_name}.{action.method}"
            if actual == expected:
                step_reward += 0.1  # Exact match
            elif action.tool_name == expected.split(".")[0]:
                step_reward += 0.05  # Right tool, wrong method

        # Reward for successful execution (no error)
        if "error" not in result:
            step_reward += 0.05
        else:
            step_reward -= 0.02  # Penalty for errors

        # Penalty for duplicate calls (exact same tool+method+params)
        if self._is_duplicate_call(action):
            step_reward -= 0.05

        # Cap step reward
        return max(-0.1, min(0.2, step_reward))

    def _is_duplicate_call(self, action: ToolOrchestrationAction) -> bool:
        """Check if this exact call was made before."""
        current = (action.tool_name, action.method, str(sorted(action.parameters.items())))
        for entry in self._episode_history[:-1]:  # Exclude last entry (current one)
            prev = entry.get("action", {})
            prev_key = (
                prev.get("tool_name", ""),
                prev.get("method", ""),
                str(sorted(prev.get("parameters", {}).items())),
            )
            if current == prev_key:
                return True
        return False

    def _check_completion(self) -> bool:
        """Heuristic: task is complete when tool call counts match the optimal sequence.

        Requires that each tool appears at least as many times as in the
        optimal sequence before declaring completion. This prevents premature
        completion (e.g., sending 1 email when 4 are needed).
        """
        if not self._task:
            return False

        # Count required tool occurrences from optimal sequence
        from collections import Counter
        required_counts = Counter()
        for entry in self._task.optimal_tool_sequence:
            tool = entry.split(".")[0]
            required_counts[tool] += 1

        # Count actual calls
        actual_counts = Counter(self._tools_called)

        # All required tools must have been called at least the required number of times
        for tool, count in required_counts.items():
            if actual_counts.get(tool, 0) < count:
                return False
        return True

    def _run_grader(self) -> None:
        """Run the grader on the current episode."""
        if self._graded or not self._task:
            return
        self._graded = True

        tool_states = self.get_tool_states()
        grader_result = self._grader.grade(
            self._task.task_id,
            self._episode_history,
            tool_states,
        )
        self._last_grader_result = {
            "task_id": self._task.task_id,
            **grader_result,
        }

        # Completion bonus
        completion_bonus = grader_result["score"] * 0.5
        self._total_reward += completion_bonus

    def _normalize_reward(self) -> float:
        """Normalize total reward to 0.0-1.0 range."""
        return round(max(0.0, min(1.0, self._total_reward)), 4)
