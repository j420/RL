# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Tool Orchestration Environment.

Standard OpenEnv endpoints (provided by create_app):
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - GET /health: Health check
    - WS /ws: WebSocket for persistent sessions

Hackathon-specific endpoints (added below):
    - GET /tasks: List all tasks with action schemas
    - POST /grader: Grade last completed episode
    - POST /baseline: Run inference baseline on all tasks
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

try:
    from ..models import ToolOrchestrationAction, ToolOrchestrationObservation
    from .tool_orchestration_env_environment import ToolOrchestrationEnvironment
    from .tasks import list_tasks
except (ImportError, SystemError):
    from models import ToolOrchestrationAction, ToolOrchestrationObservation
    from server.tool_orchestration_env_environment import ToolOrchestrationEnvironment
    from server.tasks import list_tasks


# Create the core OpenEnv app
app = create_app(
    ToolOrchestrationEnvironment,
    ToolOrchestrationAction,
    ToolOrchestrationObservation,
    env_name="tool_orchestration_env",
    max_concurrent_envs=1,
)


# =====================================================================
# Hackathon-specific endpoints
# =====================================================================

# Keep a reference to the environment instance for /grader and /baseline
_env_instance: ToolOrchestrationEnvironment | None = None


def _get_env() -> ToolOrchestrationEnvironment:
    """Get or create a shared environment instance for hackathon endpoints."""
    global _env_instance
    if _env_instance is None:
        _env_instance = ToolOrchestrationEnvironment()
    return _env_instance


@app.get("/tasks")
async def get_tasks():
    """Return all tasks with their action schemas."""
    return {"tasks": list_tasks()}


@app.post("/grader")
async def run_grader():
    """Grade the last completed episode."""
    env = _get_env()
    result = env.get_last_grader_result()
    if result is None:
        return {"error": "No completed episode", "score": 0.0}
    return result


@app.post("/baseline")
async def run_baseline():
    """Run inference baseline on all tasks. Requires API credentials."""
    api_base = os.environ.get("API_BASE_URL", "")
    model_name = os.environ.get("MODEL_NAME", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not api_base or not hf_token:
        return {"error": "API credentials not configured. Set API_BASE_URL and HF_TOKEN.", "status": "skipped"}

    # Import inline to avoid hard dependency when not running baseline
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai package not installed", "status": "skipped"}

    env = _get_env()
    client = OpenAI(base_url=api_base, api_key=hf_token)
    scores = {}

    system_prompt = (
        "You are an AI agent interacting with a tool orchestration environment. "
        "You have access to these tools:\n"
        "- database: query(sql), insert(table, data)\n"
        "- email: send(to, subject, body, attachment), search(query), list_inbox()\n"
        "- filestore: read(path), write(path, content), list(directory)\n"
        "- calculator: compute(expression), group_sum(data, group_by, aggregate), date_diff(date1, date2)\n"
        "- calendar: get_events(user, date_range), find_free_slots(users, date_range, duration_minutes), create_event(title, attendees, start, end)\n"
        "- validator: validate(data, schema_name)\n\n"
        'Respond with ONLY a JSON object: {"tool_name": "...", "method": "...", "parameters": {...}}\n'
        "Do not include any other text. Look at the workspace to see results from your previous tool calls."
    )

    import json

    for task_id in ["easy", "medium", "hard"]:
        try:
            obs = env.reset(task_id=task_id)
            while not obs.done:
                user_msg = (
                    f"Task: {obs.task_description}\n"
                    f"Step: {obs.step_number}/{obs.max_steps}\n"
                    f"Last response: {json.dumps(obs.tool_response)}\n"
                    f"Workspace: {json.dumps(obs.workspace)}"
                )

                try:
                    response = client.chat.completions.create(
                        model=model_name or "gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.0,
                        max_tokens=500,
                    )

                    content = response.choices[0].message.content.strip()
                    # Strip markdown code blocks if present
                    if content.startswith("```"):
                        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
                        if content.endswith("```"):
                            content = content[:-3]
                        content = content.strip()

                    action_data = json.loads(content)
                    action = ToolOrchestrationAction(
                        tool_name=action_data.get("tool_name", "validator"),
                        method=action_data.get("method", "validate"),
                        parameters=action_data.get("parameters", {}),
                    )
                except (json.JSONDecodeError, Exception):
                    # Fallback: harmless no-op
                    action = ToolOrchestrationAction(
                        tool_name="validator",
                        method="validate",
                        parameters={"data": {}, "schema_name": "email_format"},
                    )

                obs = env.step(action)

            grader_result = env.get_last_grader_result()
            scores[task_id] = grader_result["score"] if grader_result else 0.0
        except Exception as e:
            scores[task_id] = 0.0

    return {**scores, "status": "completed"}


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
