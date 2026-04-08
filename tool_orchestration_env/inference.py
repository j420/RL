#!/usr/bin/env python3
"""
Baseline inference script for the Tool Orchestration Environment.

Uses an OpenAI-compatible LLM to solve all 3 tasks (easy, medium, hard).
Reads configuration from environment variables:
    - API_BASE_URL: LLM API base URL (e.g., https://api.openai.com/v1)
    - MODEL_NAME: Model to use (e.g., gpt-4o-mini)
    - HF_TOKEN: API key / Hugging Face token (also accepts OPENAI_API_KEY)
    - ENV_URL: Environment server URL (default: http://localhost:7860)

Emits structured stdout logs:
    [START] — at the beginning of each task episode
    [STEP]  — after each agent action
    [END]   — at the end of each task episode
"""

import json
import os
import sys
import time

from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tool_orchestration_env.models import ToolOrchestrationAction
from tool_orchestration_env.client import ToolOrchestrationEnv


SYSTEM_PROMPT = """You are an AI agent interacting with a tool orchestration environment. You must complete the given task by making tool calls.

Available tools and their methods:

1. **database**
   - query(sql): Execute a SELECT query. Returns {rows, row_count}
   - insert(table, data): Insert a row. Returns {inserted, id}

2. **email**
   - send(to, subject, body, attachment): Send an email. Returns {status, message_id}
   - search(query): Search emails. Returns {results, count}
   - list_inbox(): List inbox emails. Returns {emails}

3. **filestore**
   - read(path): Read a file. Returns {content, size_bytes}
   - write(path, content): Write a file. Returns {status, path, size_bytes}
   - list(directory): List files. Returns {files}

4. **calculator**
   - compute(expression): Evaluate math expression. Returns {result}
   - group_sum(data, group_by, aggregate): Group and sum data. Returns {result}
   - date_diff(date1, date2): Date difference in days. Returns {days}

5. **calendar**
   - get_events(user, date_range): Get user's events. Returns {events}
   - find_free_slots(users, date_range, duration_minutes): Find common free slots. Returns {slots}
   - create_event(title, attendees, start, end): Create event. Returns {event_id, status}

6. **validator**
   - validate(data, schema_name): Validate data against schema. Returns {valid, errors}

IMPORTANT RULES:
- Respond with ONLY a valid JSON object: {"tool_name": "...", "method": "...", "parameters": {...}}
- Do NOT include any text before or after the JSON.
- Look at the workspace to see results from your previous tool calls — use that data in subsequent calls.
- If a tool returns an error, adapt your approach rather than retrying the same call.
- Plan your tool calls efficiently to complete the task in as few steps as possible."""


def run_episode(
    client_sync,
    llm_client: OpenAI,
    model_name: str,
    task_id: str,
) -> float:
    """Run a single episode and return the final reward."""
    result = client_sync.reset(task_id=task_id)
    obs = result.observation

    # [START] structured log
    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "max_steps": obs.max_steps,
        "task_description": obs.task_description[:120],
    }))

    step_num = 0
    while not result.done:
        # Build user message with full context
        user_msg = (
            f"**Task:** {obs.task_description}\n\n"
            f"**Step:** {obs.step_number}/{obs.max_steps}\n\n"
            f"**Last tool response:**\n```json\n{json.dumps(obs.tool_response, indent=2, default=str)}\n```\n\n"
        )

        if obs.workspace:
            # Show recent workspace entries (last 3 to avoid context overflow)
            recent_keys = sorted(obs.workspace.keys())[-3:]
            recent_workspace = {k: obs.workspace[k] for k in recent_keys}
            user_msg += f"**Recent workspace (previous results):**\n```json\n{json.dumps(recent_workspace, indent=2, default=str)}\n```\n"

        # Call LLM
        action_data = call_llm(llm_client, model_name, user_msg)

        # Create action
        action = ToolOrchestrationAction(
            tool_name=action_data.get("tool_name", "validator"),
            method=action_data.get("method", "validate"),
            parameters=action_data.get("parameters", {}),
        )

        # Execute step
        result = client_sync.step(action)
        obs = result.observation
        step_num += 1

        # [STEP] structured log
        print(json.dumps({
            "type": "[STEP]",
            "task_id": task_id,
            "step": step_num,
            "action": f"{action.tool_name}.{action.method}",
            "reward": result.reward,
            "done": result.done,
        }))

    # [END] structured log
    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "reward": result.reward,
        "steps": step_num,
    }))

    return result.reward


def call_llm(client: OpenAI, model: str, user_msg: str) -> dict:
    """Call the LLM and parse the JSON response. Retry once on failure."""
    for attempt in range(2):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            if attempt == 1:
                messages.append({
                    "role": "user",
                    "content": "Your previous response was not valid JSON. Respond with ONLY a JSON object: {\"tool_name\": \"...\", \"method\": \"...\", \"parameters\": {...}}",
                })

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1000,
            )

            content = response.choices[0].message.content.strip()

            # Strip markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:])
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            return json.loads(content)

        except (json.JSONDecodeError, Exception) as e:
            if attempt == 0:
                print(f"    LLM parse error (attempt 1): {e}")
                continue
            else:
                print(f"    LLM parse error (attempt 2), using fallback")
                return {
                    "tool_name": "validator",
                    "method": "validate",
                    "parameters": {"data": {}, "schema_name": "email_format"},
                }


def main():
    # Read config from environment
    api_base = os.environ.get("API_BASE_URL", "")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")

    if not api_base or not hf_token:
        print("Error: API_BASE_URL and HF_TOKEN environment variables are required.")
        print("Usage:")
        print("  export API_BASE_URL=https://api.openai.com/v1")
        print("  export MODEL_NAME=gpt-4o-mini")
        print("  export HF_TOKEN=your_api_key")
        print("  python inference.py")
        sys.exit(1)

    print(json.dumps({
        "type": "[CONFIG]",
        "API_BASE_URL": api_base,
        "MODEL_NAME": model_name,
        "ENV_URL": env_url,
    }))

    # Initialize LLM client (OpenAI-compatible)
    llm_client = OpenAI(base_url=api_base, api_key=hf_token)

    # Connect to environment via WebSocket
    env_client = ToolOrchestrationEnv(base_url=env_url)
    sync_client = env_client.sync()

    scores = {}

    with sync_client:
        for task_id in ["easy", "medium", "hard"]:
            try:
                score = run_episode(sync_client, llm_client, model_name, task_id)
                scores[task_id] = score
            except Exception as e:
                print(f"  [{task_id.upper()}] Error: {e}")
                scores[task_id] = 0.0

    # Print final results
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(json.dumps({
        "type": "[RESULTS]",
        "scores": scores,
        "average": round(avg, 4),
    }))


if __name__ == "__main__":
    main()
