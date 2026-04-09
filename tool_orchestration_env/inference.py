#!/usr/bin/env python3
"""
Baseline inference script for the Tool Orchestration Environment.

Uses an OpenAI-compatible LLM to solve all 3 tasks (easy, medium, hard).
Reads configuration from environment variables:
    - API_BASE_URL: LLM API base URL (e.g., https://api.openai.com/v1)
    - MODEL_NAME: Model to use (e.g., gpt-4o-mini)
    - HF_TOKEN: API key / Hugging Face token (also accepts OPENAI_API_KEY)
    - ENV_URL: Environment server URL (default: http://localhost:7860)

STDOUT FORMAT (mandatory for hackathon evaluation):
    [START] task=<task_name> env=tool_orchestration_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
from typing import List, Optional

from openai import OpenAI

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tool_orchestration_env.models import ToolOrchestrationAction
from tool_orchestration_env.client import ToolOrchestrationEnv


BENCHMARK = "tool_orchestration_env"
SUCCESS_THRESHOLD = 0.1

SYSTEM_PROMPT = """You are an AI agent interacting with a tool orchestration environment. You must complete the given task by making tool calls.

Available tools and their methods:

1. **database**
   - query(sql): Execute a SELECT query. Returns {rows, row_count}
   - insert(table, data): Insert a row. Returns {inserted, id}
   Tables: employees (id, name, email, department, hire_date, salary), invoices (id, vendor, amount, category, date, status), projects (id, name, team_lead_email, department, status, deadline)

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

# Task-specific guidance to help the LLM make better tool calls
TASK_HINTS = {
    "easy": (
        "STRATEGY: 1) Query employees with hire_date > '2026-03-01' to find new hires. "
        "2) For EACH new hire, send a personalized welcome email to their email address. "
        "Subject must include their name (e.g., 'Welcome to the team, Carol Johnson!'). "
        "Body must mention their department. Send exactly one email per new hire."
    ),
    "medium": (
        "STRATEGY: 1) Query all invoices from March 2026 (WHERE date LIKE '2026-03%'). "
        "2) Use calculator.group_sum with the invoice data to compute totals per category. "
        "3) Write a markdown report to 'reports/march-2026-expenses.md' with a table of "
        "category totals and a grand total row. Use markdown table format with | and ---. "
        "4) Email the report to finance@acme.com with subject 'March 2026 Expense Report' "
        "and set attachment to the file path 'reports/march-2026-expenses.md'."
    ),
    "hard": (
        "STRATEGY: 1) Query projects table for 'Project Alpha' to find team_lead_email and department. "
        "2) Query employees in that department to get team member user IDs. "
        "3) Use calendar.find_free_slots for all team members (user_01 through user_05 as needed). "
        "If a user returns CalendarServiceTimeout, note them as 'unconfirmed' and proceed with others. "
        "4) Create a calendar event at the first available common slot. "
        "5) Read 'projects/q1-review.md' for context, then write a meeting agenda to "
        "'meetings/q2-review-agenda.md' referencing Q1 accomplishments and Q2 priorities. "
        "6) Email the agenda to attendees mentioning the meeting time and any unconfirmed members."
    ),
}


# ---------------------------------------------------------------------------
# Structured logging helpers (mandatory hackathon format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    client_sync,
    llm_client: OpenAI,
    model_name: str,
    task_id: str,
) -> float:
    """Run a single episode and return the final reward."""
    result = client_sync.reset(task_id=task_id)
    obs = result.observation

    log_start(task=task_id, env=BENCHMARK, model=model_name)

    steps_taken = 0
    rewards: List[float] = []
    prev_reward = 0.01
    score = 0.01
    success = False

    try:
        # Get task-specific hints
        task_hint = TASK_HINTS.get(task_id, "")

        while not result.done:
            # Build user message with full context
            user_msg = (
                f"**Task:** {obs.task_description}\n\n"
            )

            if task_hint and obs.step_number == 0:
                user_msg += f"**Guidance:** {task_hint}\n\n"

            user_msg += (
                f"**Step:** {obs.step_number}/{obs.max_steps}\n\n"
                f"**Last tool response:**\n```json\n{json.dumps(obs.tool_response, indent=2, default=str)}\n```\n\n"
            )

            if obs.workspace:
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
            steps_taken += 1

            # Compute per-step reward delta
            current_reward = result.reward or 0.01
            step_reward = current_reward - prev_reward
            rewards.append(step_reward)
            prev_reward = current_reward

            # Extract error from tool response (if any)
            error = obs.tool_response.get("error") if isinstance(obs.tool_response, dict) else None

            log_step(
                step=steps_taken,
                action=f"{action.tool_name}.{action.method}",
                reward=step_reward,
                done=result.done,
                error=error,
            )

        score = result.reward or 0.01
        success = score >= SUCCESS_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

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
                print(f"[DEBUG] LLM parse error (attempt 1): {e}", flush=True)
                continue
            else:
                print(f"[DEBUG] LLM parse error (attempt 2), using fallback", flush=True)
                return {
                    "tool_name": "validator",
                    "method": "validate",
                    "parameters": {"data": {}, "schema_name": "email_format"},
                }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Read config from environment (defaults required by hackathon spec)
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.environ.get("HF_TOKEN")
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")

    if hf_token is None:
        raise ValueError("HF_TOKEN environment variable is required")

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
                print(f"[DEBUG] {task_id} error: {e}", flush=True)
                scores[task_id] = 0.01


if __name__ == "__main__":
    main()
