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
import time

from fastapi.responses import HTMLResponse

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


@app.post("/run-tests")
async def run_tests():
    """Run scripted integration tests on all 3 tasks + edge cases. No LLM needed."""
    env = _get_env()
    results = {"tests": [], "summary": {}}
    passed = 0
    failed = 0
    start_time = time.time()

    def check(name, condition, detail=""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        results["tests"].append({"name": name, "status": status, "detail": detail})

    # --- Easy task (perfect episode) ---
    env.reset(task_id="easy")
    obs = env.step(ToolOrchestrationAction(
        tool_name="database", method="query",
        parameters={"sql": "SELECT * FROM employees WHERE hire_date > '2026-03-01'"}
    ))
    check("easy: query finds 4 new hires", obs.tool_response.get("row_count") == 4)

    for name, dept, email in [
        ("Carol Johnson", "Engineering", "carol.johnson@acme.com"),
        ("David Kim", "Engineering", "david.kim@acme.com"),
        ("Hannah Davis", "Marketing", "hannah.davis@acme.com"),
        ("Lisa Tanaka", "Finance", "lisa.tanaka@acme.com"),
    ]:
        obs = env.step(ToolOrchestrationAction(
            tool_name="email", method="send",
            parameters={"to": email, "subject": f"Welcome to the team, {name}!", "body": f"Welcome to the {dept} department."}
        ))
    easy_reward = env._normalize_reward()
    easy_grader = env._grader_score
    check("easy: done=True", env._done)
    check("easy: score=1.0", easy_reward == 1.0, f"got {easy_reward}")
    check("easy: grader=1.0", easy_grader == 1.0, f"got {easy_grader}")

    # --- Medium task (perfect episode) ---
    env.reset(task_id="medium")
    obs = env.step(ToolOrchestrationAction(
        tool_name="database", method="query",
        parameters={"sql": "SELECT * FROM invoices WHERE date LIKE '2026-03%'"}
    ))
    rows = obs.tool_response.get("rows", [])
    check("medium: query finds 47 invoices", obs.tool_response.get("row_count") == 47)

    obs = env.step(ToolOrchestrationAction(
        tool_name="calculator", method="group_sum",
        parameters={"data": rows, "group_by": "category", "aggregate": "amount"}
    ))
    totals = obs.tool_response.get("result", {})
    check("medium: Software=$8940", totals.get("Software") == 8940.0)
    check("medium: Travel=$3200", totals.get("Travel") == 3200.0)

    report = "# March 2026 Expense Report\n\n| Category | Total |\n|---|---|\n| Software | 8940 |\n| Travel | 3200 |\n| Office Supplies | 1450 |\n| Marketing | 5600 |\n| **Grand Total** | **19190** |"
    env.step(ToolOrchestrationAction(tool_name="filestore", method="write", parameters={"path": "reports/march-2026-expenses.md", "content": report}))
    obs = env.step(ToolOrchestrationAction(
        tool_name="email", method="send",
        parameters={"to": "finance@acme.com", "subject": "March 2026 Expense Report", "body": "See attached.", "attachment": "reports/march-2026-expenses.md"}
    ))
    medium_reward = env._normalize_reward()
    check("medium: done=True", env._done)
    check("medium: score=1.0", medium_reward == 1.0, f"got {medium_reward}")

    # --- Hard task (perfect episode) ---
    env.reset(task_id="hard")
    env.step(ToolOrchestrationAction(tool_name="database", method="query", parameters={"sql": "SELECT * FROM projects WHERE name = 'Project Alpha'"}))
    env.step(ToolOrchestrationAction(tool_name="database", method="query", parameters={"sql": "SELECT * FROM employees WHERE department = 'Engineering'"}))
    obs = env.step(ToolOrchestrationAction(
        tool_name="calendar", method="find_free_slots",
        parameters={"users": ["user_01", "user_02", "user_03", "user_04"], "date_range": {"start": "2026-04-01", "end": "2026-04-07"}, "duration_minutes": 60}
    ))
    check("hard: user_03 unavailable", "user_03" in obs.tool_response.get("unavailable_users", []))
    check("hard: free slots found", len(obs.tool_response.get("slots", [])) > 0)

    env.step(ToolOrchestrationAction(tool_name="calendar", method="create_event", parameters={"title": "Q2 Review", "attendees": ["user_01", "user_02", "user_04"], "start": "2026-04-03T10:00", "end": "2026-04-03T11:00"}))
    env.step(ToolOrchestrationAction(tool_name="filestore", method="read", parameters={"path": "projects/q1-review.md"}))
    env.step(ToolOrchestrationAction(tool_name="filestore", method="write", parameters={"path": "meetings/q2-review-agenda.md", "content": "Q2 Review Agenda based on q1 accomplishments and v2.0"}))
    obs = env.step(ToolOrchestrationAction(tool_name="email", method="send", parameters={"to": "engineering@acme.com", "subject": "Q2 Review Meeting", "body": "Q2 review meeting scheduled."}))
    hard_reward = env._normalize_reward()
    hard_grader = env._grader_score
    check("hard: done=True", env._done)
    check("hard: score=1.0", hard_reward == 1.0, f"got {hard_reward}")
    check("hard: error_handling graded", hard_grader == 1.0, f"got {hard_grader}")

    # --- Edge cases ---
    env.reset(task_id="easy")
    obs = env.step(ToolOrchestrationAction(tool_name="FAKE_TOOL", method="x", parameters={}))
    check("edge: invalid tool returns error", "error" in obs.tool_response)

    env.reset(task_id="easy")
    obs = env.step(ToolOrchestrationAction(tool_name="calculator", method="compute", parameters={"expression": "().__class__.__bases__[0]"}))
    check("edge: calculator blocks dunder attack", "error" in obs.tool_response)

    obs = env.step(ToolOrchestrationAction(tool_name="calculator", method="compute", parameters={"expression": "sqrt(144) + abs(-5)"}))
    check("edge: calculator allows safe math", obs.tool_response.get("result") == 17.0)

    obs = env.step(ToolOrchestrationAction(tool_name="database", method="query", parameters={"sql": "DROP TABLE employees"}))
    check("edge: SQL injection blocked", "error" in obs.tool_response)

    # --- Grader variance ---
    from .grader import Grader
    g = Grader()
    empty = g.grade("easy", [])
    check("grader: empty history score=0", empty["score"] == 0.0)
    check("grader: score != constant", easy_grader != empty["score"])

    # --- Summary ---
    elapsed = round(time.time() - start_time, 2)
    results["summary"] = {
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "elapsed_seconds": elapsed,
        "scores": {"easy": easy_reward, "medium": medium_reward, "hard": hard_reward},
    }
    return results


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the environment dashboard."""
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tool Orchestration Environment</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; line-height: 1.6; }
  .container { max-width: 1100px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 28px; margin-bottom: 4px; color: #f8fafc; }
  .subtitle { color: #94a3b8; margin-bottom: 24px; font-size: 15px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
  .card h2 { font-size: 16px; color: #38bdf8; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
  .card h3 { font-size: 14px; color: #94a3b8; margin: 12px 0 6px; text-transform: uppercase; letter-spacing: 0.5px; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .badge-easy { background: #065f46; color: #6ee7b7; }
  .badge-medium { background: #78350f; color: #fbbf24; }
  .badge-hard { background: #7f1d1d; color: #fca5a5; }
  .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #334155; font-size: 14px; }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { color: #94a3b8; }
  .stat-value { color: #f8fafc; font-weight: 600; }
  .tools-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .tool-chip { background: #334155; padding: 4px 12px; border-radius: 8px; font-size: 13px; color: #cbd5e1; }
  .criteria-bar { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 13px; }
  .criteria-bar .bar-bg { flex: 1; height: 8px; background: #334155; border-radius: 4px; overflow: hidden; }
  .criteria-bar .bar-fill { height: 100%; background: #38bdf8; border-radius: 4px; }
  .criteria-bar .weight { color: #94a3b8; min-width: 36px; text-align: right; }
  button { background: #2563eb; color: white; border: none; padding: 12px 28px; border-radius: 8px; font-size: 15px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #475569; cursor: not-allowed; }
  .btn-row { display: flex; gap: 12px; align-items: center; margin-bottom: 24px; flex-wrap: wrap; }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .dot-green { background: #22c55e; }
  .dot-red { background: #ef4444; }
  .dot-gray { background: #64748b; }
  #results-panel { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; display: none; }
  #results-panel h2 { font-size: 18px; color: #f8fafc; margin-bottom: 16px; }
  .score-cards { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
  .score-card { text-align: center; padding: 16px 24px; background: #0f172a; border-radius: 10px; min-width: 120px; }
  .score-card .label { font-size: 12px; color: #94a3b8; text-transform: uppercase; }
  .score-card .value { font-size: 32px; font-weight: 700; margin-top: 4px; }
  .score-card .value.perfect { color: #22c55e; }
  .score-card .value.good { color: #38bdf8; }
  .score-card .value.low { color: #fbbf24; }
  .test-list { max-height: 400px; overflow-y: auto; }
  .test-item { display: flex; align-items: center; gap: 8px; padding: 6px 0; border-bottom: 1px solid #1e293b; font-size: 13px; font-family: monospace; }
  .test-pass { color: #22c55e; }
  .test-fail { color: #ef4444; }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid #475569; border-top-color: #38bdf8; border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .endpoints { font-size: 13px; font-family: monospace; }
  .endpoints div { padding: 3px 0; }
  .method { display: inline-block; width: 44px; font-weight: 700; }
  .method-get { color: #22c55e; }
  .method-post { color: #38bdf8; }
  .method-ws { color: #a78bfa; }
</style>
</head>
<body>
<div class="container">
  <h1>Tool Orchestration Environment</h1>
  <p class="subtitle">Multi-tool orchestration RL environment for training AI agents to chain business APIs</p>

  <div class="btn-row">
    <button id="run-btn" onclick="runTests()">Run All Tests</button>
    <button onclick="checkHealth()" style="background:#334155">Check Health</button>
    <span id="health-status"></span>
    <span id="run-status" style="font-size:14px; color:#94a3b8;"></span>
  </div>

  <div id="results-panel">
    <h2>Test Results</h2>
    <div class="score-cards" id="score-cards"></div>
    <div id="summary-line" style="margin-bottom:12px; font-size:14px; color:#94a3b8;"></div>
    <div class="test-list" id="test-list"></div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Easy Task <span class="badge badge-easy">Easy</span></h2>
      <p style="font-size:13px; color:#cbd5e1; margin-bottom:12px;">Send welcome emails to all employees hired after 2026-03-01</p>
      <div class="stat-row"><span class="stat-label">Max Steps</span><span class="stat-value">10</span></div>
      <div class="stat-row"><span class="stat-label">Tools</span><span class="stat-value">database, email</span></div>
      <div class="stat-row"><span class="stat-label">Baseline Score</span><span class="stat-value">~0.82</span></div>
      <h3>Grading Criteria</h3>
      <div class="criteria-bar"><span style="min-width:120px">correct_recipients</span><div class="bar-bg"><div class="bar-fill" style="width:35%"></div></div><span class="weight">35%</span></div>
      <div class="criteria-bar"><span style="min-width:120px">correct_query</span><div class="bar-bg"><div class="bar-fill" style="width:25%"></div></div><span class="weight">25%</span></div>
      <div class="criteria-bar"><span style="min-width:120px">correct_subjects</span><div class="bar-bg"><div class="bar-fill" style="width:20%"></div></div><span class="weight">20%</span></div>
      <div class="criteria-bar"><span style="min-width:120px">correct_body</span><div class="bar-bg"><div class="bar-fill" style="width:20%"></div></div><span class="weight">20%</span></div>
    </div>

    <div class="card">
      <h2>Medium Task <span class="badge badge-medium">Medium</span></h2>
      <p style="font-size:13px; color:#cbd5e1; margin-bottom:12px;">Generate March 2026 expense report with category totals</p>
      <div class="stat-row"><span class="stat-label">Max Steps</span><span class="stat-value">15</span></div>
      <div class="stat-row"><span class="stat-label">Tools</span><span class="stat-value">database, calculator, filestore, email</span></div>
      <div class="stat-row"><span class="stat-label">Baseline Score</span><span class="stat-value">~0.55</span></div>
      <h3>Grading Criteria</h3>
      <div class="criteria-bar"><span style="min-width:120px">correct_report</span><div class="bar-bg"><div class="bar-fill" style="width:25%"></div></div><span class="weight">25%</span></div>
      <div class="criteria-bar"><span style="min-width:120px">correct_calculation</span><div class="bar-bg"><div class="bar-fill" style="width:20%"></div></div><span class="weight">20%</span></div>
      <div class="criteria-bar"><span style="min-width:120px">correct_email</span><div class="bar-bg"><div class="bar-fill" style="width:20%"></div></div><span class="weight">20%</span></div>
      <div class="criteria-bar"><span style="min-width:120px">completeness</span><div class="bar-bg"><div class="bar-fill" style="width:20%"></div></div><span class="weight">20%</span></div>
      <div class="criteria-bar"><span style="min-width:120px">correct_query</span><div class="bar-bg"><div class="bar-fill" style="width:15%"></div></div><span class="weight">15%</span></div>
    </div>

    <div class="card">
      <h2>Hard Task <span class="badge badge-hard">Hard</span></h2>
      <p style="font-size:13px; color:#cbd5e1; margin-bottom:12px;">Schedule Q2 review meeting with error handling for unavailable calendars</p>
      <div class="stat-row"><span class="stat-label">Max Steps</span><span class="stat-value">20</span></div>
      <div class="stat-row"><span class="stat-label">Tools</span><span class="stat-value">database, calendar, filestore, email</span></div>
      <div class="stat-row"><span class="stat-label">Baseline Score</span><span class="stat-value">~0.22</span></div>
      <h3>Grading Criteria</h3>
      <div class="criteria-bar"><span style="min-width:120px">calendar_check</span><div class="bar-bg"><div class="bar-fill" style="width:15%"></div></div><span class="weight">15%</span></div>
      <div class="criteria-bar"><span style="min-width:120px">error_handling</span><div class="bar-bg"><div class="bar-fill" style="width:15%"></div></div><span class="weight">15%</span></div>
      <div class="criteria-bar"><span style="min-width:120px">meeting_created</span><div class="bar-bg"><div class="bar-fill" style="width:15%"></div></div><span class="weight">15%</span></div>
      <div class="criteria-bar"><span style="min-width:120px">email_sent</span><div class="bar-bg"><div class="bar-fill" style="width:15%"></div></div><span class="weight">15%</span></div>
      <div class="criteria-bar"><span style="min-width:120px">project + team</span><div class="bar-bg"><div class="bar-fill" style="width:20%"></div></div><span class="weight">20%</span></div>
      <div class="criteria-bar"><span style="min-width:120px">agenda + edge</span><div class="bar-bg"><div class="bar-fill" style="width:20%"></div></div><span class="weight">20%</span></div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Available Tools</h2>
      <div class="tools-list">
        <span class="tool-chip">database</span>
        <span class="tool-chip">email</span>
        <span class="tool-chip">filestore</span>
        <span class="tool-chip">calculator</span>
        <span class="tool-chip">calendar</span>
        <span class="tool-chip">validator</span>
      </div>
      <h3>Reward Formula</h3>
      <div style="font-size:13px; color:#cbd5e1; margin-top:8px;">
        <code style="color:#38bdf8">final = step_progress * 0.3 + grader_score * 0.7</code>
        <div style="margin-top:6px; color:#94a3b8;">Per-step: +0.10 sequence match, +0.05 success, -0.05 duplicate, -0.10 invalid tool</div>
      </div>
    </div>

    <div class="card">
      <h2>API Endpoints</h2>
      <div class="endpoints">
        <div><span class="method method-get">GET</span> /health</div>
        <div><span class="method method-get">GET</span> /tasks</div>
        <div><span class="method method-get">GET</span> /schema</div>
        <div><span class="method method-get">GET</span> /state</div>
        <div><span class="method method-post">POST</span> /reset</div>
        <div><span class="method method-post">POST</span> /step</div>
        <div><span class="method method-post">POST</span> /grader</div>
        <div><span class="method method-post">POST</span> /run-tests</div>
        <div><span class="method method-ws">WS</span> /ws</div>
      </div>
    </div>
  </div>
</div>

<script>
async function checkHealth() {
  const el = document.getElementById('health-status');
  try {
    const r = await fetch('/health');
    const d = await r.json();
    el.innerHTML = '<span class="status-dot dot-green"></span> Healthy';
  } catch(e) {
    el.innerHTML = '<span class="status-dot dot-red"></span> Unreachable';
  }
}

async function runTests() {
  const btn = document.getElementById('run-btn');
  const status = document.getElementById('run-status');
  const panel = document.getElementById('results-panel');
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Running tests...';
  panel.style.display = 'block';
  document.getElementById('score-cards').innerHTML = '';
  document.getElementById('test-list').innerHTML = '';
  document.getElementById('summary-line').innerHTML = '';

  try {
    const r = await fetch('/run-tests', {method: 'POST'});
    const d = await r.json();
    const s = d.summary;
    const scores = s.scores;

    function scoreClass(v) { return v >= 1.0 ? 'perfect' : v >= 0.5 ? 'good' : 'low'; }

    document.getElementById('score-cards').innerHTML = `
      <div class="score-card"><div class="label">Easy</div><div class="value ${scoreClass(scores.easy)}">${scores.easy.toFixed(2)}</div></div>
      <div class="score-card"><div class="label">Medium</div><div class="value ${scoreClass(scores.medium)}">${scores.medium.toFixed(2)}</div></div>
      <div class="score-card"><div class="label">Hard</div><div class="value ${scoreClass(scores.hard)}">${scores.hard.toFixed(2)}</div></div>
      <div class="score-card"><div class="label">Tests</div><div class="value ${s.failed === 0 ? 'perfect' : 'low'}">${s.passed}/${s.total}</div></div>
    `;

    document.getElementById('summary-line').innerHTML =
      `${s.passed}/${s.total} passed in ${s.elapsed_seconds}s` +
      (s.failed === 0 ? ' &mdash; <span style="color:#22c55e">ALL TESTS PASSED</span>' : ` &mdash; <span style="color:#ef4444">${s.failed} FAILED</span>`);

    document.getElementById('test-list').innerHTML = d.tests.map(t =>
      `<div class="test-item"><span class="${t.status === 'PASS' ? 'test-pass' : 'test-fail'}">${t.status === 'PASS' ? '✓' : '✗'}</span> ${t.name}${t.detail ? ' <span style="color:#64748b">(' + t.detail + ')</span>' : ''}</div>`
    ).join('');

    status.innerHTML = '';
  } catch(e) {
    status.innerHTML = '<span style="color:#ef4444">Error: ' + e.message + '</span>';
  }
  btn.disabled = false;
}

// Auto-check health on load
checkHealth();
</script>
</body>
</html>"""


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
