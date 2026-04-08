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

import json
import os

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


_NL_SYSTEM_PROMPT = (
    "You are a tool-call translator. The user describes an action in natural language. "
    "You MUST respond with ONLY a JSON object — no other text, no markdown.\n\n"
    "Available tools and methods:\n"
    "- database: query(sql), insert(table, data)\n"
    "- email: send(to, subject, body), search(query), list_inbox()\n"
    "- filestore: read(path), write(path, content), list(directory)\n"
    "- calculator: compute(expression), group_sum(data, group_by, aggregate), date_diff(date1, date2)\n"
    "- calendar: get_events(user, date_range), find_free_slots(users, date_range, duration_minutes), create_event(title, attendees, start, end)\n"
    "- validator: validate(data, schema_name)\n\n"
    'Respond with ONLY: {"tool_name": "...", "method": "...", "parameters": {...}}\n'
    "No explanation. No markdown. Just the JSON object."
)

_DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _interpret_nl(text: str, task_description: str = "") -> dict:
    """Use an LLM to convert natural language into a structured tool call."""
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("HF_TOKEN not configured. Set it in your Space secrets.")

    api_base = os.environ.get("API_BASE_URL", "")
    model = os.environ.get("MODEL_NAME", "") or _DEFAULT_HF_MODEL

    if not api_base:
        api_base = f"https://router.huggingface.co/hf-inference/models/{model}/v1"

    from openai import OpenAI
    client = OpenAI(base_url=api_base, api_key=hf_token)

    user_msg = text
    if task_description:
        user_msg = f"Current task: {task_description}\n\nUser request: {text}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _NL_SYSTEM_PROMPT},
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

    return json.loads(content)


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


@app.post("/interact")
async def interact(payload: dict):
    """Interactive endpoint for the web dashboard. Maintains state between calls."""
    env = _get_env()
    action_type = payload.get("action", "")

    if action_type == "reset":
        task_id = payload.get("task_id", "easy")
        obs = env.reset(task_id=task_id)
        return {
            "observation": {
                "tool_response": obs.tool_response,
                "task_description": obs.task_description,
                "available_tools": obs.available_tools,
                "workspace": obs.workspace,
                "step_number": obs.step_number,
                "max_steps": obs.max_steps,
            },
            "reward": 0.0,
            "done": False,
        }

    elif action_type == "step":
        tool_name = payload.get("tool_name", "")
        method = payload.get("method", "")
        parameters = payload.get("parameters", {})
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters) if parameters.strip() else {}
            except json.JSONDecodeError:
                return {"error": "Invalid JSON in parameters"}

        act = ToolOrchestrationAction(tool_name=tool_name, method=method, parameters=parameters)
        obs = env.step(act)
        result = {
            "observation": {
                "tool_response": obs.tool_response,
                "task_description": obs.task_description,
                "workspace": obs.workspace,
                "step_number": obs.step_number,
                "max_steps": obs.max_steps,
            },
            "reward": obs.reward,
            "done": obs.done,
        }
        if obs.done:
            grader = env.get_last_grader_result()
            if grader:
                result["grader"] = grader
        return result

    elif action_type == "nl":
        text = payload.get("text", "").strip()
        if not text:
            return {"error": "No text provided."}
        try:
            task_desc = ""
            if hasattr(env, '_current_task') and env._current_task:
                task_desc = env._current_task.get("description", "")
            parsed = _interpret_nl(text, task_desc)
            tool_name = parsed.get("tool_name", "")
            method = parsed.get("method", "")
            parameters = parsed.get("parameters", {})
        except ValueError as e:
            return {"error": str(e)}
        except (json.JSONDecodeError, Exception) as e:
            return {"error": f"Could not interpret input: {e}"}

        act = ToolOrchestrationAction(tool_name=tool_name, method=method, parameters=parameters)
        obs = env.step(act)
        result = {
            "interpreted_as": {"tool_name": tool_name, "method": method, "parameters": parameters},
            "observation": {
                "tool_response": obs.tool_response,
                "task_description": obs.task_description,
                "workspace": obs.workspace,
                "step_number": obs.step_number,
                "max_steps": obs.max_steps,
            },
            "reward": obs.reward,
            "done": obs.done,
        }
        if obs.done:
            grader = env.get_last_grader_result()
            if grader:
                result["grader"] = grader
        return result

    elif action_type == "state":
        state = env.state
        return {
            "episode_id": state.episode_id,
            "task_id": state.task_id,
            "step_count": state.step_count,
            "total_reward": state.total_reward,
            "tools_called": state.tools_called,
            "done": state.done,
        }

    return {"error": f"Unknown action '{action_type}'. Use 'reset', 'step', or 'state'."}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the environment dashboard."""
    return DASHBOARD_HTML


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tool Orchestration Environment</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.6}
.container{max-width:1100px;margin:0 auto;padding:20px}
h1{font-size:26px;color:#f8fafc}
.about{color:#94a3b8;font-size:14px;margin:8px 0 20px;max-width:800px}
.about strong{color:#cbd5e1}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
.card{background:#1e293b;border-radius:10px;padding:16px;border:1px solid #334155}
.card h2{font-size:15px;color:#38bdf8;margin-bottom:10px}
label{display:block;font-size:12px;color:#94a3b8;margin:8px 0 3px;text-transform:uppercase;letter-spacing:.5px}
select,input,textarea{width:100%;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:8px 10px;font-size:13px;font-family:inherit}
textarea{font-family:'SF Mono',Consolas,monospace;min-height:60px;resize:vertical}
select{cursor:pointer}
.btn{display:inline-block;padding:9px 20px;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer;border:none;transition:background .15s}
.btn-blue{background:#2563eb;color:#fff}.btn-blue:hover{background:#1d4ed8}
.btn-green{background:#059669;color:#fff}.btn-green:hover{background:#047857}
.btn-gray{background:#334155;color:#cbd5e1}.btn-gray:hover{background:#475569}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-row{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}
.status-bar{display:flex;gap:16px;padding:10px 14px;background:#0f172a;border-radius:8px;font-size:13px;margin-bottom:12px;flex-wrap:wrap;align-items:center}
.status-bar .item{display:flex;align-items:center;gap:5px}
.status-bar .val{color:#f8fafc;font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.dot-g{background:#22c55e}.dot-r{background:#ef4444}.dot-y{background:#eab308}.dot-x{background:#475569}
pre{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:12px;font-size:12px;overflow:auto;max-height:300px;white-space:pre-wrap;word-break:break-word;color:#cbd5e1;font-family:'SF Mono',Consolas,monospace}
.history{max-height:260px;overflow-y:auto}
.h-entry{padding:6px 0;border-bottom:1px solid #1e293b;font-size:12px;font-family:monospace}
.h-step{color:#38bdf8;font-weight:700;margin-right:4px}
.h-action{color:#a78bfa}.h-reward{color:#22c55e}
.grader-row{display:flex;justify-content:space-between;padding:4px 0;font-size:13px;border-bottom:1px solid #334155}
.grader-row:last-child{border:none}
.grader-bar{flex:1;height:6px;background:#334155;border-radius:3px;margin:0 10px;overflow:hidden;align-self:center}
.grader-fill{height:100%;background:#38bdf8;border-radius:3px;transition:width .3s}
.tools-ref{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.chip{background:#334155;padding:2px 10px;border-radius:6px;font-size:12px;color:#94a3b8;cursor:pointer;transition:background .15s}
.chip:hover{background:#475569;color:#e2e8f0}
.score-big{font-size:40px;font-weight:700;text-align:center;padding:10px 0}
.score-big.s-green{color:#22c55e}.score-big.s-blue{color:#38bdf8}.score-big.s-yellow{color:#eab308}.score-big.s-gray{color:#475569}
</style>
</head>
<body>
<div class="container">
<h1>Tool Orchestration Environment</h1>
<p class="about">
An RL environment where AI agents chain <strong>6 simulated business tools</strong> (database, email, filestore, calculator, calendar, validator) to complete multi-step workflows.
Agents receive a task description, choose which tool to call at each step, and earn rewards based on both <strong>correct tool sequencing</strong> (30%) and <strong>output quality</strong> judged by a deterministic grader (70%).
Three tasks test increasing difficulty: simple data lookup, multi-step report generation, and a complex scheduling task with deliberate error injection to test resilience.
</p>

<div class="status-bar">
  <div class="item"><span class="dot dot-x" id="health-dot"></span> <span id="health-txt">Checking...</span></div>
  <div class="item">Step: <span class="val" id="st-step">-</span></div>
  <div class="item">Reward: <span class="val" id="st-reward">-</span></div>
  <div class="item">Task: <span class="val" id="st-task">-</span></div>
  <div class="item" id="st-done-wrap" style="display:none"><span class="dot dot-g"></span> <span class="val">Episode Complete</span></div>
</div>

<div class="grid">
  <!-- Left: Controls -->
  <div>
    <div class="card" style="margin-bottom:16px">
      <h2>1. Start Episode</h2>
      <label>Task</label>
      <select id="task-select">
        <option value="easy">Easy &mdash; Welcome Emails (10 steps)</option>
        <option value="medium">Medium &mdash; Expense Report (15 steps)</option>
        <option value="hard">Hard &mdash; Schedule Meeting (20 steps)</option>
      </select>
      <div class="btn-row">
        <button class="btn btn-green" onclick="doReset()">Reset Environment</button>
      </div>
    </div>

    <div class="card" style="margin-bottom:16px">
      <h2>2. Tell the Agent What to Do</h2>
      <label>Your instruction</label>
      <input type="text" id="nl-input" placeholder="e.g. look up all employees in the database" autocomplete="off">
      <div style="font-size:11px;color:#64748b;margin-top:4px">Type in plain English. An LLM will interpret your instruction into a tool call.</div>
      <div class="btn-row">
        <button class="btn btn-blue" onclick="doStep()" id="step-btn">Submit</button>
        <span id="loading-indicator" style="display:none;color:#94a3b8;font-size:12px">Interpreting...</span>
      </div>
    </div>

    <div class="card">
      <h2>Tool Reference</h2>
      <div style="font-size:12px;color:#94a3b8;margin-bottom:6px">Available tools and methods you can ask about</div>
      <div id="tool-ref"></div>
    </div>
  </div>

  <!-- Right: Output -->
  <div>
    <div class="card" style="margin-bottom:16px">
      <h2>Task Description</h2>
      <div id="task-desc" style="font-size:13px;color:#cbd5e1;min-height:40px">Reset an environment to see the task.</div>
    </div>

    <div class="card" id="interpreted-card" style="margin-bottom:16px;display:none">
      <h2>Interpreted As</h2>
      <pre id="interpreted-output" style="color:#a78bfa;font-size:13px"></pre>
    </div>

    <div class="card" style="margin-bottom:16px">
      <h2>Tool Response</h2>
      <pre id="response-output">No response yet. Reset an environment and execute a step.</pre>
    </div>

    <div class="card" style="margin-bottom:16px">
      <h2>Episode History</h2>
      <div class="history" id="history-list"><div style="color:#475569;font-size:13px">No steps yet.</div></div>
    </div>

    <div class="card" id="grader-card" style="display:none">
      <h2>Grader Results</h2>
      <div class="score-big s-gray" id="grader-score">-</div>
      <div id="grader-breakdown"></div>
    </div>
  </div>
</div>
</div>

<script>
const TOOL_INFO = {
  database: ['query(sql)', 'insert(table, data)'],
  email: ['send(to, subject, body)', 'search(query)', 'list_inbox()'],
  filestore: ['read(path)', 'write(path, content)', 'list(directory)'],
  calculator: ['compute(expression)', 'group_sum(data, group_by, aggregate)', 'date_diff(date1, date2)'],
  calendar: ['get_events(user, date_range)', 'find_free_slots(users, date_range, duration_minutes)', 'create_event(title, attendees, start, end)'],
  validator: ['validate(data, schema_name)']
};
let history = [];

// Build tool reference (read-only, no click actions)
(function(){
  let html = '';
  for (const [t, methods] of Object.entries(TOOL_INFO)) {
    html += `<div style="margin-bottom:6px"><strong style="color:#cbd5e1;font-size:13px">${t}</strong><div class="tools-ref">`;
    for (const m of methods) html += `<span class="chip">${m}</span>`;
    html += '</div></div>';
  }
  document.getElementById('tool-ref').innerHTML = html;
})();

// Enter key submits
document.getElementById('nl-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') { e.preventDefault(); doStep(); }
});

async function api(body) {
  const r = await fetch('/interact', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  return r.json();
}

async function doReset() {
  const task = document.getElementById('task-select').value;
  history = [];
  document.getElementById('history-list').innerHTML = '<div style="color:#475569;font-size:13px">No steps yet.</div>';
  document.getElementById('grader-card').style.display = 'none';
  document.getElementById('st-done-wrap').style.display = 'none';
  document.getElementById('interpreted-card').style.display = 'none';
  const d = await api({action: 'reset', task_id: task});
  document.getElementById('task-desc').textContent = d.observation.task_description;
  document.getElementById('response-output').textContent = JSON.stringify(d.observation.tool_response, null, 2);
  document.getElementById('st-step').textContent = `0/${d.observation.max_steps}`;
  document.getElementById('st-reward').textContent = '0.0000';
  document.getElementById('st-task').textContent = task;
  document.getElementById('step-btn').disabled = false;
  document.getElementById('nl-input').value = '';
}

async function doStep() {
  const text = document.getElementById('nl-input').value.trim();
  if (!text) { alert('Type an instruction first'); return; }

  document.getElementById('step-btn').disabled = true;
  document.getElementById('loading-indicator').style.display = 'inline';

  let d;
  try {
    d = await api({action: 'nl', text: text});
  } finally {
    document.getElementById('loading-indicator').style.display = 'none';
  }

  if (d.error) {
    document.getElementById('response-output').textContent = 'Error: ' + d.error;
    document.getElementById('step-btn').disabled = false;
    return;
  }

  // Show interpreted tool call
  if (d.interpreted_as) {
    const ia = d.interpreted_as;
    document.getElementById('interpreted-card').style.display = 'block';
    document.getElementById('interpreted-output').textContent = `${ia.tool_name}.${ia.method}(${JSON.stringify(ia.parameters)})`;
  }

  const obs = d.observation;
  document.getElementById('response-output').textContent = JSON.stringify(obs.tool_response, null, 2);
  document.getElementById('st-step').textContent = `${obs.step_number}/${obs.max_steps}`;
  document.getElementById('st-reward').textContent = (d.reward || 0).toFixed(4);

  const ia = d.interpreted_as || {};
  history.push({tool: ia.tool_name||'?', method: ia.method||'?', input: text, reward: d.reward, done: d.done});
  let hhtml = '';
  history.forEach((h, i) => {
    hhtml += `<div class="h-entry"><span class="h-step">[${i+1}]</span> <span class="h-action">${h.tool}.${h.method}</span> <span style="color:#64748b">"${h.input}"</span> <span class="h-reward">reward=${(h.reward||0).toFixed(4)}</span>${h.done ? ' <span style="color:#22c55e">DONE</span>' : ''}</div>`;
  });
  document.getElementById('history-list').innerHTML = hhtml;

  document.getElementById('nl-input').value = '';
  document.getElementById('step-btn').disabled = d.done;

  if (d.done) {
    document.getElementById('st-done-wrap').style.display = 'flex';
    if (d.grader) {
      const gc = document.getElementById('grader-card');
      gc.style.display = 'block';
      const sc = d.grader.score;
      const cls = sc >= 0.9 ? 's-green' : sc >= 0.5 ? 's-blue' : sc > 0 ? 's-yellow' : 's-gray';
      document.getElementById('grader-score').className = 'score-big ' + cls;
      document.getElementById('grader-score').textContent = sc.toFixed(4);
      let bhtml = '';
      for (const [k, v] of Object.entries(d.grader.breakdown || {})) {
        const pct = (v * 100).toFixed(0);
        bhtml += `<div class="grader-row"><span style="min-width:140px">${k}</span><div class="grader-bar"><div class="grader-fill" style="width:${pct}%"></div></div><span style="min-width:40px;text-align:right;color:#f8fafc">${v.toFixed(2)}</span></div>`;
      }
      document.getElementById('grader-breakdown').innerHTML = bhtml;
    }
  }
}

// Health check on load
(async()=>{
  try {
    const r = await fetch('/health');
    if (r.ok) { document.getElementById('health-dot').className='dot dot-g'; document.getElementById('health-txt').textContent='Healthy'; }
  } catch(e) { document.getElementById('health-dot').className='dot dot-r'; document.getElementById('health-txt').textContent='Unreachable'; }
})();
</script>
</body>
</html>"""


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
