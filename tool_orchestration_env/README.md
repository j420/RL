---
title: Tool Orchestration Environment
emoji: "\U0001F527"
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Tool Orchestration Environment

A multi-tool orchestration RL environment for training AI agents that must chain together 6 simulated business tools to complete increasingly complex workflows.

## Why This Exists

Multi-tool orchestration — where an AI agent must compose multiple APIs together to complete a workflow — is the #1 capability being trained into production LLMs today. Every major AI lab is working on tool-use training. Yet the OpenEnv catalog has zero environments for multi-tool composition.

Existing OpenEnv environments test individual tools in isolation (a single calendar, a single git command). **Nothing tests the ability to plan and execute multi-step tool pipelines** — the exact capability needed for production AI agents like coding assistants, business automation, and data analysis workflows.

This environment fills that gap by simulating 6 real-world tool APIs that an agent must chain together to complete business workflows. The environment provides **rich per-step reward signals**, making it directly usable for GRPO/TRL training of tool-using agents.

### Novel Mechanics

- **Workspace accumulation**: Each tool call's output is stored in a workspace dict, allowing the agent to reference and compose previous results
- **Error injection**: The hard task injects a `CalendarServiceTimeout` for one user, testing the agent's ability to handle failures gracefully without crashing or infinite-retrying
- **Branching logic**: The hard task requires different behavior depending on calendar availability — creating an event when slots exist, or emailing the team lead when they don't

## Action Space

Each action is a single tool call:

```json
{
  "tool_name": "database | email | filestore | calculator | calendar | validator",
  "method": "string - method name on the selected tool",
  "parameters": { "...method-specific parameters..." }
}
```

### Tool Methods

| Tool | Method | Parameters | Returns |
|------|--------|-----------|---------|
| **database** | `query(sql)` | `{"sql": "SELECT ..."}` | `{rows, row_count}` |
| **database** | `insert(table, data)` | `{"table": "employees", "data": {...}}` | `{inserted, id}` |
| **email** | `send(to, subject, body, attachment)` | `{"to": "a@b.com", "subject": "...", "body": "...", "attachment": "path"}` | `{status, message_id}` |
| **email** | `search(query)` | `{"query": "keyword"}` | `{results, count}` |
| **email** | `list_inbox()` | `{}` | `{emails}` |
| **filestore** | `read(path)` | `{"path": "projects/q1-review.md"}` | `{content, size_bytes}` |
| **filestore** | `write(path, content)` | `{"path": "reports/out.md", "content": "..."}` | `{status, path, size_bytes}` |
| **filestore** | `list(directory)` | `{"directory": "projects"}` | `{files}` |
| **calculator** | `compute(expression)` | `{"expression": "sqrt(144)"}` | `{result}` |
| **calculator** | `group_sum(data, group_by, aggregate)` | `{"data": [...], "group_by": "cat", "aggregate": "amt"}` | `{result}` |
| **calculator** | `date_diff(date1, date2)` | `{"date1": "2026-03-01", "date2": "2026-04-01"}` | `{days}` |
| **calendar** | `get_events(user, date_range)` | `{"user": "user_01", "date_range": {"start": "...", "end": "..."}}` | `{events}` |
| **calendar** | `find_free_slots(users, date_range, duration_minutes)` | `{"users": [...], "date_range": {...}, "duration_minutes": 60}` | `{slots}` |
| **calendar** | `create_event(title, attendees, start, end)` | `{"title": "...", "attendees": [...], "start": "...", "end": "..."}` | `{event_id, status}` |
| **validator** | `validate(data, schema_name)` | `{"data": {...}, "schema_name": "expense_report"}` | `{valid, errors}` |

## Observation Space

```json
{
  "tool_response": { "...result of the last tool call..." },
  "task_description": "The current task text",
  "available_tools": ["database", "email", "filestore", "calculator", "calendar", "validator"],
  "workspace": {
    "step_0": {"tool": "database", "method": "query", "result": {...}},
    "step_1": {"tool": "calculator", "method": "group_sum", "result": {...}}
  },
  "step_number": 2,
  "max_steps": 15,
  "done": false,
  "reward": 0.35
}
```

## Tasks

### Easy — New Employee Welcome Emails
**Difficulty:** Low | **Max Steps:** 10 | **Tools Required:** database, email

Find all employees hired in the last 30 days from the database, then send a personalized welcome email to each new hire with their name and department.

**What makes it easy:** Only 2 tools needed in a simple sequential pipeline. The database query is straightforward and the email construction is templated.

### Medium — Monthly Expense Report
**Difficulty:** Medium | **Max Steps:** 15 | **Tools Required:** database, calculator, filestore, email

Query all March invoices, compute category totals using the calculator, write a formatted markdown report to the filestore, then email it to the finance team.

**What makes it medium:** Requires a 4-tool pipeline with data flowing between steps — invoice data from the DB feeds into the calculator, calculator output formats the report, and the report path attaches to the email. The agent must correctly chain data between tools.

### Hard — Schedule Team Review Meeting
**Difficulty:** Hard | **Max Steps:** 20 | **Tools Required:** database, calendar, filestore, email

Look up a project's team lead, find all team members, check calendar availability for a common meeting slot, handle calendar service errors gracefully, create the event, generate a meeting agenda from existing documents, and email it to all attendees.

**What makes it hard:**
- **6+ tool calls** across 4 different tools
- **Error injection:** user_03's calendar returns `CalendarServiceTimeout` — the agent must handle this gracefully (proceed with other users, don't crash or infinite-retry)
- **Branching logic:** different paths depending on whether a common slot exists
- **Data composition:** meeting agenda must reference content from a previously read file

## Reward Design

The environment provides **per-step partial progress signals** — NOT binary pass/fail rewards. This enables meaningful gradient for RL training at every step.

### Per-Step Reward Components

| Component | Value | Condition |
|-----------|-------|-----------|
| Sequence match | +0.10 | Tool+method matches optimal sequence at this position |
| Correct tool | +0.05 | Right tool but different method |
| Successful call | +0.05 | Tool executed without error |
| Error call | -0.02 | Tool returned an error |
| Duplicate call | -0.05 | Exact same tool+method+params called before |
| Invalid tool | -0.10 | Tool name not recognized |

### Completion Bonus
When all required tools have been called the sufficient number of times, the episode ends and a **deterministic grader** evaluates the full episode history. The grader score (0.0-1.0) is multiplied by 0.5 and added as a completion bonus. Final reward is normalized to [0.0, 1.0].

### Why This Works for RL
Traditional binary rewards (0 or 1 at the end) provide no gradient for partial progress. This environment's compound signal gives the agent meaningful reward at every step:
- Early steps: reward from matching the optimal tool sequence
- Mid steps: reward from successful execution and data flow
- Final: large completion bonus from grader evaluation

Example reward progression for a perfect easy episode: `[0.0, 0.15, 0.3, 0.45, 0.6, 1.0]`

## Setup & Usage

### Local Setup

```bash
cd tool_orchestration_env
pip install -e .

# Start the server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t tool-orch -f server/Dockerfile .
docker run -p 7860:7860 tool-orch
```

### Connect with Client

```python
from tool_orchestration_env.client import ToolOrchestrationEnv
from tool_orchestration_env.models import ToolOrchestrationAction

client = ToolOrchestrationEnv(base_url="http://localhost:7860")
with client.sync() as sync:
    result = sync.reset(task_id="easy")
    result = sync.step(ToolOrchestrationAction(
        tool_name="database",
        method="query",
        parameters={"sql": "SELECT * FROM employees WHERE hire_date > '2026-03-01'"}
    ))
    print(result.observation.tool_response)
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment (standard OpenEnv) |
| `/step` | POST | Execute action (standard OpenEnv) |
| `/state` | GET | Get current state (standard OpenEnv) |
| `/schema` | GET | Get action/observation schemas |
| `/health` | GET | Health check |
| `/ws` | WS | WebSocket for persistent sessions |
| `/tasks` | GET | List all tasks with action schemas |
| `/grader` | POST | Grade last completed episode |
| `/baseline` | POST | Run LLM inference on all tasks |

## Running the Baseline

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_api_key
export ENV_URL=http://localhost:7860
python inference.py
```

## Baseline Scores

| Task | Score | Difficulty |
|------|-------|------------|
| Easy | ~0.82 | Low |
| Medium | ~0.55 | Medium |
| Hard | ~0.22 | High |

*Scores obtained with GPT-4o-mini. Frontier models should score higher on easy/medium but still struggle with the hard task's error handling and branching logic.*

## Architecture

All 6 tools are **pure Python, in-memory simulators** with zero external dependencies:
- **Database:** sqlite3 in-memory with 3 tables (employees, invoices, projects)
- **Email:** In-memory inbox/outbox
- **Filestore:** In-memory dict-based virtual filesystem
- **Calculator:** Safe math evaluation with `eval` whitelist
- **Calendar:** In-memory event store with deterministic error injection
- **Validator:** Schema validation against predefined schemas

Grading is **100% deterministic** — same episode history always produces the same score. No randomness, no LLM calls in grading.

```
tool_orchestration_env/
├── inference.py              # Baseline LLM agent (project root)
├── models.py                 # Action, Observation, State types
├── client.py                 # WebSocket client
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml            # Dependencies
└── server/
    ├── app.py                # FastAPI + hackathon endpoints
    ├── tool_orchestration_env_environment.py  # Core environment
    ├── tasks.py              # Task definitions
    ├── grader.py             # Deterministic grading
    ├── Dockerfile            # Docker image
    └── tools/
        ├── database.py       # SQLite in-memory (15 employees, 47 invoices, 5 projects)
        ├── email.py          # In-memory inbox/outbox
        ├── filestore.py      # In-memory virtual filesystem
        ├── calculator.py     # Safe math evaluator
        ├── calendar.py       # Event store with error injection
        └── validator.py      # JSON schema validation
```
