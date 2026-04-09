# Tool Orchestration Environment - Developer Guide

Hard-won knowledge from building and debugging this OpenEnv hackathon submission.

## Evaluation System

The hackathon evaluator works in two layers:

1. **Stdout parsing**: Runs `inference.py` and parses structured `[START]`/`[STEP]`/`[END]` lines from stdout. The `score=` field in the `[END]` line is how the evaluator reads each task's final score.

2. **HF Space probing**: Calls REST endpoints (`/health`, `/reset`, `/step`, `/grader`, `/schema`) on the deployed HuggingFace Space.

### The [END] line format is critical

```
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
```

- `score=` **must be present** and strictly in (0, 1) exclusive — not 0.0, not 1.0
- Omitting `score=` causes: "Each task's score must be strictly between 0 and 1"
- The evaluator likely defaults missing scores to 0.0, which fails validation
- Score uses 3 decimal places (`.3f`); rewards use 2 (`.2f`)

### Score bounds: (0.01, 0.99)

The evaluation requires all scores **strictly between 0 and 1**. Exact 0.0 or 1.0 fail.
We enforce this at 4 layers:

| Layer | File | What it does |
|---|---|---|
| 1. Grader methods | `grader.py` `_grade_*()` | Clamps each breakdown value via `_clamp()` |
| 2. Grader wrapper | `grader.py` `grade()` | Re-clamps score + breakdown after method returns |
| 3. Environment | `tool_orchestration_env_environment.py` `_normalize_reward()` | Clamps combined reward to (0.01, 0.99) |
| 4. HTTP middleware | `app.py` `ScoreClampMiddleware` | Intercepts ALL JSON HTTP responses, clamps score fields |

Layer 4 (middleware) only covers HTTP responses, not WebSocket messages.
WebSocket values are already clamped by layers 1-3.

### Middleware gotcha: Content-Length

When modifying JSON response bodies in Starlette middleware, you **must strip the
original `content-length` header** before returning the new Response. Otherwise
`json.dumps()` may produce a different byte length than the original serializer
(Pydantic's `model_dump_json()`), causing:
```
h11._util.LocalProtocolError: Too much data for declared Content-Length
```

## Deployment

The repo structure is `RL/tool_orchestration_env/...`. The HF Space only gets the
inner directory via git subtree:

```bash
git checkout main && git pull origin main
ref=$(git subtree split --prefix tool_orchestration_env)
git push hf "${ref}:main" --force
```

After deploy, verify: `curl -X POST https://<space-url>/verify-scores`

## Seed Data Dependencies

The grader checks exact values against hardcoded seed data. **Do not change
database seed data without updating the grader.**

### Easy task - New hires (hired after 2026-03-01)
| Email | Name | Department |
|---|---|---|
| carol.johnson@acme.com | Carol Johnson | Engineering |
| david.kim@acme.com | David Kim | Engineering |
| hannah.davis@acme.com | Hannah Davis | Marketing |
| lisa.tanaka@acme.com | Lisa Tanaka | Finance |

### Medium task - Invoice totals (March 2026)
| Category | Total |
|---|---|
| Software | $8,940 |
| Travel | $3,200 |
| Office Supplies | $1,450 |
| Marketing | $5,600 |
| **Grand Total** | **$19,190** |

### Hard task - Calendar error injection
- `user_03` returns `CalendarServiceTimeout` on calendar queries
- `find_free_slots` reports `user_03` in `unavailable_users` list
- Common free slot for others: April 3, 2026, 10:00-11:00
- Project Alpha team lead: Engineering department

## Reward Formula

```
reward = step_fraction * 0.3 + grader_score * 0.7

where:
  step_fraction = clamp(total_step_reward / max_possible_step_reward, 0, 1)
  grader_score  = weighted sum of clamped breakdown scores (only after episode ends)
```

Per-step rewards:
- +0.10 for exact tool+method match at correct sequence position
- +0.05 for correct tool but wrong method
- +0.05 for successful execution (no error in result)
- -0.02 for tool errors
- -0.05 for duplicate calls (same tool+method+params)
- -0.10 for invalid tool name

Final reward is always clamped to (0.01, 0.99).

## Workspace Mechanic

Each step's result is stored in `workspace[f"step_{n}"]`:
```python
{"tool": "database", "method": "query", "result": {"rows": [...], "row_count": 4}}
```

Agents should reference workspace data in subsequent calls (e.g., use query results
to compose emails). The inference script sends the 3 most recent workspace entries
as context to the LLM.

## Grader State Tracking

The grader checks TWO sources for each criterion:

1. **Episode history** — list of `{"action": {...}, "result": {...}}` dicts
2. **Tool states** — snapshots of side effects:
   - `email_outbox`: list of sent emails (from EmailTool)
   - `files`: dict of path -> content (from FileStoreTool)

Tool states take priority. History is used as fallback when states are empty.

## REST vs WebSocket

- **REST endpoints** (`/reset`, `/step`) are **stateless** — each creates a fresh
  environment instance. Cannot run real episodes via REST alone.
- **WebSocket** (`/ws`) maintains persistent sessions with dedicated env instances.
  This is what `inference.py` and the evaluation client use.
- **Hackathon endpoints** (`/interact`, `/grader`, `/baseline`) share a singleton
  env via `_get_env()`. The `/grader` endpoint returns the last completed episode's
  grader result from this shared instance.

## Grader Validation Depth

The grader doesn't just check tool presence — it validates:

### SQL Semantic Validation
- Checks for proper WHERE clauses (not just keyword presence)
- Verifies query results match expected data (row counts, email sets)
- Partial credit for correct table but missing filters

### Math Result Verification
- group_sum results are compared against expected category totals
- Tolerance of 0.01 for floating point comparison
- Partial credit scaled by number of correct categories

### Attachment Validation
- Verifies attachment path matches the actual written file path
- Tiered scoring: has attachment > reasonable path > any attachment

### Duplicate Detection
- Easy task penalizes duplicate emails to the same recipient (-0.15 per dup)
- Environment penalizes duplicate tool calls (same tool+method+params)

### Meeting Time Verification (Hard)
- Checks if created event matches known free slot (April 3, 10:00-11:00)
- Tiered: correct start+end > correct start only > event exists

### Agenda Quality (Hard)
- Checks for specific Q1 review references (v2.0, latency, mobile, Series B)
- Bonus for reading source material AND referencing specific items

## Contextual Step Rewards

Beyond matching the optimal tool sequence, the environment gives quality bonuses:
- +0.02 for database queries returning data
- +0.02 for group_sum producing results
- +0.02 for successful calendar event creation
- +0.01 for successful email send / file write

## Task-Specific Inference Hints

`inference.py` provides task-specific strategy guidance on the first step of each
task via `TASK_HINTS`. This helps the LLM follow the optimal tool sequence without
hardcoding actions.

## Pre-Submission Checklist

1. `inference.py` has `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars
2. Defaults set only for `API_BASE_URL` and `MODEL_NAME` (NOT `HF_TOKEN`)
3. All LLM calls use `from openai import OpenAI` client
4. Stdout follows `[START]`/`[STEP]`/`[END]` format **exactly** (including `score=`)
5. `openenv validate` passes (both local and against running server)
6. Docker builds and runs successfully
7. HF Space deploys and responds to health checks
