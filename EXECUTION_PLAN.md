# EXECUTION PLAN — Multi-Tool Orchestration OpenEnv Environment
# Meta PyTorch OpenEnv Hackathon | Deadline: April 7, 2026 11:59 PM IST

---

## HOW TO USE THIS DOCUMENT

This is your execution guide for Claude Code. It has 3 layers:

1. **CRITICAL PATH** — The exact sequence of steps. Each step has a GATE (what must be true before moving on). Never skip a gate.

2. **CHECKPOINT MAP** — After each step, which automated checklist items are now satisfied. You can track your progress against the 20-item checklist.

3. **WIN LAYER** — At specific steps, extra work that doesn't affect pass/fail but directly increases your rubric score with human judges.

---

## EXECUTION SEQUENCE

```
PHASE A: Foundation (Steps 1-2)     → Gets you: valid scaffold, validate passes
PHASE B: Core Engine (Steps 3-5)    → Gets you: working environment with tools + tasks
PHASE C: Compliance (Steps 6-8)     → Gets you: all endpoints, inference, Docker
PHASE D: Quality Pass (Steps 9-10)  → Gets you: polished code, strong README, deploy
```

Each phase builds on the previous. If Phase A gates don't pass, nothing else matters. If Phase C gates don't pass, you're disqualified regardless of how clever your environment is.

---

## PHASE A: FOUNDATION
**Goal:** Valid OpenEnv project that passes `openenv validate`
**Time estimate:** 30-45 minutes
**Checklist items satisfied after this phase:** 1, 18

---

### STEP 1 — Scaffold + Validate Skeleton

```
1. Install openenv-core: pip install openenv-core
2. Scaffold: openenv init tool_orchestration_env
3. Examine every generated file — understand the base classes, the expected interfaces, the app.py structure
4. Run: openenv validate --verbose
5. If validate fails, fix issues until it passes
6. Show me the full file tree and the validate output
```

**GATE:** `openenv validate` passes with zero errors on the skeleton.

**Checklist items now GREEN:**
- [x] `openenv validate` passes (item 1)
- [x] `openenv.yaml` has correct metadata (item 18)

---

### STEP 2 — Define Data Models

```
ToolAction (extend the generated Action base class):
  - tool_name: str  # one of: database, email, filestore, calculator, calendar, validator
  - method: str     # method name on that tool
  - parameters: dict  # method-specific params, default empty dict

ToolObservation (extend the generated Observation base class):
  - tool_response: dict       # result from tool call or error
  - task_description: str     # current task text
  - available_tools: list[str]  # tool names agent can call
  - workspace: dict           # outputs from previous steps, keyed "step_0", "step_1" etc
  - step_number: int
  - max_steps: int
  - done: bool
  - reward: float

State model:
  - episode_id: str
  - task_id: str
  - step_count: int
  - total_reward: float
  - tools_called: list[str]
  - done: bool
```

**GATE:** `openenv validate` still passes after model changes. Models are importable.

---

## PHASE B: CORE ENGINE
**Goal:** Working environment with 6 tools, 3 tasks, graders, rewards
**Time estimate:** 2-3 hours
**Checklist items satisfied after this phase:** 4, 5, 6, 7, 8, 9, 10, 11

---

### STEP 3 — Build the 6 Simulated Tools

All pure Python, zero external API calls, fully deterministic. Each tool has reset(), execute(method, params) -> dict, and describe() -> dict.

#### database.py — DatabaseTool
sqlite3 in-memory. On reset(), creates and seeds 3 tables:
- employees: 15 rows (id, name, email, department, hire_date, salary). Departments: Engineering, Marketing, Finance, HR. 4 employees hired after 2026-03-01.
- invoices: 47 rows (id, vendor, amount, category, date, status). Categories: Software($8940 total), Travel($3200), Office Supplies($1450), Marketing($5600). Grand total: $19,190.
- projects: 5 rows (id, name, team_lead_email, department, status, deadline). "Project Alpha" has team_lead in Engineering.

Methods: query(sql) -> {rows, row_count}, insert(table, data) -> {inserted, id}
Validate SQL: only SELECT and INSERT allowed.

**IMPORTANT:** The seed data numbers MUST be exact because graders check against them.

#### email.py — EmailTool
In-memory inbox and outbox.
On reset(), seed inbox with 5 emails.
Methods: send(to, subject, body, attachment=None) -> {status, message_id}, search(query) -> {results, count}, list_inbox() -> {emails}

#### filestore.py — FileStoreTool
In-memory dict of path -> content.
On reset(), seed with project files and templates.
Methods: read(path) -> {content, size_bytes}, write(path, content) -> {status, path, size_bytes}, list(directory) -> {files}

#### calculator.py — CalculatorTool
Stateless math evaluator.
Methods: compute(expression) -> {result}, group_sum(data, group_by, aggregate) -> {result}, date_diff(date1, date2) -> {days}

#### calendar.py — CalendarTool
In-memory event store for 5 users.
On reset(task_id), seed with events. Common free slot on April 3, 2026 10:00-11:00.
Error injection: If task_id == "hard", user_03 returns CalendarServiceTimeout.
Methods: get_events(user, date_range), find_free_slots(users, date_range, duration_minutes), create_event(title, attendees, start, end)

#### validator.py — ValidatorTool
JSON schema validation.
Methods: validate(data, schema_name) -> {valid, errors}
Schemas: "expense_report", "meeting_invite", "email_format"

**GATE:** All 6 tools can be imported, reset(), and execute() without errors. Test script passes.

---

### STEP 4 — Define Tasks + Graders

#### 3 Tasks:

**TASK "easy"** — New Employee Welcome Emails (max 10 steps)
- Query DB for employees hired after 2026-03-01
- Send welcome emails to each new hire
- Grading: correct_query(0.25), correct_recipients(0.35), correct_subjects(0.20), correct_body(0.20)

**TASK "medium"** — Monthly Expense Report (max 15 steps)
- Query invoices, calculate totals, write report, email it
- Grading: correct_query(0.15), correct_calculation(0.20), correct_report(0.25), correct_email(0.20), report_completeness(0.20)

**TASK "hard"** — Schedule Team Review Meeting (max 20 steps)
- Look up project, find team, check calendars (with error handling), create event, write agenda, email team
- Grading: project_lookup(0.10), team_query(0.10), calendar_check(0.15), error_handling(0.15), meeting_created(0.15), agenda_quality(0.10), email_sent(0.15), edge_case(0.10)

#### Grader:
- 100% deterministic — same history = same score
- Each criterion checked independently against episode history
- Score = weighted sum of criteria scores (0.0-1.0)

**GATE:** Grader returns correct scores for manually constructed perfect and imperfect episodes.

---

### STEP 5 — Wire Up Environment Logic

Core environment class with reset(), step(), state() methods.

**reset(task_id):** Initialize tools, clear state, return initial observation
**step(action):** Validate action, execute tool, calculate per-step reward, check completion
**state():** Return current episode metadata

**Reward Design:**
- +0.1 for correct tool at correct sequence position
- +0.05 for successful non-error call
- -0.05 for duplicate calls
- -0.1 for invalid tool names
- Completion bonus: grader_score * 0.5
- Final reward normalized to 0.0-1.0

**GATE:** `openenv validate` passes, reset/step work, rewards vary per step, done=True after completion.

---

## PHASE C: COMPLIANCE
**Goal:** All endpoints work, inference runs, Docker builds
**Time estimate:** 1.5-2 hours

---

### STEP 6 — Hackathon-Specific Endpoints

Three additional REST endpoints on FastAPI:
- **GET /tasks** — Returns all 3 tasks with action schemas
- **POST /grader** — Runs grader on last completed episode
- **POST /baseline** — Triggers inference for all 3 tasks

**GATE:** All 3 endpoints return valid JSON.

---

### STEP 7 — Inference Script

`inference.py` at PROJECT ROOT:
- Reads: API_BASE_URL, MODEL_NAME, HF_TOKEN
- Uses openai package with configurable base_url
- Loops: reset → LLM decides action → step → repeat until done
- Handles LLM errors gracefully (retry once, then fallback)

**GATE:** inference.py runs without crashes, produces 3 scores.

---

### STEP 8 — Dockerfile + Local Docker Test

- Base: python:3.11-slim
- Port: 7860 (HF Spaces default)
- All deps installed

**GATE:** `docker build` succeeds, `docker run` starts server, endpoints respond.

---

## PHASE D: QUALITY PASS
**Goal:** Polish everything, README, deploy, verify

---

### STEP 9 — README + Documentation

HF Space frontmatter + sections:
- Why This Exists (motivation)
- Action Space (every tool, every method, with examples)
- Observation Space
- Tasks (all 3 with difficulty explanation)
- Reward Design
- Setup & Usage
- Baseline Scores
- Architecture

---

### STEP 10 — Deploy + Final Verification

Run all 20 checklist items. Deploy to HF Space. Verify live.

---

## 20-ITEM CHECKLIST

### Automated Gates
- [ ] 1.  openenv validate passes
- [ ] 2.  docker build && docker run works
- [ ] 3.  HF Space deploys and responds to reset()
- [ ] 4.  reset() produces clean initial observation
- [ ] 5.  step()/state() work correctly
- [ ] 6.  3 tasks exist (easy, medium, hard)
- [ ] 7.  Graders produce deterministic 0.0-1.0 scores

### Functional
- [ ] 8.  Action/Observation types well-designed and documented
- [ ] 9.  Reward provides per-step partial progress signal
- [ ] 10. Reward penalizes wasteful/destructive actions
- [ ] 11. inference.py exists at project root
- [ ] 12. inference.py reads API_BASE_URL, MODEL_NAME, HF_TOKEN
- [ ] 13. /tasks endpoint returns task list with action schema
- [ ] 14. /grader endpoint returns grader score
- [ ] 15. /baseline endpoint triggers inference
- [ ] 16. README has all 5 required sections
- [ ] 17. Hard task scores < 0.5 for frontier models
- [ ] 18. openenv.yaml has correct metadata
- [ ] 19. Zero external dependencies for environment
- [ ] 20. All above verified end-to-end

---

## RUBRIC MAPPING

| Rubric Category (weight) | Where it's built | Key differentiator |
|---|---|---|
| Real-world utility (30%) | Step 3 (tools), Step 9 (README) | No multi-tool env exists in OpenEnv catalog |
| Task & grader quality (25%) | Step 4 (tasks + graders) | Hard task error injection + branching |
| Environment design (20%) | Step 5 (env logic) | Workspace mechanic + per-step reward signal |
| Code quality (15%) | All steps | Start from openenv init, type hints, clean errors |
| Creativity & novelty (10%) | Step 4, Step 9 | CalendarServiceTimeout, workspace data flow, branching |
