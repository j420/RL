#!/usr/bin/env python3
"""
Live integration test against a running HF Space.
No LLM API key needed — uses scripted actions.

Usage:
    python test_live.py                                          # tests local server
    python test_live.py https://madfrog1-mto.hf.space            # tests HF Space
"""

import sys
import requests

from tool_orchestration_env.client import ToolOrchestrationEnv
from tool_orchestration_env.models import ToolOrchestrationAction

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        failed += 1


# =====================================================================
print(f"\n{'='*60}")
print(f"Testing: {URL}")
print(f"{'='*60}")

# --- REST endpoint tests ---
print(f"\n--- REST Endpoints ---")

r = requests.get(f"{URL}/health", timeout=10)
check("/health returns 200", r.status_code == 200, f"got {r.status_code}")

r = requests.get(f"{URL}/tasks", timeout=10)
check("/tasks returns 3 tasks", len(r.json().get("tasks", [])) == 3)

r = requests.get(f"{URL}/schema", timeout=10)
schema = r.json()
check("/schema has action+observation", "action" in schema and "observation" in schema)

r = requests.post(f"{URL}/reset", json={}, timeout=10)
check("/reset POST {} returns 200", r.status_code == 200)
reset_data = r.json()
check("/reset has observation", "observation" in reset_data)

# --- WebSocket full episode tests ---
print(f"\n--- WebSocket: Easy Task (perfect episode) ---")

client = ToolOrchestrationEnv(base_url=URL).sync()
with client:
    result = client.reset(task_id="easy")
    obs = result.observation
    check("reset returns task description", len(obs.task_description) > 20)
    check("reset returns 6 tools", len(obs.available_tools) == 6)
    check("reset step_number=0", obs.step_number == 0)
    check("reset done=False", result.done is False)

    # Step 1: Query new hires
    result = client.step(ToolOrchestrationAction(
        tool_name="database", method="query",
        parameters={"sql": "SELECT * FROM employees WHERE hire_date > '2026-03-01'"}
    ))
    check("query finds 4 new hires", result.observation.tool_response.get("row_count") == 4)
    check("reward > 0 after correct step", (result.reward or 0) > 0)

    # Steps 2-5: Send welcome emails
    new_hires = [
        ("Carol Johnson", "Engineering", "carol.johnson@acme.com"),
        ("David Kim", "Engineering", "david.kim@acme.com"),
        ("Hannah Davis", "Marketing", "hannah.davis@acme.com"),
        ("Lisa Tanaka", "Finance", "lisa.tanaka@acme.com"),
    ]
    for name, dept, email in new_hires:
        result = client.step(ToolOrchestrationAction(
            tool_name="email", method="send",
            parameters={
                "to": email,
                "subject": f"Welcome to the team, {name}!",
                "body": f"Welcome to the {dept} department.",
            }
        ))

    check("easy perfect: done=True", result.done is True)
    check("easy perfect: score>=0.95", result.reward >= 0.95, f"got {result.reward}")

    # Check state
    state = client.state()
    check("state.task_id=easy", state.task_id == "easy")
    check("state.done=True", state.done is True)

# --- Medium Task ---
print(f"\n--- WebSocket: Medium Task (perfect episode) ---")

client = ToolOrchestrationEnv(base_url=URL).sync()
with client:
    result = client.reset(task_id="medium")

    # Query invoices
    result = client.step(ToolOrchestrationAction(
        tool_name="database", method="query",
        parameters={"sql": "SELECT * FROM invoices WHERE date LIKE '2026-03%'"}
    ))
    check("medium query: 47 invoices", result.observation.tool_response.get("row_count") == 47)

    # Calculator
    rows = result.observation.tool_response.get("rows", [])
    result = client.step(ToolOrchestrationAction(
        tool_name="calculator", method="group_sum",
        parameters={"data": rows, "group_by": "category", "aggregate": "amount"}
    ))
    totals = result.observation.tool_response.get("result", {})
    check("medium calc: Software=8940", totals.get("Software") == 8940.0)

    # Write report
    report = "# March 2026 Expense Report\n\n| Category | Total |\n|---|---|\n| Software | 8940 |\n| Travel | 3200 |\n| Office Supplies | 1450 |\n| Marketing | 5600 |\n| **Grand Total** | **19190** |"
    result = client.step(ToolOrchestrationAction(
        tool_name="filestore", method="write",
        parameters={"path": "reports/march-2026-expenses.md", "content": report}
    ))

    # Email
    result = client.step(ToolOrchestrationAction(
        tool_name="email", method="send",
        parameters={
            "to": "finance@acme.com",
            "subject": "March 2026 Expense Report",
            "body": "Please find the March expense report attached.",
            "attachment": "reports/march-2026-expenses.md",
        }
    ))
    check("medium perfect: done=True", result.done is True)
    check("medium perfect: score>=0.95", result.reward >= 0.95, f"got {result.reward}")

# --- Hard Task ---
print(f"\n--- WebSocket: Hard Task (perfect episode) ---")

client = ToolOrchestrationEnv(base_url=URL).sync()
with client:
    result = client.reset(task_id="hard")

    # Project lookup
    result = client.step(ToolOrchestrationAction(
        tool_name="database", method="query",
        parameters={"sql": "SELECT * FROM projects WHERE name = 'Project Alpha'"}
    ))
    check("hard: found Project Alpha", result.observation.tool_response.get("row_count") == 1)

    # Team query
    result = client.step(ToolOrchestrationAction(
        tool_name="database", method="query",
        parameters={"sql": "SELECT * FROM employees WHERE department = 'Engineering'"}
    ))
    check("hard: 4 engineering members", result.observation.tool_response.get("row_count") == 4)

    # Find free slots (with error injection for user_03)
    result = client.step(ToolOrchestrationAction(
        tool_name="calendar", method="find_free_slots",
        parameters={
            "users": ["user_01", "user_02", "user_03", "user_04"],
            "date_range": {"start": "2026-04-01", "end": "2026-04-07"},
            "duration_minutes": 60,
        }
    ))
    resp = result.observation.tool_response
    check("hard: user_03 unavailable", "user_03" in resp.get("unavailable_users", []))
    check("hard: free slots found", len(resp.get("slots", [])) > 0)

    # Create event
    result = client.step(ToolOrchestrationAction(
        tool_name="calendar", method="create_event",
        parameters={
            "title": "Q2 Review",
            "attendees": ["user_01", "user_02", "user_04"],
            "start": "2026-04-03T10:00",
            "end": "2026-04-03T11:00",
        }
    ))
    check("hard: event created", result.observation.tool_response.get("status") == "created")

    # Read Q1 review
    result = client.step(ToolOrchestrationAction(
        tool_name="filestore", method="read",
        parameters={"path": "projects/q1-review.md"}
    ))
    check("hard: read q1-review.md", "Accomplishments" in result.observation.tool_response.get("content", ""))

    # Write agenda
    result = client.step(ToolOrchestrationAction(
        tool_name="filestore", method="write",
        parameters={
            "path": "meetings/q2-review-agenda.md",
            "content": "Q2 Review Agenda based on q1 accomplishments and v2.0 launch",
        }
    ))

    # Email
    result = client.step(ToolOrchestrationAction(
        tool_name="email", method="send",
        parameters={
            "to": "engineering@acme.com",
            "subject": "Q2 Review Meeting Invitation",
            "body": "Please join the Q2 review meeting.",
        }
    ))
    check("hard perfect: done=True", result.done is True)
    check("hard perfect: score>=0.95", result.reward >= 0.95, f"got {result.reward}")

# --- Edge cases ---
print(f"\n--- WebSocket: Edge Cases ---")

client = ToolOrchestrationEnv(base_url=URL).sync()
with client:
    # Invalid tool
    result = client.reset(task_id="easy")
    result = client.step(ToolOrchestrationAction(tool_name="FAKE", method="x", parameters={}))
    check("invalid tool: returns error", "error" in result.observation.tool_response)
    check("invalid tool: reward<=0.05", result.reward <= 0.05)

    # Step after done
    result = client.reset(task_id="easy")
    result = client.step(ToolOrchestrationAction(
        tool_name="database", method="query",
        parameters={"sql": "SELECT * FROM employees WHERE hire_date > '2026-03-01'"}
    ))
    for email in ["carol.johnson@acme.com", "david.kim@acme.com", "hannah.davis@acme.com", "lisa.tanaka@acme.com"]:
        result = client.step(ToolOrchestrationAction(
            tool_name="email", method="send",
            parameters={"to": email, "subject": "Welcome!", "body": "Hello Engineering"}
        ))
    check("edge: episode completes", result.done is True)

    result = client.step(ToolOrchestrationAction(tool_name="database", method="query", parameters={"sql": "SELECT 1"}))
    check("edge: step after done returns error", "error" in result.observation.tool_response)
    check("edge: still done=True", result.done is True)

# --- Summary ---
print(f"\n{'='*60}")
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAILURES: {failed}")
print(f"{'='*60}\n")

sys.exit(0 if failed == 0 else 1)
