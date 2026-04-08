"""Task definitions for the Tool Orchestration Environment.

Three tasks of increasing difficulty, each with:
- description: what the agent must accomplish
- optimal_tool_sequence: ideal sequence of tool calls
- max_steps: step budget
- grading_criteria: weighted criteria for scoring (sum to 1.0)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Task:
    task_id: str
    description: str
    difficulty: str
    optimal_tool_sequence: List[str]
    max_steps: int
    grading_criteria: Dict[str, float]  # criterion_name -> weight (sum to 1.0)


TASKS: Dict[str, Task] = {
    "easy": Task(
        task_id="easy",
        difficulty="easy",
        description=(
            "Find all employees hired in the last 30 days (after 2026-03-01) from the database. "
            "For each new hire, send a welcome email to their email address with subject "
            "'Welcome to the team, [NAME]!' and body that mentions their department. "
            "Today's date is 2026-04-01."
        ),
        optimal_tool_sequence=["database.query", "email.send", "email.send", "email.send", "email.send"],
        max_steps=10,
        grading_criteria={
            "correct_query": 0.25,
            "correct_recipients": 0.35,
            "correct_subjects": 0.20,
            "correct_body": 0.20,
        },
    ),
    "medium": Task(
        task_id="medium",
        difficulty="medium",
        description=(
            "Generate the March 2026 expense report: "
            "(1) Query all invoices from March 2026 from the database, "
            "(2) Use the calculator to compute total amount per category, "
            "(3) Save a formatted markdown report to 'reports/march-2026-expenses.md' with a table "
            "of categories and totals plus grand total, "
            "(4) Email the report to 'finance@acme.com' with subject 'March 2026 Expense Report' "
            "and attach the file path."
        ),
        optimal_tool_sequence=[
            "database.query",
            "calculator.group_sum",
            "filestore.write",
            "email.send",
        ],
        max_steps=15,
        grading_criteria={
            "correct_query": 0.15,
            "correct_calculation": 0.20,
            "correct_report": 0.25,
            "correct_email": 0.20,
            "report_completeness": 0.20,
        },
    ),
    "hard": Task(
        task_id="hard",
        difficulty="hard",
        description=(
            "Schedule a Q2 review meeting: "
            "(1) Query the projects table for 'Project Alpha' to get the team lead, "
            "(2) Query all employees in the team lead's department, "
            "(3) Check calendar availability for all team members for a 60-min slot "
            "between 2026-04-01 and 2026-04-07, "
            "(4) If a common slot exists, create a calendar event, "
            "(5) Read 'projects/q1-review.md' to create a meeting agenda, "
            "write it to 'meetings/q2-review-agenda.md', "
            "(6) Email the agenda to all attendees. "
            "NOTE: If a team member's calendar service returns an error, mark them as "
            "'unconfirmed' and proceed with available members. If no common slot exists "
            "among available members, email the team lead asking them to pick a time."
        ),
        optimal_tool_sequence=[
            "database.query",
            "database.query",
            "calendar.find_free_slots",
            "calendar.create_event",
            "filestore.read",
            "filestore.write",
            "email.send",
        ],
        max_steps=20,
        grading_criteria={
            "project_lookup": 0.10,
            "team_query": 0.10,
            "calendar_check": 0.15,
            "error_handling": 0.15,
            "meeting_created": 0.15,
            "agenda_quality": 0.10,
            "email_sent": 0.15,
            "edge_case": 0.10,
        },
    ),
}


def get_task(task_id: str) -> Task:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> List[Dict[str, Any]]:
    """Return all tasks as dicts with action schema included."""
    action_schema = {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "enum": ["database", "email", "filestore", "calculator", "calendar", "validator"],
            },
            "method": {"type": "string", "description": "Method to call on the tool"},
            "parameters": {"type": "object", "description": "Method-specific parameters"},
        },
        "required": ["tool_name", "method"],
    }

    return [
        {
            "task_id": t.task_id,
            "description": t.description,
            "difficulty": t.difficulty,
            "max_steps": t.max_steps,
            "action_schema": action_schema,
        }
        for t in TASKS.values()
    ]
