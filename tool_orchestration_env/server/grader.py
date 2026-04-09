"""Deterministic grading for the Tool Orchestration Environment.

Each task has weighted criteria checked against the episode history.
Same history always produces the same score. No randomness, no LLM calls.

The grader inspects:
- Actions taken (tool_name, method, parameters)
- Tool state (email outbox, filestore files) for verifying side effects
"""

from typing import Any, Dict, List, Optional

# Score bounds — evaluation requires strictly between 0 and 1
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _clamp(v: float) -> float:
    """Clamp a score to (0, 1) exclusive."""
    return round(max(_SCORE_MIN, min(_SCORE_MAX, float(v))), 4)


class Grader:
    """Grades completed episodes against task-specific criteria."""

    def grade(
        self,
        task_id: str,
        episode_history: List[Dict[str, Any]],
        tool_states: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Grade an episode.

        Args:
            task_id: "easy", "medium", or "hard"
            episode_history: list of {"action": {...}, "result": {...}} dicts
            tool_states: optional dict with tool state snapshots:
                - "email_outbox": list of sent emails
                - "files": dict of path -> content
                - "calendar_events_created": list of created events

        Returns:
            {"score": float 0.0-1.0, "breakdown": {criterion: score}}
        """
        try:
            if task_id == "easy":
                result = self._grade_easy(episode_history, tool_states or {})
            elif task_id == "medium":
                result = self._grade_medium(episode_history, tool_states or {})
            elif task_id == "hard":
                result = self._grade_hard(episode_history, tool_states or {})
            else:
                return {"score": _SCORE_MIN, "breakdown": {}, "error": f"Unknown task_id: {task_id}"}
        except Exception:
            # Defensive: malformed history should never crash the grader
            return {"score": _SCORE_MIN, "breakdown": {}}

        # Clamp ALL scores to (0, 1) exclusive — evaluation requires strictly between 0 and 1
        for k in result.get("breakdown", {}):
            result["breakdown"][k] = _clamp(result["breakdown"][k])
        result["score"] = _clamp(result["score"])
        return result

    # =====================================================================
    # EASY: New Employee Welcome Emails
    # =====================================================================
    def _grade_easy(
        self, history: List[Dict[str, Any]], tool_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        breakdown: Dict[str, float] = {}

        # Expected new hires (hired after 2026-03-01)
        expected_emails = {
            "carol.johnson@acme.com": ("Carol Johnson", "Engineering"),
            "david.kim@acme.com": ("David Kim", "Engineering"),
            "hannah.davis@acme.com": ("Hannah Davis", "Marketing"),
            "lisa.tanaka@acme.com": ("Lisa Tanaka", "Finance"),
        }

        # --- correct_query (0.25) ---
        # Did the agent query for employees with hire_date filter?
        query_score = 0.0
        for step in history:
            action = step.get("action") or {}
            if action.get("tool_name") == "database" and action.get("method") == "query":
                sql = str(action.get("parameters", {}).get("sql", "")).upper()
                if "HIRE_DATE" in sql and ("2026-03" in str(action.get("parameters", {}).get("sql", "")) or "2026-03-01" in str(action.get("parameters", {}).get("sql", ""))):
                    query_score = 1.0
                    break
                elif "EMPLOYEE" in sql:
                    query_score = 0.5  # Queried employees but no date filter
                    break
        breakdown["correct_query"] = query_score

        # --- correct_recipients (0.35) ---
        # Check emails sent via outbox or action history
        sent_to = set()
        outbox = tool_states.get("email_outbox", [])
        if outbox:
            for email in outbox:
                sent_to.add(email.get("to", ""))
        else:
            # Fall back to action history
            for step in history:
                action = step.get("action") or {}
                if action.get("tool_name") == "email" and action.get("method") == "send":
                    sent_to.add(action.get("parameters", {}).get("to", ""))

        expected_set = set(expected_emails.keys())
        correct_recipients = sent_to & expected_set
        extra_recipients = sent_to - expected_set

        if correct_recipients == expected_set and not extra_recipients:
            breakdown["correct_recipients"] = 1.0
        elif correct_recipients == expected_set:
            breakdown["correct_recipients"] = 0.8  # All correct but some extras
        elif len(correct_recipients) > 0:
            breakdown["correct_recipients"] = len(correct_recipients) / len(expected_set) * 0.7
        else:
            breakdown["correct_recipients"] = 0.0

        # --- correct_subjects (0.20) ---
        # Check that subjects contain employee names
        subject_score = 0.0
        subjects = []
        if outbox:
            for email in outbox:
                if email.get("to") in expected_set:
                    subjects.append((email.get("to"), email.get("subject", "")))
        else:
            for step in history:
                action = step.get("action") or {}
                if action.get("tool_name") == "email" and action.get("method") == "send":
                    to = action.get("parameters", {}).get("to", "")
                    if to in expected_set:
                        subjects.append((to, action.get("parameters", {}).get("subject", "")))

        names_found = 0
        for to, subject in subjects:
            name, _ = expected_emails.get(to, ("", ""))
            # Check if name (or first name) appears in subject
            if name and (name.lower() in subject.lower() or name.split()[0].lower() in subject.lower()):
                names_found += 1

        if len(subjects) > 0:
            subject_score = names_found / len(expected_set)
        breakdown["correct_subjects"] = min(1.0, subject_score)

        # --- correct_body (0.20) ---
        # Check that bodies mention the employee's department
        body_score = 0.0
        bodies = []
        if outbox:
            for email in outbox:
                if email.get("to") in expected_set:
                    bodies.append((email.get("to"), email.get("body", "")))
        else:
            for step in history:
                action = step.get("action") or {}
                if action.get("tool_name") == "email" and action.get("method") == "send":
                    to = action.get("parameters", {}).get("to", "")
                    if to in expected_set:
                        bodies.append((to, action.get("parameters", {}).get("body", "")))

        depts_found = 0
        for to, body in bodies:
            _, dept = expected_emails.get(to, ("", ""))
            if dept and dept.lower() in body.lower():
                depts_found += 1

        if len(bodies) > 0:
            body_score = depts_found / len(expected_set)
        breakdown["correct_body"] = min(1.0, body_score)

        # Clamp all breakdown values
        breakdown = {k: _clamp(v) for k, v in breakdown.items()}

        # Weighted score
        weights = {"correct_query": 0.25, "correct_recipients": 0.35, "correct_subjects": 0.20, "correct_body": 0.20}
        score = sum(breakdown[k] * weights[k] for k in weights)

        return {"score": _clamp(score), "breakdown": breakdown}

    # =====================================================================
    # MEDIUM: Monthly Expense Report
    # =====================================================================
    def _grade_medium(
        self, history: List[Dict[str, Any]], tool_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        breakdown: Dict[str, float] = {}

        expected_totals = {
            "Software": 8940.0,
            "Travel": 3200.0,
            "Office Supplies": 1450.0,
            "Marketing": 5600.0,
        }
        expected_grand_total = 19190.0

        # --- correct_query (0.15) ---
        query_score = 0.0
        for step in history:
            action = step.get("action") or {}
            if action.get("tool_name") == "database" and action.get("method") == "query":
                sql = str(action.get("parameters", {}).get("sql", "")).upper()
                if "INVOICE" in sql:
                    if "2026-03" in str(action.get("parameters", {}).get("sql", "")) or "MARCH" in sql:
                        query_score = 1.0
                    else:
                        query_score = 0.7  # Queried invoices but no month filter
                    break
        breakdown["correct_query"] = query_score

        # --- correct_calculation (0.20) ---
        calc_score = 0.0
        for step in history:
            action = step.get("action") or {}
            if action.get("tool_name") == "calculator":
                if action.get("method") == "group_sum":
                    calc_score = 1.0
                    break
                elif action.get("method") == "compute":
                    calc_score = 0.5  # Used calculator but not group_sum
        breakdown["correct_calculation"] = calc_score

        # --- correct_report (0.25) ---
        report_score = 0.0
        files = tool_states.get("files", {})
        report_content = ""

        # Check filestore for the written report
        for path, content in files.items():
            if "march" in path.lower() and "expense" in path.lower():
                report_content = content
                report_score = 0.3  # File exists
                break
            elif "report" in path.lower():
                report_content = content
                report_score = 0.2
                break

        if not report_content:
            # Check action history for filestore.write
            for step in history:
                action = step.get("action") or {}
                if action.get("tool_name") == "filestore" and action.get("method") == "write":
                    path = action.get("parameters", {}).get("path", "")
                    content = action.get("parameters", {}).get("content", "")
                    if content:
                        report_content = content
                        report_score = 0.3
                        break

        if report_content:
            # Check content quality
            content_lower = report_content.lower()
            categories_found = sum(
                1 for cat in expected_totals if cat.lower() in content_lower
            )
            if categories_found == 4:
                report_score = max(report_score, 0.6)

            # Check if totals appear
            totals_found = sum(
                1 for total in expected_totals.values()
                if str(int(total)) in report_content or str(total) in report_content
            )
            if totals_found == 4:
                report_score = max(report_score, 0.8)

            # Check grand total
            if str(int(expected_grand_total)) in report_content or str(expected_grand_total) in report_content:
                report_score = 1.0

        breakdown["correct_report"] = report_score

        # --- correct_email (0.20) ---
        email_score = 0.0
        outbox = tool_states.get("email_outbox", [])
        sent_emails = []
        if outbox:
            sent_emails = outbox
        else:
            for step in history:
                action = step.get("action") or {}
                if action.get("tool_name") == "email" and action.get("method") == "send":
                    sent_emails.append(action.get("parameters", {}))

        for email in sent_emails:
            to = str(email.get("to", "")).lower()
            subject = str(email.get("subject", "")).lower()
            if "finance" in to:
                email_score = 0.5  # Correct recipient
                if "expense" in subject or "march" in subject:
                    email_score = 0.8  # Correct subject
                if email.get("attachment"):
                    email_score = 1.0  # Has attachment
                break

        breakdown["correct_email"] = email_score

        # --- report_completeness (0.20) ---
        completeness_score = 0.0
        if report_content:
            content_lower = report_content.lower()
            checks = [
                any(cat.lower() in content_lower for cat in expected_totals),  # Has categories
                str(int(expected_grand_total)) in report_content,  # Has grand total
                "software" in content_lower and "travel" in content_lower,  # Multiple categories
                "|" in report_content or "---" in report_content or "total" in content_lower,  # Table/structure
            ]
            completeness_score = sum(checks) / len(checks)

        breakdown["report_completeness"] = completeness_score

        # Clamp all breakdown values
        breakdown = {k: _clamp(v) for k, v in breakdown.items()}

        # Weighted score
        weights = {
            "correct_query": 0.15, "correct_calculation": 0.20, "correct_report": 0.25,
            "correct_email": 0.20, "report_completeness": 0.20,
        }
        score = sum(breakdown[k] * weights[k] for k in weights)

        return {"score": _clamp(score), "breakdown": breakdown}

    # =====================================================================
    # HARD: Schedule Team Review Meeting
    # =====================================================================
    def _grade_hard(
        self, history: List[Dict[str, Any]], tool_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        breakdown: Dict[str, float] = {}

        # --- project_lookup (0.10) ---
        proj_score = 0.0
        for step in history:
            action = step.get("action") or {}
            if action.get("tool_name") == "database" and action.get("method") == "query":
                sql = str(action.get("parameters", {}).get("sql", ""))
                if "project" in sql.lower() and "alpha" in sql.lower():
                    proj_score = 1.0
                    break
                elif "project" in sql.lower():
                    proj_score = 0.5
        breakdown["project_lookup"] = proj_score

        # --- team_query (0.10) ---
        team_score = 0.0
        for step in history:
            action = step.get("action") or {}
            if action.get("tool_name") == "database" and action.get("method") == "query":
                sql = str(action.get("parameters", {}).get("sql", ""))
                if "employee" in sql.lower() and ("engineering" in sql.lower() or "department" in sql.lower()):
                    team_score = 1.0
                    break
        breakdown["team_query"] = team_score

        # --- calendar_check (0.15) ---
        cal_score = 0.0
        for step in history:
            action = step.get("action") or {}
            if action.get("tool_name") == "calendar":
                if action.get("method") == "find_free_slots":
                    cal_score = 1.0
                    break
                elif action.get("method") == "get_events":
                    cal_score = 0.5  # Used calendar but not find_free_slots
        breakdown["calendar_check"] = cal_score

        # --- error_handling (0.15) ---
        # In hard mode, user_03 returns CalendarServiceTimeout.
        # Good handling: no crash, max 1 retry, proceeds with other users.
        error_score = 0.0
        user_03_attempts = 0
        proceeded_after_error = False

        for i, step in enumerate(history):
            action = step.get("action") or {}
            result = step.get("result") or {}

            # Count user_03 calendar queries
            if action.get("tool_name") == "calendar":
                params = action.get("parameters", {})
                users = params.get("users", [])
                user = params.get("user", "")
                if user == "user_03" or "user_03" in users:
                    user_03_attempts += 1

                # Did they proceed to create event or do other work after the error?
                if result.get("error") == "CalendarServiceTimeout":
                    # Check if subsequent actions exist (agent didn't crash)
                    if i < len(history) - 1:
                        proceeded_after_error = True

                # Also detect find_free_slots handling user_03 internally
                # (returns unavailable_users instead of raising an error)
                if action.get("method") == "find_free_slots":
                    unavailable = result.get("unavailable_users", [])
                    if "user_03" in unavailable:
                        proceeded_after_error = True

        if proceeded_after_error:
            error_score = 0.7
            if user_03_attempts <= 2:  # Didn't infinite-retry
                error_score = 1.0
        elif user_03_attempts > 0:
            error_score = 0.3  # At least encountered the error

        breakdown["error_handling"] = error_score

        # --- meeting_created (0.15) ---
        meeting_score = 0.0
        for step in history:
            action = step.get("action") or {}
            if action.get("tool_name") == "calendar" and action.get("method") == "create_event":
                params = action.get("parameters", {})
                if params.get("title") and params.get("attendees") and params.get("start"):
                    meeting_score = 1.0
                else:
                    meeting_score = 0.5
                break
        breakdown["meeting_created"] = meeting_score

        # --- agenda_quality (0.10) ---
        agenda_score = 0.0
        # Check if agent read q1-review.md
        read_review = False
        for step in history:
            action = step.get("action") or {}
            if action.get("tool_name") == "filestore" and action.get("method") == "read":
                path = str(action.get("parameters", {}).get("path", ""))
                if "q1-review" in path or "q1_review" in path:
                    read_review = True
                    break

        # Check if agent wrote an agenda file
        wrote_agenda = False
        files = tool_states.get("files", {})
        for path, content in files.items():
            if "agenda" in path.lower() or "q2" in path.lower():
                wrote_agenda = True
                if read_review and ("q1" in content.lower() or "accomplishment" in content.lower() or "v2.0" in content.lower()):
                    agenda_score = 1.0
                else:
                    agenda_score = 0.5
                break

        if not wrote_agenda:
            for step in history:
                action = step.get("action") or {}
                if action.get("tool_name") == "filestore" and action.get("method") == "write":
                    path = str(action.get("parameters", {}).get("path", ""))
                    content = str(action.get("parameters", {}).get("content", ""))
                    if "agenda" in path.lower() or "q2" in path.lower():
                        wrote_agenda = True
                        if read_review and ("q1" in content.lower() or "accomplishment" in content.lower()):
                            agenda_score = 1.0
                        else:
                            agenda_score = 0.5
                        break

        if read_review and not wrote_agenda:
            agenda_score = 0.3  # At least read the source material

        breakdown["agenda_quality"] = agenda_score

        # --- email_sent (0.15) ---
        email_score = 0.0
        outbox = tool_states.get("email_outbox", [])
        sent_emails = []
        if outbox:
            sent_emails = outbox
        else:
            for step in history:
                action = step.get("action") or {}
                if action.get("tool_name") == "email" and action.get("method") == "send":
                    sent_emails.append(action.get("parameters", {}))

        if sent_emails:
            email_score = 0.5  # At least one email sent
            for email in sent_emails:
                body = str(email.get("body", "")).lower()
                subject = str(email.get("subject", "")).lower()
                if "q2" in subject or "review" in subject or "meeting" in subject or "agenda" in body:
                    email_score = 1.0
                    break

        breakdown["email_sent"] = email_score

        # --- edge_case (0.10) ---
        # Did the agent handle the situation correctly based on slot availability?
        edge_score = 0.0
        # In hard mode, user_03 is unavailable but a slot still exists for others.
        # The agent should proceed with available members and optionally note user_03 as unconfirmed.

        # If meeting was created AND error was handled, full marks
        if meeting_score > 0 and error_score >= 0.7:
            edge_score = 1.0
        elif meeting_score > 0 or error_score >= 0.7:
            edge_score = 0.5

        breakdown["edge_case"] = edge_score

        # Clamp all breakdown values
        breakdown = {k: _clamp(v) for k, v in breakdown.items()}

        # Weighted score
        weights = {
            "project_lookup": 0.10, "team_query": 0.10, "calendar_check": 0.15,
            "error_handling": 0.15, "meeting_created": 0.15, "agenda_quality": 0.10,
            "email_sent": 0.15, "edge_case": 0.10,
        }
        score = sum(breakdown[k] * weights[k] for k in weights)

        return {"score": _clamp(score), "breakdown": breakdown}
