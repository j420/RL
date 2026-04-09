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
        # Check SQL semantics: must query employees with proper date filter
        query_score = 0.0
        for step in history:
            action = step.get("action") or {}
            result = step.get("result") or {}
            if action.get("tool_name") == "database" and action.get("method") == "query":
                sql_raw = str(action.get("parameters", {}).get("sql", ""))
                sql = sql_raw.upper()
                has_employee_table = "EMPLOYEE" in sql
                has_date_filter = (
                    "HIRE_DATE" in sql
                    and any(op in sql for op in [">", ">=", "BETWEEN", "AFTER"])
                    and ("2026-03" in sql_raw or "2026-03-01" in sql_raw)
                )
                # Verify query actually returned correct results
                rows = result.get("rows", [])
                returned_emails = {r.get("email", "") for r in rows if isinstance(r, dict)}
                expected_set_emails = set(expected_emails.keys())

                if has_employee_table and has_date_filter:
                    query_score = 0.7
                    # Bonus: verify result correctness
                    if returned_emails >= expected_set_emails:
                        query_score = 1.0
                    elif len(returned_emails & expected_set_emails) > 0:
                        query_score = 0.85
                    break
                elif has_employee_table:
                    query_score = 0.4  # Queried employees but no date filter
                    if returned_emails >= expected_set_emails:
                        query_score = 0.6  # Got right results anyway
                    break
        breakdown["correct_query"] = query_score

        # --- correct_recipients (0.30) ---
        # Check emails sent via outbox or action history
        sent_to = set()
        sent_to_list = []  # Track order for duplicate detection
        outbox = tool_states.get("email_outbox", [])
        if outbox:
            for email in outbox:
                sent_to.add(email.get("to", ""))
                sent_to_list.append(email.get("to", ""))
        else:
            for step in history:
                action = step.get("action") or {}
                if action.get("tool_name") == "email" and action.get("method") == "send":
                    to = action.get("parameters", {}).get("to", "")
                    sent_to.add(to)
                    sent_to_list.append(to)

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

        # Penalty for duplicate emails to same recipient
        from collections import Counter
        dup_count = sum(1 for c in Counter(sent_to_list).values() if c > 1)
        if dup_count > 0:
            breakdown["correct_recipients"] = max(0.0, breakdown["correct_recipients"] - 0.15 * dup_count)

        # --- correct_subjects (0.20) ---
        # Check that subjects contain employee names and "welcome" keyword
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
        welcome_found = 0
        for to, subject in subjects:
            name, _ = expected_emails.get(to, ("", ""))
            if name and (name.lower() in subject.lower() or name.split()[0].lower() in subject.lower()):
                names_found += 1
            if "welcome" in subject.lower():
                welcome_found += 1

        if len(subjects) > 0:
            name_ratio = names_found / len(expected_set)
            welcome_ratio = welcome_found / len(subjects)
            subject_score = name_ratio * 0.7 + welcome_ratio * 0.3
        breakdown["correct_subjects"] = min(1.0, subject_score)

        # --- correct_body (0.25) ---
        # Check that bodies mention the employee's department and name
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
        names_in_body = 0
        for to, body in bodies:
            name, dept = expected_emails.get(to, ("", ""))
            if dept and dept.lower() in body.lower():
                depts_found += 1
            if name and (name.lower() in body.lower() or name.split()[0].lower() in body.lower()):
                names_in_body += 1

        if len(bodies) > 0:
            dept_ratio = depts_found / len(expected_set)
            name_ratio = names_in_body / len(expected_set)
            body_score = dept_ratio * 0.7 + name_ratio * 0.3
        breakdown["correct_body"] = min(1.0, body_score)

        # Clamp all breakdown values
        breakdown = {k: _clamp(v) for k, v in breakdown.items()}

        # Weighted score
        weights = {"correct_query": 0.25, "correct_recipients": 0.30, "correct_subjects": 0.20, "correct_body": 0.25}
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
        # Validate SQL semantics: must query invoices with March 2026 date filter
        query_score = 0.0
        query_returned_correct = False
        for step in history:
            action = step.get("action") or {}
            result = step.get("result") or {}
            if action.get("tool_name") == "database" and action.get("method") == "query":
                sql_raw = str(action.get("parameters", {}).get("sql", ""))
                sql = sql_raw.upper()
                if "INVOICE" in sql:
                    has_date_filter = (
                        "2026-03" in sql_raw
                        or "MARCH" in sql
                        or ("DATE" in sql and "2026" in sql_raw)
                    )
                    # Verify query returned reasonable data
                    row_count = result.get("row_count", 0)
                    if has_date_filter:
                        query_score = 0.7
                        if row_count == 47:  # All March invoices
                            query_score = 1.0
                        elif row_count > 0:
                            query_score = 0.85
                    else:
                        query_score = 0.5  # Queried invoices but no month filter
                    break
        breakdown["correct_query"] = query_score

        # --- correct_calculation (0.25) ---
        # Verify the calculation results match expected totals
        calc_score = 0.0
        for step in history:
            action = step.get("action") or {}
            result = step.get("result") or {}
            if action.get("tool_name") == "calculator":
                if action.get("method") == "group_sum":
                    calc_score = 0.5  # Used correct method
                    # Verify results match expected totals
                    group_result = result.get("result", {})
                    if isinstance(group_result, dict):
                        correct_totals = 0
                        for cat, expected_val in expected_totals.items():
                            actual = group_result.get(cat, None)
                            if actual is not None and abs(float(actual) - expected_val) < 0.01:
                                correct_totals += 1
                        if correct_totals == 4:
                            calc_score = 1.0
                        elif correct_totals > 0:
                            calc_score = 0.5 + (correct_totals / 4) * 0.4
                    break
                elif action.get("method") == "compute":
                    # Check if compute result matches any expected value
                    compute_result = result.get("result")
                    if compute_result is not None:
                        if abs(float(compute_result) - expected_grand_total) < 0.01:
                            calc_score = 0.6  # Computed grand total directly
                        else:
                            calc_score = 0.3  # Used calculator but unclear result
        breakdown["correct_calculation"] = calc_score

        # --- correct_report (0.20) ---
        report_score = 0.0
        report_path = ""
        files = tool_states.get("files", {})
        report_content = ""

        # Check filestore for the written report
        for path, content in files.items():
            if "march" in path.lower() and "expense" in path.lower():
                report_content = content
                report_path = path
                report_score = 0.3  # File with correct name
                break
            elif "report" in path.lower() or "expense" in path.lower():
                report_content = content
                report_path = path
                report_score = 0.2
                break

        if not report_content:
            for step in history:
                action = step.get("action") or {}
                if action.get("tool_name") == "filestore" and action.get("method") == "write":
                    path = action.get("parameters", {}).get("path", "")
                    content = action.get("parameters", {}).get("content", "")
                    if content:
                        report_content = content
                        report_path = path
                        report_score = 0.2
                        break

        if report_content:
            content_lower = report_content.lower()
            categories_found = sum(
                1 for cat in expected_totals if cat.lower() in content_lower
            )
            if categories_found == 4:
                report_score = max(report_score, 0.6)

            # Check if totals appear (verify exact values)
            totals_found = sum(
                1 for total in expected_totals.values()
                if str(int(total)) in report_content or f"{total:.2f}" in report_content or str(total) in report_content
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
                email_score = 0.4  # Correct recipient
                if "expense" in subject or "march" in subject:
                    email_score = 0.6  # Correct subject
                attachment = email.get("attachment", "")
                if attachment:
                    email_score = 0.8  # Has attachment
                    # Verify attachment path matches the report file
                    if report_path and attachment == report_path:
                        email_score = 1.0  # Attachment matches actual report
                    elif report_path and ("expense" in str(attachment).lower() or "report" in str(attachment).lower()):
                        email_score = 0.9  # Reasonable attachment path
                break

        breakdown["correct_email"] = email_score

        # --- report_completeness (0.20) ---
        completeness_score = 0.0
        if report_content:
            content_lower = report_content.lower()
            checks = [
                all(cat.lower() in content_lower for cat in expected_totals),  # Has ALL categories
                str(int(expected_grand_total)) in report_content,  # Has grand total
                "software" in content_lower and "travel" in content_lower
                and "office" in content_lower and "marketing" in content_lower,  # All 4 categories
                "|" in report_content or "---" in report_content,  # Markdown table formatting
                "total" in content_lower,  # Mentions total
                report_content.count("\n") >= 4,  # Multi-line report (not one-liner)
            ]
            completeness_score = sum(checks) / len(checks)

        breakdown["report_completeness"] = completeness_score

        # Clamp all breakdown values
        breakdown = {k: _clamp(v) for k, v in breakdown.items()}

        # Weighted score
        weights = {
            "correct_query": 0.15, "correct_calculation": 0.25, "correct_report": 0.20,
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
        # Verify meeting is scheduled at the correct time slot (April 3, 10:00-11:00)
        meeting_score = 0.0
        meeting_time_correct = False
        for step in history:
            action = step.get("action") or {}
            if action.get("tool_name") == "calendar" and action.get("method") == "create_event":
                params = action.get("parameters", {})
                if params.get("title") and params.get("attendees") and params.get("start"):
                    meeting_score = 0.6  # Event created with required fields
                    # Check if time matches the known free slot
                    start = str(params.get("start", ""))
                    end = str(params.get("end", ""))
                    if "2026-04-03" in start and "10:00" in start:
                        meeting_score = 0.8  # Correct start time
                        if "11:00" in end:
                            meeting_score = 1.0  # Correct start AND end
                    # Check attendees include engineering team
                    attendees = params.get("attendees", [])
                    if isinstance(attendees, list) and len(attendees) >= 2:
                        meeting_time_correct = True
                elif params.get("title"):
                    meeting_score = 0.3  # Event created but incomplete
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

        # Check if agent wrote an agenda file — verify content quality
        wrote_agenda = False
        agenda_content = ""
        files = tool_states.get("files", {})
        for path, content in files.items():
            if "agenda" in path.lower() or ("q2" in path.lower() and "review" in path.lower()):
                wrote_agenda = True
                agenda_content = content
                break

        if not wrote_agenda:
            for step in history:
                action = step.get("action") or {}
                if action.get("tool_name") == "filestore" and action.get("method") == "write":
                    path = str(action.get("parameters", {}).get("path", ""))
                    content = str(action.get("parameters", {}).get("content", ""))
                    if "agenda" in path.lower() or ("q2" in path.lower() and "review" in path.lower()):
                        wrote_agenda = True
                        agenda_content = content
                        break

        if wrote_agenda and agenda_content:
            content_lower = agenda_content.lower()
            # Check for specific Q1 review items that show the agent used the source material
            q1_refs = [
                "v2.0" in content_lower or "shipped" in content_lower,
                "latency" in content_lower or "api" in content_lower,
                "mobile" in content_lower or "app" in content_lower,
                "series b" in content_lower or "hiring" in content_lower,
            ]
            q1_ref_count = sum(q1_refs)

            if read_review and q1_ref_count >= 2:
                agenda_score = 1.0  # Read source + references specific items
            elif read_review and q1_ref_count >= 1:
                agenda_score = 0.8
            elif read_review:
                agenda_score = 0.6  # Read source but generic agenda
            elif q1_ref_count >= 1:
                agenda_score = 0.5  # Somehow referenced items without reading
            else:
                agenda_score = 0.3  # Wrote agenda but no Q1 references
        elif read_review and not wrote_agenda:
            agenda_score = 0.2  # At least read the source material

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
            email_score = 0.3  # At least one email sent
            for email in sent_emails:
                body = str(email.get("body", "")).lower()
                subject = str(email.get("subject", "")).lower()
                to = str(email.get("to", "")).lower()
                has_relevant_subject = "q2" in subject or "review" in subject or "meeting" in subject
                has_relevant_body = "agenda" in body or "meeting" in body or "q2" in body
                has_attachment = bool(email.get("attachment"))

                if has_relevant_subject:
                    email_score = 0.6
                if has_relevant_subject and has_relevant_body:
                    email_score = 0.8
                if has_relevant_subject and has_attachment:
                    email_score = 1.0
                    break
                if has_relevant_subject and has_relevant_body:
                    email_score = max(email_score, 0.8)

        breakdown["email_sent"] = email_score

        # --- edge_case (0.10) ---
        # In hard mode, user_03 is unavailable but a slot still exists for others.
        # The agent should proceed with available members and note user_03 as unconfirmed.
        edge_score = 0.0

        # Check if agent mentioned user_03 status in email or agenda
        mentioned_unavailable = False
        for email in sent_emails:
            body = str(email.get("body", "")).lower()
            if "user_03" in body or "unconfirmed" in body or "unavailable" in body:
                mentioned_unavailable = True
                break
        if agenda_content and ("user_03" in agenda_content.lower() or "unconfirmed" in agenda_content.lower() or "unavailable" in agenda_content.lower()):
            mentioned_unavailable = True

        if meeting_score > 0 and error_score >= 0.7:
            edge_score = 0.8
            if mentioned_unavailable:
                edge_score = 1.0  # Full marks: handled error + noted it
        elif meeting_score > 0 or error_score >= 0.7:
            edge_score = 0.5
        elif mentioned_unavailable:
            edge_score = 0.3

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
