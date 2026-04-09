"""Comprehensive tests for the Tool Orchestration Environment.

Covers: tool unit tests, grader determinism, reward formula,
edge cases, calculator security, and full episode runs.
"""

import sys
import os

# Ensure imports work from the test directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.tool_orchestration_env_environment import ToolOrchestrationEnvironment
from server.tools.calculator import CalculatorTool
from server.tools.database import DatabaseTool
from server.tools.email import EmailTool
from server.tools.filestore import FileStoreTool
from server.tools.calendar import CalendarTool
from server.tools.validator import ValidatorTool
from server.grader import Grader
from models import ToolOrchestrationAction


# =====================================================================
# Tool Unit Tests
# =====================================================================


class TestDatabaseTool:
    def test_query_employees(self):
        db = DatabaseTool()
        db.reset()
        result = db.execute("query", {"sql": "SELECT * FROM employees WHERE hire_date > '2026-03-01'"})
        assert "rows" in result
        assert result["row_count"] == 4

    def test_query_missing_sql(self):
        db = DatabaseTool()
        db.reset()
        result = db.execute("query", {})
        assert "error" in result

    def test_blocks_drop(self):
        db = DatabaseTool()
        db.reset()
        result = db.execute("query", {"sql": "DROP TABLE employees"})
        assert "error" in result

    def test_insert(self):
        db = DatabaseTool()
        db.reset()
        result = db.execute("insert", {"table": "employees", "data": {"id": 99, "name": "Test", "email": "test@acme.com", "department": "HR", "hire_date": "2026-04-01", "salary": 50000}})
        assert result.get("inserted") is True

    def test_unknown_method(self):
        db = DatabaseTool()
        db.reset()
        result = db.execute("delete", {})
        assert "error" in result


class TestEmailTool:
    def test_send_and_outbox(self):
        email = EmailTool()
        email.reset()
        result = email.execute("send", {"to": "test@acme.com", "subject": "Hi", "body": "Hello"})
        assert result["status"] == "sent"
        assert len(email.get_outbox()) == 1

    def test_send_missing_to(self):
        email = EmailTool()
        email.reset()
        result = email.execute("send", {"subject": "Hi", "body": "Hello"})
        assert "error" in result

    def test_search(self):
        email = EmailTool()
        email.reset()
        result = email.execute("search", {"query": "Q2 Kickoff"})
        assert result["count"] >= 1

    def test_list_inbox(self):
        email = EmailTool()
        email.reset()
        result = email.execute("list_inbox", {})
        assert len(result["emails"]) == 5

    def test_reset_clears_outbox(self):
        email = EmailTool()
        email.reset()
        email.execute("send", {"to": "a@b.com", "subject": "S", "body": "B"})
        assert len(email.get_outbox()) == 1
        email.reset()
        assert len(email.get_outbox()) == 0


class TestFileStoreTool:
    def test_read_seeded_file(self):
        fs = FileStoreTool()
        fs.reset()
        result = fs.execute("read", {"path": "projects/q1-review.md"})
        assert "content" in result
        assert "Q1 Review" in result["content"]

    def test_write_and_read(self):
        fs = FileStoreTool()
        fs.reset()
        fs.execute("write", {"path": "test/file.txt", "content": "hello"})
        result = fs.execute("read", {"path": "test/file.txt"})
        assert result["content"] == "hello"

    def test_read_missing_file(self):
        fs = FileStoreTool()
        fs.reset()
        result = fs.execute("read", {"path": "nonexistent.txt"})
        assert "error" in result

    def test_reset_clears_written_files(self):
        fs = FileStoreTool()
        fs.reset()
        fs.execute("write", {"path": "new.txt", "content": "data"})
        assert "new.txt" in fs.get_files()
        fs.reset()
        assert "new.txt" not in fs.get_files()


class TestCalendarTool:
    def test_get_events(self):
        cal = CalendarTool()
        cal.reset()
        result = cal.execute("get_events", {"user": "user_01"})
        assert "events" in result
        assert len(result["events"]) > 0

    def test_error_injection_hard_mode(self):
        cal = CalendarTool()
        cal.reset(task_id="hard")
        result = cal.execute("get_events", {"user": "user_03"})
        assert result.get("error") == "CalendarServiceTimeout"

    def test_no_error_easy_mode(self):
        cal = CalendarTool()
        cal.reset(task_id="easy")
        result = cal.execute("get_events", {"user": "user_03"})
        assert "error" not in result

    def test_find_free_slots_hard_mode_unavailable(self):
        cal = CalendarTool()
        cal.reset(task_id="hard")
        result = cal.execute("find_free_slots", {
            "users": ["user_01", "user_02", "user_03"],
            "date_range": {"start": "2026-04-01", "end": "2026-04-07"},
            "duration_minutes": 60,
        })
        assert "user_03" in result.get("unavailable_users", [])
        assert "slots" in result

    def test_create_event(self):
        cal = CalendarTool()
        cal.reset()
        result = cal.execute("create_event", {
            "title": "Test Meeting",
            "attendees": ["user_01", "user_02"],
            "start": "2026-04-03T10:00",
            "end": "2026-04-03T11:00",
        })
        assert result["status"] == "created"


class TestCalculatorTool:
    def test_compute_basic(self):
        calc = CalculatorTool()
        result = calc.execute("compute", {"expression": "2 + 3 * 4"})
        assert result["result"] == 14

    def test_compute_math_functions(self):
        calc = CalculatorTool()
        result = calc.execute("compute", {"expression": "sqrt(16)"})
        assert result["result"] == 4.0

    def test_group_sum(self):
        calc = CalculatorTool()
        result = calc.execute("group_sum", {
            "data": [{"cat": "A", "val": 10}, {"cat": "A", "val": 20}, {"cat": "B", "val": 5}],
            "group_by": "cat",
            "aggregate": "val",
        })
        assert result["result"] == {"A": 30.0, "B": 5.0}

    def test_date_diff(self):
        calc = CalculatorTool()
        result = calc.execute("date_diff", {"date1": "2026-01-01", "date2": "2026-01-31"})
        assert result["days"] == 30

    def test_missing_expression(self):
        calc = CalculatorTool()
        result = calc.execute("compute", {})
        assert "error" in result


class TestValidatorTool:
    def test_valid_email(self):
        v = ValidatorTool()
        result = v.execute("validate", {"data": {"to": "a@b.com", "subject": "S", "body": "B"}, "schema_name": "email_format"})
        assert result["valid"] is True

    def test_invalid_email_missing_field(self):
        v = ValidatorTool()
        result = v.execute("validate", {"data": {"to": "a@b.com"}, "schema_name": "email_format"})
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_unknown_schema(self):
        v = ValidatorTool()
        result = v.execute("validate", {"data": {}, "schema_name": "nonexistent"})
        assert "error" in result


# =====================================================================
# Calculator Security Tests
# =====================================================================


class TestCalculatorSecurity:
    def test_blocks_dunder_class(self):
        calc = CalculatorTool()
        result = calc.execute("compute", {"expression": "().__class__.__bases__[0].__subclasses__()"})
        assert "error" in result
        assert "forbidden" in result["error"].lower()

    def test_blocks_dunder_builtins(self):
        calc = CalculatorTool()
        result = calc.execute("compute", {"expression": "''.__class__.__mro__"})
        assert "error" in result

    def test_blocks_import(self):
        calc = CalculatorTool()
        result = calc.execute("compute", {"expression": "__import__('os').system('ls')"})
        assert "error" in result

    def test_blocks_exec(self):
        calc = CalculatorTool()
        result = calc.execute("compute", {"expression": "exec('print(1)')"})
        assert "error" in result

    def test_blocks_getattr(self):
        calc = CalculatorTool()
        result = calc.execute("compute", {"expression": "getattr(int, 'real')"})
        assert "error" in result

    def test_allows_safe_math(self):
        calc = CalculatorTool()
        result = calc.execute("compute", {"expression": "abs(-5) + round(3.7) + min(1, 2)"})
        assert result["result"] == 10


# =====================================================================
# Grader Determinism Tests
# =====================================================================


class TestGraderDeterminism:
    def test_easy_same_history_same_score(self):
        grader = Grader()
        history = [
            {"action": {"tool_name": "database", "method": "query", "parameters": {"sql": "SELECT * FROM employees WHERE hire_date > '2026-03-01'"}}, "result": {"rows": [], "row_count": 4}},
            {"action": {"tool_name": "email", "method": "send", "parameters": {"to": "carol.johnson@acme.com", "subject": "Welcome Carol Johnson", "body": "Welcome to Engineering"}}, "result": {"status": "sent"}},
        ]
        r1 = grader.grade("easy", history)
        r2 = grader.grade("easy", history)
        assert r1["score"] == r2["score"]
        assert r1["breakdown"] == r2["breakdown"]

    def test_hard_same_history_same_score(self):
        grader = Grader()
        history = [
            {"action": {"tool_name": "database", "method": "query", "parameters": {"sql": "SELECT * FROM projects WHERE name = 'Project Alpha'"}}, "result": {"rows": []}},
        ]
        r1 = grader.grade("hard", history)
        r2 = grader.grade("hard", history)
        assert r1["score"] == r2["score"]

    def test_scores_strictly_between_0_and_1(self):
        grader = Grader()
        for task_id in ["easy", "medium", "hard"]:
            result = grader.grade(task_id, [])
            assert 0.0 < result["score"] < 1.0, f"Score must be strictly in (0, 1), got {result['score']} for {task_id}"

            result = grader.grade(task_id, [{"action": {"tool_name": "database", "method": "query", "parameters": {}}, "result": {}}])
            assert 0.0 < result["score"] < 1.0, f"Score must be strictly in (0, 1), got {result['score']} for {task_id}"


# =====================================================================
# Grader find_free_slots Error Handling Detection
# =====================================================================


class TestGraderFindFreeSlotsDetection:
    def test_detects_unavailable_users_in_find_free_slots(self):
        grader = Grader()
        history = [
            {"action": {"tool_name": "database", "method": "query", "parameters": {"sql": "SELECT * FROM projects WHERE name = 'Project Alpha'"}}, "result": {"rows": [{"name": "Project Alpha"}]}},
            {"action": {"tool_name": "database", "method": "query", "parameters": {"sql": "SELECT * FROM employees WHERE department = 'Engineering'"}}, "result": {"rows": []}},
            {"action": {"tool_name": "calendar", "method": "find_free_slots", "parameters": {"users": ["user_01", "user_02", "user_03"], "date_range": {"start": "2026-04-01", "end": "2026-04-07"}}}, "result": {"slots": [{"start": "2026-04-03T10:00", "end": "2026-04-03T11:00"}], "unavailable_users": ["user_03"], "warning": "Could not check availability for: user_03"}},
            {"action": {"tool_name": "calendar", "method": "create_event", "parameters": {"title": "Q2 Review", "attendees": ["user_01", "user_02"], "start": "2026-04-03T10:00", "end": "2026-04-03T11:00"}}, "result": {"event_id": "evt_123", "status": "created"}},
            {"action": {"tool_name": "filestore", "method": "read", "parameters": {"path": "projects/q1-review.md"}}, "result": {"content": "Q1 Review accomplishments v2.0"}},
            {"action": {"tool_name": "filestore", "method": "write", "parameters": {"path": "meetings/q2-review-agenda.md", "content": "Q2 Agenda based on q1 accomplishments"}}, "result": {"status": "written"}},
            {"action": {"tool_name": "email", "method": "send", "parameters": {"to": "team@acme.com", "subject": "Q2 Review Meeting", "body": "Please see the agenda"}}, "result": {"status": "sent"}},
        ]
        tool_states = {
            "email_outbox": [{"to": "team@acme.com", "subject": "Q2 Review Meeting", "body": "Please see the agenda"}],
            "files": {"meetings/q2-review-agenda.md": "Q2 Agenda based on q1 accomplishments"},
        }
        result = grader.grade("hard", history, tool_states)
        assert result["breakdown"]["error_handling"] >= 0.7, f"error_handling should be >= 0.7, got {result['breakdown']['error_handling']}"


# =====================================================================
# Reward Formula Tests
# =====================================================================


class TestRewardFormula:
    def _make_env(self, task_id="easy"):
        env = ToolOrchestrationEnvironment()
        env.reset(task_id=task_id)
        return env

    def test_reward_starts_near_zero(self):
        env = self._make_env()
        assert env.state.total_reward <= 0.02

    def test_reward_increases_with_correct_steps(self):
        env = self._make_env()
        obs = env.step(ToolOrchestrationAction(
            tool_name="database", method="query",
            parameters={"sql": "SELECT * FROM employees WHERE hire_date > '2026-03-01'"},
        ))
        assert obs.reward > 0.0

    def test_perfect_steps_bad_content_gives_low_reward(self):
        """Right tools in right order but empty/wrong parameters should score ~0.3, not ~0.9."""
        env = self._make_env()
        # Step through optimal sequence with minimal params
        env.step(ToolOrchestrationAction(tool_name="database", method="query", parameters={"sql": "SELECT 1"}))
        env.step(ToolOrchestrationAction(tool_name="email", method="send", parameters={"to": "nobody@x.com", "subject": "x", "body": "x"}))
        env.step(ToolOrchestrationAction(tool_name="email", method="send", parameters={"to": "nobody2@x.com", "subject": "x", "body": "x"}))
        env.step(ToolOrchestrationAction(tool_name="email", method="send", parameters={"to": "nobody3@x.com", "subject": "x", "body": "x"}))
        obs = env.step(ToolOrchestrationAction(tool_name="email", method="send", parameters={"to": "nobody4@x.com", "subject": "x", "body": "x"}))

        # With bad content, grader score should be low → reward should be well below 0.9
        assert obs.done, "Episode should be done after completing tool sequence"
        assert obs.reward < 0.6, f"Bad content should give reward < 0.6, got {obs.reward}"

    def test_perfect_episode_gives_high_reward(self):
        """Perfect sequence + perfect content should give reward close to 1.0."""
        env = self._make_env()
        env.step(ToolOrchestrationAction(
            tool_name="database", method="query",
            parameters={"sql": "SELECT * FROM employees WHERE hire_date > '2026-03-01'"},
        ))
        env.step(ToolOrchestrationAction(
            tool_name="email", method="send",
            parameters={"to": "carol.johnson@acme.com", "subject": "Welcome to the team, Carol Johnson!", "body": "Welcome to the Engineering department."},
        ))
        env.step(ToolOrchestrationAction(
            tool_name="email", method="send",
            parameters={"to": "david.kim@acme.com", "subject": "Welcome to the team, David Kim!", "body": "Welcome to the Engineering department."},
        ))
        env.step(ToolOrchestrationAction(
            tool_name="email", method="send",
            parameters={"to": "hannah.davis@acme.com", "subject": "Welcome to the team, Hannah Davis!", "body": "Welcome to the Marketing department."},
        ))
        obs = env.step(ToolOrchestrationAction(
            tool_name="email", method="send",
            parameters={"to": "lisa.tanaka@acme.com", "subject": "Welcome to the team, Lisa Tanaka!", "body": "Welcome to the Finance department."},
        ))
        assert obs.done
        assert obs.reward >= 0.9, f"Perfect episode should give reward >= 0.9, got {obs.reward}"
        assert obs.reward < 1.0, f"Reward must be strictly < 1.0, got {obs.reward}"

    def test_reward_always_strictly_between_0_and_1(self):
        env = self._make_env()
        # Make several bad calls
        for _ in range(10):
            obs = env.step(ToolOrchestrationAction(tool_name="validator", method="validate", parameters={"data": {}, "schema_name": "email_format"}))
            assert 0.0 < obs.reward < 1.0, f"Reward must be strictly in (0, 1), got {obs.reward}"


# =====================================================================
# Edge Case Tests
# =====================================================================


class TestEdgeCases:
    def test_step_after_done(self):
        env = ToolOrchestrationEnvironment()
        env.reset(task_id="easy")
        # Force episode to end by stepping max_steps times
        for _ in range(10):
            env.step(ToolOrchestrationAction(tool_name="validator", method="validate", parameters={"data": {}, "schema_name": "email_format"}))
        # Now try stepping after done
        obs = env.step(ToolOrchestrationAction(tool_name="database", method="query", parameters={"sql": "SELECT 1"}))
        assert obs.done is True
        assert "error" in obs.tool_response

    def test_step_without_reset(self):
        env = ToolOrchestrationEnvironment()
        obs = env.step(ToolOrchestrationAction(tool_name="database", method="query", parameters={"sql": "SELECT 1"}))
        assert obs.done is True
        assert "error" in obs.tool_response

    def test_invalid_tool_name(self):
        env = ToolOrchestrationEnvironment()
        env.reset(task_id="easy")
        obs = env.step(ToolOrchestrationAction(tool_name="nonexistent", method="foo", parameters={}))
        assert "error" in obs.tool_response
        assert "nonexistent" in obs.tool_response["error"]

    def test_invalid_tool_recorded_in_history(self):
        """Issue 5: invalid tool calls must appear in episode history."""
        env = ToolOrchestrationEnvironment()
        env.reset(task_id="easy")
        env.step(ToolOrchestrationAction(tool_name="fake_tool", method="bar", parameters={}))
        assert env.state.step_count == 1
        # History should have 1 entry matching the step count
        assert len(env._episode_history) == 1
        assert env._episode_history[0]["action"]["tool_name"] == "fake_tool"

    def test_reset_without_task_id_picks_random(self):
        env = ToolOrchestrationEnvironment()
        obs = env.reset()
        assert obs.task_description != ""
        assert env.state.task_id in ["easy", "medium", "hard"]

    def test_reset_clears_state(self):
        env = ToolOrchestrationEnvironment()
        env.reset(task_id="easy")
        env.step(ToolOrchestrationAction(tool_name="database", method="query", parameters={"sql": "SELECT 1"}))
        env.reset(task_id="medium")
        assert env.state.step_count == 0
        assert env.state.task_id == "medium"
        assert env.state.total_reward <= 0.02


# =====================================================================
# Completion Heuristic Tests
# =====================================================================


class TestCompletionHeuristic:
    def test_one_email_does_not_complete_easy(self):
        """Easy task needs 4 emails, not 1."""
        env = ToolOrchestrationEnvironment()
        env.reset(task_id="easy")
        env.step(ToolOrchestrationAction(tool_name="database", method="query", parameters={"sql": "SELECT * FROM employees WHERE hire_date > '2026-03-01'"}))
        obs = env.step(ToolOrchestrationAction(tool_name="email", method="send", parameters={"to": "carol.johnson@acme.com", "subject": "Welcome", "body": "Hi"}))
        assert obs.done is False, "1 email should not complete the easy task (needs 4)"

    def test_completion_triggers_at_correct_count(self):
        env = ToolOrchestrationEnvironment()
        env.reset(task_id="easy")
        env.step(ToolOrchestrationAction(tool_name="database", method="query", parameters={"sql": "SELECT 1"}))
        for i in range(3):
            obs = env.step(ToolOrchestrationAction(tool_name="email", method="send", parameters={"to": f"user{i}@x.com", "subject": "S", "body": "B"}))
            assert obs.done is False
        # 4th email completes: 1 database + 4 emails matches optimal
        obs = env.step(ToolOrchestrationAction(tool_name="email", method="send", parameters={"to": "user3@x.com", "subject": "S", "body": "B"}))
        assert obs.done is True
