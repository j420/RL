"""Simulated in-memory calendar tool.

Manages events for 5 users (user_01 through user_05) over April 1-7, 2026.
A common free slot exists on April 3, 2026 10:00-11:00 for all users.

Error injection: When task_id == "hard", querying user_03's calendar returns
a CalendarServiceTimeout error — testing the agent's error-handling ability.
"""

import uuid
from typing import Any, Dict, List


class CalendarTool:
    def __init__(self) -> None:
        self._events: Dict[str, List[Dict[str, Any]]] = {}
        self._task_id: str = ""

    def reset(self, task_id: str = "") -> None:
        self._task_id = task_id
        self._events = {}
        self._seed()

    def _seed(self) -> None:
        # Create events for 5 users, April 1-7 2026
        # Leave April 3 10:00-11:00 free for ALL users (the target meeting slot)

        self._events["user_01"] = [
            {"id": "evt_01a", "title": "Team Standup", "start": "2026-04-01T09:00", "end": "2026-04-01T09:30"},
            {"id": "evt_01b", "title": "Sprint Planning", "start": "2026-04-01T14:00", "end": "2026-04-01T15:00"},
            {"id": "evt_01c", "title": "Code Review", "start": "2026-04-02T10:00", "end": "2026-04-02T11:00"},
            {"id": "evt_01d", "title": "1:1 with Manager", "start": "2026-04-03T14:00", "end": "2026-04-03T14:30"},
            {"id": "evt_01e", "title": "Design Review", "start": "2026-04-04T09:00", "end": "2026-04-04T10:00"},
            {"id": "evt_01f", "title": "Demo Prep", "start": "2026-04-07T11:00", "end": "2026-04-07T12:00"},
        ]

        self._events["user_02"] = [
            {"id": "evt_02a", "title": "All Hands", "start": "2026-04-01T10:00", "end": "2026-04-01T11:00"},
            {"id": "evt_02b", "title": "Product Sync", "start": "2026-04-02T09:00", "end": "2026-04-02T10:00"},
            {"id": "evt_02c", "title": "Lunch Meeting", "start": "2026-04-02T12:00", "end": "2026-04-02T13:00"},
            {"id": "evt_02d", "title": "Client Call", "start": "2026-04-03T15:00", "end": "2026-04-03T16:00"},
            {"id": "evt_02e", "title": "Retrospective", "start": "2026-04-04T14:00", "end": "2026-04-04T15:00"},
        ]

        self._events["user_03"] = [
            {"id": "evt_03a", "title": "Architecture Review", "start": "2026-04-01T11:00", "end": "2026-04-01T12:00"},
            {"id": "evt_03b", "title": "Mentoring Session", "start": "2026-04-02T14:00", "end": "2026-04-02T15:00"},
            {"id": "evt_03c", "title": "Security Audit", "start": "2026-04-03T09:00", "end": "2026-04-03T09:45"},
            # April 3 10:00-11:00 is free
            {"id": "evt_03d", "title": "Vendor Demo", "start": "2026-04-04T10:00", "end": "2026-04-04T11:00"},
            {"id": "evt_03e", "title": "Team Lunch", "start": "2026-04-07T12:00", "end": "2026-04-07T13:00"},
        ]

        self._events["user_04"] = [
            {"id": "evt_04a", "title": "Budget Review", "start": "2026-04-01T09:00", "end": "2026-04-01T10:00"},
            {"id": "evt_04b", "title": "Strategy Meeting", "start": "2026-04-02T11:00", "end": "2026-04-02T12:00"},
            {"id": "evt_04c", "title": "Training", "start": "2026-04-03T13:00", "end": "2026-04-03T14:00"},
            {"id": "evt_04d", "title": "Performance Review", "start": "2026-04-04T09:00", "end": "2026-04-04T10:00"},
            {"id": "evt_04e", "title": "Quarterly Planning", "start": "2026-04-07T10:00", "end": "2026-04-07T11:30"},
        ]

        self._events["user_05"] = [
            {"id": "evt_05a", "title": "Onboarding", "start": "2026-04-01T09:00", "end": "2026-04-01T11:00"},
            {"id": "evt_05b", "title": "IT Setup", "start": "2026-04-02T09:00", "end": "2026-04-02T10:00"},
            {"id": "evt_05c", "title": "Team Introduction", "start": "2026-04-03T11:00", "end": "2026-04-03T12:00"},
            {"id": "evt_05d", "title": "HR Orientation", "start": "2026-04-04T14:00", "end": "2026-04-04T15:30"},
            {"id": "evt_05e", "title": "Mentor Meeting", "start": "2026-04-07T09:00", "end": "2026-04-07T10:00"},
        ]

    def execute(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "get_events":
            return self._get_events(params)
        elif method == "find_free_slots":
            return self._find_free_slots(params)
        elif method == "create_event":
            return self._create_event(params)
        else:
            return {"error": f"Unknown method '{method}'. Available: get_events, find_free_slots, create_event"}

    def _get_events(self, params: Dict[str, Any]) -> Dict[str, Any]:
        user = params.get("user", "")
        date_range = params.get("date_range", {})

        if not user:
            return {"error": "Missing required parameter 'user'"}

        # Error injection for hard task
        if self._task_id == "hard" and user == "user_03":
            return {
                "error": "CalendarServiceTimeout",
                "message": "Calendar for user_03 unavailable. Retry or skip.",
            }

        if user not in self._events:
            return {"error": f"Unknown user '{user}'. Available: user_01 through user_05"}

        events = self._events[user]

        # Filter by date range if provided
        if date_range:
            start = date_range.get("start", "")
            end = date_range.get("end", "")
            if start and end:
                events = [
                    e for e in events
                    if e["start"][:10] >= start and e["start"][:10] <= end
                ]

        return {"events": events}

    def _find_free_slots(self, params: Dict[str, Any]) -> Dict[str, Any]:
        users = params.get("users", [])
        date_range = params.get("date_range", {})
        duration_minutes = params.get("duration_minutes", 60)

        if not users:
            return {"error": "Missing required parameter 'users' (list of user IDs)"}
        if not date_range:
            return {"error": "Missing required parameter 'date_range' ({start, end})"}

        start_date = date_range.get("start", "2026-04-01")
        end_date = date_range.get("end", "2026-04-07")

        # Error injection for hard task — user_03 unavailable
        unavailable_users = []
        active_users = []
        for user in users:
            if self._task_id == "hard" and user == "user_03":
                unavailable_users.append(user)
            elif user in self._events:
                active_users.append(user)
            else:
                return {"error": f"Unknown user '{user}'"}

        # Find slots where all active users are free
        # Check hourly slots from 09:00-17:00 each day
        slots = []
        from datetime import datetime, timedelta

        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            # Check hourly slots from 09:00 to 16:00 (last 1hr slot ending at 17:00)
            for hour in range(9, 17):
                slot_start = f"{date_str}T{hour:02d}:00"
                slot_end_hour = hour + (duration_minutes // 60)
                slot_end_min = duration_minutes % 60
                if slot_end_hour > 17:
                    continue
                slot_end = f"{date_str}T{slot_end_hour:02d}:{slot_end_min:02d}"

                # Check if all active users are free during this slot
                all_free = True
                for user in active_users:
                    for event in self._events.get(user, []):
                        # Check overlap: event overlaps slot if event_start < slot_end and event_end > slot_start
                        if event["start"] < slot_end and event["end"] > slot_start:
                            all_free = False
                            break
                    if not all_free:
                        break

                if all_free:
                    slots.append({"start": slot_start, "end": slot_end})

            current += timedelta(days=1)

        result: Dict[str, Any] = {"slots": slots}
        if unavailable_users:
            result["unavailable_users"] = unavailable_users
            result["warning"] = f"Could not check availability for: {', '.join(unavailable_users)} (service timeout)"

        return result

    def _create_event(self, params: Dict[str, Any]) -> Dict[str, Any]:
        title = params.get("title", "")
        attendees = params.get("attendees", [])
        start = params.get("start", "")
        end = params.get("end", "")

        if not title:
            return {"error": "Missing required parameter 'title'"}
        if not attendees:
            return {"error": "Missing required parameter 'attendees' (list of user IDs)"}
        if not start or not end:
            return {"error": "Missing required parameters 'start' and 'end' (YYYY-MM-DDTHH:MM)"}

        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        event = {
            "id": event_id,
            "title": title,
            "attendees": attendees,
            "start": start,
            "end": end,
        }

        # Add event to each attendee's calendar
        for user in attendees:
            if user in self._events:
                self._events[user].append(event)

        return {"event_id": event_id, "status": "created", "attendees_count": len(attendees)}

    def describe(self) -> Dict[str, Any]:
        return {
            "name": "calendar",
            "description": "Calendar management for scheduling and event creation",
            "methods": {
                "get_events": {
                    "description": "Get events for a user within an optional date range",
                    "parameters": {
                        "user": "user ID (e.g., 'user_01')",
                        "date_range": "(optional) {start: YYYY-MM-DD, end: YYYY-MM-DD}",
                    },
                    "returns": {"events": "list of event dicts"},
                },
                "find_free_slots": {
                    "description": "Find common free time slots for multiple users",
                    "parameters": {
                        "users": "list of user IDs",
                        "date_range": "{start: YYYY-MM-DD, end: YYYY-MM-DD}",
                        "duration_minutes": "meeting duration (default: 60)",
                    },
                    "returns": {"slots": "list of {start, end}"},
                },
                "create_event": {
                    "description": "Create a calendar event",
                    "parameters": {
                        "title": "event title",
                        "attendees": "list of user IDs",
                        "start": "YYYY-MM-DDTHH:MM",
                        "end": "YYYY-MM-DDTHH:MM",
                    },
                    "returns": {"event_id": "unique id", "status": "created"},
                },
            },
        }
