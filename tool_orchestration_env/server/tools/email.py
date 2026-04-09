"""Simulated in-memory email tool.

Provides inbox (seeded with 5 emails) and outbox for sending.
All operations are in-memory — no external services.
"""

import re
import uuid
from typing import Any, Dict, List

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


class EmailTool:
    def __init__(self) -> None:
        self._inbox: List[Dict[str, Any]] = []
        self._outbox: List[Dict[str, Any]] = []

    def reset(self, task_id: str = "") -> None:
        self._outbox = []
        self._inbox = [
            {
                "id": "msg_001",
                "from": "ceo@acme.com",
                "to": "all@acme.com",
                "subject": "Q2 Kickoff",
                "body": "Team, let's make Q2 our best quarter yet. Key priorities: mobile app launch, Series B close, and hiring.",
                "date": "2026-03-28",
                "read": True,
            },
            {
                "id": "msg_002",
                "from": "hr@acme.com",
                "to": "managers@acme.com",
                "subject": "New Hire Onboarding Checklist",
                "body": "Please ensure all new hires complete their onboarding tasks within the first week.",
                "date": "2026-03-29",
                "read": False,
            },
            {
                "id": "msg_003",
                "from": "finance@acme.com",
                "to": "all@acme.com",
                "subject": "Expense Report Deadline",
                "body": "All March expense reports must be submitted by April 5th.",
                "date": "2026-03-30",
                "read": False,
            },
            {
                "id": "msg_004",
                "from": "alice.chen@acme.com",
                "to": "engineering@acme.com",
                "subject": "Project Alpha Q2 Planning",
                "body": "Let's schedule a Q2 review meeting for Project Alpha. I'll coordinate calendars.",
                "date": "2026-03-31",
                "read": True,
            },
            {
                "id": "msg_005",
                "from": "eve.wilson@acme.com",
                "to": "marketing@acme.com",
                "subject": "Brand Refresh Update",
                "body": "The new brand guidelines are ready for review. Please check the shared drive.",
                "date": "2026-04-01",
                "read": False,
            },
        ]

    def execute(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "send":
            return self._send(params)
        elif method == "search":
            return self._search(params)
        elif method == "list_inbox":
            return self._list_inbox()
        else:
            return {"error": f"Unknown method '{method}'. Available: send, search, list_inbox"}

    def _send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        to = params.get("to", "")
        subject = params.get("subject", "")
        body = params.get("body", "")

        if not to:
            return {"error": "Missing required parameter 'to'"}
        if not _EMAIL_RE.match(to):
            return {"error": f"Invalid email address format: '{to}'"}
        if not subject:
            return {"error": "Missing required parameter 'subject'"}
        if not body:
            return {"error": "Missing required parameter 'body'"}

        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        email = {
            "id": message_id,
            "to": to,
            "subject": subject,
            "body": body,
            "attachment": params.get("attachment"),
            "date": "2026-04-01",
            "status": "sent",
        }
        self._outbox.append(email)

        return {"status": "sent", "message_id": message_id}

    def _search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "").lower()
        if not query:
            return {"error": "Missing required parameter 'query'"}

        results = []
        for msg in self._inbox + self._outbox:
            searchable = " ".join([
                str(msg.get("from", "")),
                str(msg.get("to", "")),
                str(msg.get("subject", "")),
                str(msg.get("body", "")),
            ]).lower()
            if query in searchable:
                results.append(msg)

        return {"results": results, "count": len(results)}

    def _list_inbox(self) -> Dict[str, Any]:
        return {"emails": self._inbox}

    def get_outbox(self) -> List[Dict[str, Any]]:
        """Expose outbox for grader inspection."""
        return list(self._outbox)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": "email",
            "description": "Email system with inbox and outbox",
            "methods": {
                "send": {
                    "description": "Send an email",
                    "parameters": {
                        "to": "recipient email address",
                        "subject": "email subject",
                        "body": "email body text",
                        "attachment": "(optional) file path to attach",
                    },
                    "returns": {"status": "sent", "message_id": "unique id"},
                },
                "search": {
                    "description": "Search inbox and outbox",
                    "parameters": {"query": "search string"},
                    "returns": {"results": "matching emails", "count": "number of matches"},
                },
                "list_inbox": {
                    "description": "List all inbox emails",
                    "parameters": {},
                    "returns": {"emails": "list of email dicts"},
                },
            },
        }
