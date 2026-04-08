"""Simulated in-memory file storage tool.

Provides a virtual filesystem backed by a dict. Seeded with project
documents and templates that tasks reference.
"""

from typing import Any, Dict


class FileStoreTool:
    def __init__(self) -> None:
        self._files: Dict[str, str] = {}

    def reset(self, task_id: str = "") -> None:
        self._files = {
            "projects/q1-review.md": (
                "# Q1 Review\n"
                "\n"
                "## Accomplishments\n"
                "- Shipped v2.0 of the platform\n"
                "- Reduced API latency by 40%\n"
                "- Onboarded 3 new enterprise clients\n"
                "\n"
                "## Challenges\n"
                "- Hiring delays in backend team\n"
                "- Q1 revenue target missed by 8%\n"
                "\n"
                "## Q2 Priorities\n"
                "- Launch mobile app beta\n"
                "- Hire 4 engineers\n"
                "- Close Series B pipeline"
            ),
            "templates/email-template.txt": (
                "Subject: {{subject}}\n"
                "To: {{to}}\n"
                "\n"
                "Dear {{name}},\n"
                "\n"
                "{{body}}\n"
                "\n"
                "Best regards,\n"
                "{{sender}}"
            ),
        }

    def execute(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "read":
            return self._read(params)
        elif method == "write":
            return self._write(params)
        elif method == "list":
            return self._list(params)
        else:
            return {"error": f"Unknown method '{method}'. Available: read, write, list"}

    def _read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = params.get("path", "")
        if not path:
            return {"error": "Missing required parameter 'path'"}

        content = self._files.get(path)
        if content is None:
            return {"error": f"File not found: '{path}'"}

        return {"content": content, "size_bytes": len(content.encode("utf-8"))}

    def _write(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path = params.get("path", "")
        content = params.get("content", "")
        if not path:
            return {"error": "Missing required parameter 'path'"}
        if not content:
            return {"error": "Missing required parameter 'content'"}

        self._files[path] = content
        return {"status": "written", "path": path, "size_bytes": len(content.encode("utf-8"))}

    def _list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        directory = params.get("directory", "")
        if directory and not directory.endswith("/"):
            directory += "/"

        files = []
        for path in sorted(self._files.keys()):
            if not directory or path.startswith(directory):
                files.append({"path": path, "size_bytes": len(self._files[path].encode("utf-8"))})

        return {"files": files}

    def get_files(self) -> Dict[str, str]:
        """Expose files for grader inspection."""
        return dict(self._files)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": "filestore",
            "description": "Virtual file storage for reading and writing documents",
            "methods": {
                "read": {
                    "description": "Read a file",
                    "parameters": {"path": "file path"},
                    "returns": {"content": "file content", "size_bytes": "file size"},
                },
                "write": {
                    "description": "Write content to a file",
                    "parameters": {"path": "file path", "content": "file content"},
                    "returns": {"status": "written", "path": "file path", "size_bytes": "file size"},
                },
                "list": {
                    "description": "List files in a directory",
                    "parameters": {"directory": "(optional) directory path to filter"},
                    "returns": {"files": "list of {path, size_bytes}"},
                },
            },
        }
