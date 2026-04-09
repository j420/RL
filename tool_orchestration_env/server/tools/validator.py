"""Simulated JSON schema validation tool.

Validates data dicts against predefined schemas: expense_report,
meeting_invite, email_format. Pure in-memory, no external deps.
"""

from typing import Any, Dict, List


# Schema definitions
_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "expense_report": {
        "required_fields": ["categories", "totals", "grand_total"],
        "field_types": {
            "categories": list,
            "totals": dict,
            "grand_total": (int, float),
        },
    },
    "meeting_invite": {
        "required_fields": ["title", "attendees", "time"],
        "field_types": {
            "title": str,
            "attendees": list,
            "time": (str, dict),
        },
    },
    "email_format": {
        "required_fields": ["to", "subject", "body"],
        "field_types": {
            "to": str,
            "subject": str,
            "body": str,
        },
    },
    "employee_data": {
        "required_fields": ["name", "email", "department"],
        "field_types": {
            "name": str,
            "email": str,
            "department": str,
        },
    },
    "invoice_data": {
        "required_fields": ["vendor", "amount", "category"],
        "field_types": {
            "vendor": str,
            "amount": (int, float),
            "category": str,
        },
    },
    "calendar_event": {
        "required_fields": ["title", "attendees", "start", "end"],
        "field_types": {
            "title": str,
            "attendees": list,
            "start": str,
            "end": str,
        },
    },
}


class ValidatorTool:
    def __init__(self) -> None:
        pass

    def reset(self, task_id: str = "") -> None:
        # Stateless tool — nothing to reset
        pass

    def execute(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "validate":
            return self._validate(params)
        else:
            return {"error": f"Unknown method '{method}'. Available: validate"}

    def _validate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        data = params.get("data", {})
        schema_name = params.get("schema_name", "")

        if not schema_name:
            return {"error": "Missing required parameter 'schema_name'"}
        if schema_name not in _SCHEMAS:
            return {
                "error": f"Unknown schema '{schema_name}'. Available: {', '.join(_SCHEMAS.keys())}"
            }

        schema = _SCHEMAS[schema_name]
        errors: List[str] = []

        # Check required fields
        for field in schema["required_fields"]:
            if field not in data:
                errors.append(f"Missing required field: '{field}'")
            else:
                # Check type
                expected_type = schema["field_types"].get(field)
                if expected_type and not isinstance(data[field], expected_type):
                    errors.append(
                        f"Field '{field}' must be {expected_type}, got {type(data[field]).__name__}"
                    )

        return {"valid": len(errors) == 0, "errors": errors}

    def describe(self) -> Dict[str, Any]:
        return {
            "name": "validator",
            "description": "JSON schema validator for structured data",
            "methods": {
                "validate": {
                    "description": "Validate data against a named schema",
                    "parameters": {
                        "data": "dict to validate",
                        "schema_name": "one of: expense_report, meeting_invite, email_format",
                    },
                    "returns": {"valid": "bool", "errors": "list of error strings"},
                },
            },
            "available_schemas": list(_SCHEMAS.keys()),
        }
