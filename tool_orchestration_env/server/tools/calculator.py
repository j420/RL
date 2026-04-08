"""Simulated calculator tool.

Provides safe math evaluation, group aggregation, and date difference.
No external dependencies — uses only Python stdlib.
"""

import math
from datetime import datetime
from typing import Any, Dict, List


# Whitelist of safe math functions/constants for compute()
_SAFE_NAMES: Dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "pow": pow,
    # math module
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


class CalculatorTool:
    def __init__(self) -> None:
        pass

    def reset(self, task_id: str = "") -> None:
        # Stateless tool — nothing to reset
        pass

    def execute(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "compute":
            return self._compute(params)
        elif method == "group_sum":
            return self._group_sum(params)
        elif method == "date_diff":
            return self._date_diff(params)
        else:
            return {"error": f"Unknown method '{method}'. Available: compute, group_sum, date_diff"}

    def _compute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        expression = params.get("expression", "")
        if not expression:
            return {"error": "Missing required parameter 'expression'"}

        try:
            # Evaluate with only safe builtins
            result = eval(expression, {"__builtins__": {}}, _SAFE_NAMES)
            return {"result": result}
        except Exception as e:
            return {"error": f"Computation error: {str(e)}"}

    def _group_sum(self, params: Dict[str, Any]) -> Dict[str, Any]:
        data = params.get("data", [])
        group_by = params.get("group_by", "")
        aggregate = params.get("aggregate", "")

        if not data:
            return {"error": "Missing required parameter 'data' (list of dicts)"}
        if not group_by:
            return {"error": "Missing required parameter 'group_by'"}
        if not aggregate:
            return {"error": "Missing required parameter 'aggregate'"}

        if not isinstance(data, list):
            return {"error": "'data' must be a list of dicts"}

        groups: Dict[str, float] = {}
        for item in data:
            if not isinstance(item, dict):
                return {"error": f"Each item in 'data' must be a dict, got {type(item).__name__}"}
            key = str(item.get(group_by, "unknown"))
            val = item.get(aggregate, 0)
            try:
                val = float(val)
            except (TypeError, ValueError):
                return {"error": f"Cannot convert '{val}' to number for aggregation"}
            groups[key] = groups.get(key, 0.0) + val

        return {"result": groups}

    def _date_diff(self, params: Dict[str, Any]) -> Dict[str, Any]:
        date1 = params.get("date1", "")
        date2 = params.get("date2", "")

        if not date1 or not date2:
            return {"error": "Missing required parameters 'date1' and 'date2' (YYYY-MM-DD)"}

        try:
            d1 = datetime.strptime(date1, "%Y-%m-%d")
            d2 = datetime.strptime(date2, "%Y-%m-%d")
            return {"days": abs((d2 - d1).days)}
        except ValueError as e:
            return {"error": f"Date parse error: {str(e)}. Use YYYY-MM-DD format."}

    def describe(self) -> Dict[str, Any]:
        return {
            "name": "calculator",
            "description": "Math evaluator with grouping and date operations",
            "methods": {
                "compute": {
                    "description": "Evaluate a mathematical expression safely",
                    "parameters": {"expression": "math expression string"},
                    "returns": {"result": "computed value"},
                },
                "group_sum": {
                    "description": "Group a list of dicts by a key and sum another key",
                    "parameters": {
                        "data": "list of dicts",
                        "group_by": "key to group by",
                        "aggregate": "key to sum",
                    },
                    "returns": {"result": "dict of group -> sum"},
                },
                "date_diff": {
                    "description": "Calculate the difference in days between two dates",
                    "parameters": {"date1": "YYYY-MM-DD", "date2": "YYYY-MM-DD"},
                    "returns": {"days": "integer number of days"},
                },
            },
        }
