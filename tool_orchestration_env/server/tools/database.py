"""Simulated in-memory SQL database tool.

Uses sqlite3 in-memory to provide a realistic database with 3 tables:
- employees (15 rows): id, name, email, department, hire_date, salary
- invoices (47 rows): id, vendor, amount, category, date, status
- projects (5 rows): id, name, team_lead_email, department, status, deadline

Only SELECT and INSERT are allowed. All data is deterministic and
hardcoded — graders depend on exact values.
"""

import sqlite3
from typing import Any, Dict


class DatabaseTool:
    def __init__(self) -> None:
        self._conn: sqlite3.Connection | None = None

    def reset(self, task_id: str = "") -> None:
        if self._conn:
            self._conn.close()
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._seed()

    def _seed(self) -> None:
        assert self._conn is not None
        c = self._conn.cursor()

        # --- employees (15 rows) ---
        c.execute("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                department TEXT NOT NULL,
                hire_date TEXT NOT NULL,
                salary REAL NOT NULL
            )
        """)

        employees = [
            # Engineering (4 people, 2 are new hires after 2026-03-01)
            (1, "Alice Chen", "alice.chen@acme.com", "Engineering", "2024-06-15", 125000),
            (2, "Bob Martinez", "bob.martinez@acme.com", "Engineering", "2025-01-10", 118000),
            (3, "Carol Johnson", "carol.johnson@acme.com", "Engineering", "2026-03-10", 105000),
            (4, "David Kim", "david.kim@acme.com", "Engineering", "2026-03-15", 110000),
            # Marketing (4 people, 1 new hire)
            (5, "Eve Wilson", "eve.wilson@acme.com", "Marketing", "2024-03-01", 95000),
            (6, "Frank Brown", "frank.brown@acme.com", "Marketing", "2025-07-20", 92000),
            (7, "Grace Lee", "grace.lee@acme.com", "Marketing", "2025-11-05", 88000),
            (8, "Hannah Davis", "hannah.davis@acme.com", "Marketing", "2026-03-20", 90000),
            # Finance (4 people, 1 new hire)
            (9, "Ivan Petrov", "ivan.petrov@acme.com", "Finance", "2023-09-12", 115000),
            (10, "Julia Santos", "julia.santos@acme.com", "Finance", "2024-08-30", 108000),
            (11, "Kevin O'Brien", "kevin.obrien@acme.com", "Finance", "2025-05-18", 102000),
            (12, "Lisa Tanaka", "lisa.tanaka@acme.com", "Finance", "2026-03-05", 98000),
            # HR (3 people, no new hires)
            (13, "Mike Rivera", "mike.rivera@acme.com", "HR", "2023-02-14", 88000),
            (14, "Nina Johansson", "nina.johansson@acme.com", "HR", "2024-11-01", 85000),
            (15, "Oscar Patel", "oscar.patel@acme.com", "HR", "2025-08-22", 82000),
        ]
        c.executemany(
            "INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)", employees
        )

        # --- invoices (47 rows) ---
        # Category totals: Software=$8940, Travel=$3200, Office Supplies=$1450, Marketing=$5600
        # Grand total: $19,190
        c.execute("""
            CREATE TABLE invoices (
                id INTEGER PRIMARY KEY,
                vendor TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                date TEXT NOT NULL,
                status TEXT NOT NULL
            )
        """)

        invoices = []
        inv_id = 1

        # Software: $8940 total (12 invoices)
        software_amounts = [1200, 850, 750, 600, 950, 800, 700, 550, 640, 500, 1200, 200]
        software_vendors = [
            "CloudStack Inc", "DevTools Pro", "SaaS Platform Co", "CodeReview Ltd",
            "CI/CD Systems", "MonitorAll Inc", "LogAnalytics", "SecureCode Co",
            "APIGateway Inc", "TestRunner Pro", "DocGen Ltd", "LintMaster"
        ]
        assert sum(software_amounts) == 8940
        for i, (amt, vendor) in enumerate(zip(software_amounts, software_vendors)):
            day = (i % 28) + 1
            invoices.append((inv_id, vendor, amt, "Software", f"2026-03-{day:02d}", "paid"))
            inv_id += 1

        # Travel: $3200 total (10 invoices)
        travel_amounts = [450, 380, 520, 280, 350, 300, 260, 220, 240, 200]
        travel_vendors = [
            "AirTravel Co", "Hotel Express", "CarRental Inc", "MealCard Co",
            "TaxiService", "ConferenceTravel", "AirTravel Co", "Hotel Express",
            "TrainBooker", "ParkingPlus"
        ]
        assert sum(travel_amounts) == 3200
        for i, (amt, vendor) in enumerate(zip(travel_amounts, travel_vendors)):
            day = (i % 28) + 1
            invoices.append((inv_id, vendor, amt, "Travel", f"2026-03-{day:02d}", "paid"))
            inv_id += 1

        # Office Supplies: $1450 total (12 invoices)
        office_amounts = [180, 150, 120, 200, 95, 130, 110, 85, 75, 100, 105, 100]
        office_vendors = [
            "OfficeMax", "PaperCo", "InkJet Supply", "FurniturePro",
            "CleanDesk Co", "OfficeMax", "TechAccessories", "PrinterWorld",
            "StationeryPlus", "OfficeMax", "ErgonomicPro", "SupplyDepot"
        ]
        assert sum(office_amounts) == 1450
        for i, (amt, vendor) in enumerate(zip(office_amounts, office_vendors)):
            day = (i % 28) + 1
            invoices.append((inv_id, vendor, amt, "Office Supplies", f"2026-03-{day:02d}", "paid"))
            inv_id += 1

        # Marketing: $5600 total (13 invoices)
        marketing_amounts = [800, 650, 550, 500, 450, 400, 380, 350, 320, 300, 280, 350, 270]
        marketing_vendors = [
            "AdPlatform Inc", "SocialMedia Pro", "ContentCreators", "SEO Experts",
            "EmailCampaign Co", "DesignStudio", "PrintMedia Ltd", "EventSponsors",
            "InfluencerNet", "VideoProduction", "BrandAnalytics", "WebAds Inc",
            "MarketResearch Co"
        ]
        assert sum(marketing_amounts) == 5600
        for i, (amt, vendor) in enumerate(zip(marketing_amounts, marketing_vendors)):
            day = (i % 28) + 1
            invoices.append((inv_id, vendor, amt, "Marketing", f"2026-03-{day:02d}", "paid"))
            inv_id += 1

        assert len(invoices) == 47
        assert inv_id == 48  # next id would be 48
        c.executemany("INSERT INTO invoices VALUES (?, ?, ?, ?, ?, ?)", invoices)

        # --- projects (5 rows) ---
        c.execute("""
            CREATE TABLE projects (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                team_lead_email TEXT NOT NULL,
                department TEXT NOT NULL,
                status TEXT NOT NULL,
                deadline TEXT NOT NULL
            )
        """)

        projects = [
            (1, "Project Alpha", "alice.chen@acme.com", "Engineering", "active", "2026-06-30"),
            (2, "Brand Refresh", "eve.wilson@acme.com", "Marketing", "active", "2026-05-15"),
            (3, "Budget Automation", "ivan.petrov@acme.com", "Finance", "planning", "2026-07-01"),
            (4, "Employee Portal", "mike.rivera@acme.com", "HR", "active", "2026-04-30"),
            (5, "Cloud Migration", "bob.martinez@acme.com", "Engineering", "planning", "2026-08-15"),
        ]
        c.executemany("INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?)", projects)

        self._conn.commit()

    def execute(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._conn is None:
            return {"error": "Database not initialized. Call reset() first."}

        if method == "query":
            return self._query(params)
        elif method == "insert":
            return self._insert(params)
        else:
            return {"error": f"Unknown method '{method}'. Available: query, insert"}

    def _query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sql = params.get("sql", "")
        if not sql:
            return {"error": "Missing required parameter 'sql'"}

        # Only allow SELECT statements
        stripped = sql.strip().upper()
        if not stripped.startswith("SELECT"):
            return {"error": "Only SELECT queries are allowed. Use insert() for INSERT operations."}

        # Block dangerous keywords
        for keyword in ["DROP", "DELETE", "UPDATE", "ALTER", "CREATE"]:
            if keyword in stripped:
                return {"error": f"Forbidden keyword '{keyword}' in query."}

        try:
            cursor = self._conn.cursor()
            cursor.execute(sql)
            rows = [dict(row) for row in cursor.fetchall()]
            return {"rows": rows, "row_count": len(rows)}
        except sqlite3.Error as e:
            return {"error": f"SQL error: {str(e)}"}

    def _insert(self, params: Dict[str, Any]) -> Dict[str, Any]:
        table = params.get("table", "")
        data = params.get("data", {})
        if not table:
            return {"error": "Missing required parameter 'table'"}
        if not data:
            return {"error": "Missing required parameter 'data'"}

        # Validate table name exists
        if table not in ("employees", "invoices", "projects"):
            return {"error": f"Unknown table '{table}'. Available: employees, invoices, projects"}

        try:
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?"] * len(data))
            sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            cursor = self._conn.cursor()
            cursor.execute(sql, list(data.values()))
            self._conn.commit()
            return {"inserted": True, "id": cursor.lastrowid}
        except sqlite3.Error as e:
            return {"error": f"Insert error: {str(e)}"}

    def describe(self) -> Dict[str, Any]:
        return {
            "name": "database",
            "description": "SQL database with employees, invoices, and projects tables",
            "methods": {
                "query": {
                    "description": "Execute a SELECT query",
                    "parameters": {"sql": "SQL SELECT statement"},
                    "returns": {"rows": "list of row dicts", "row_count": "number of rows"},
                },
                "insert": {
                    "description": "Insert a row into a table",
                    "parameters": {"table": "table name", "data": "dict of column:value"},
                    "returns": {"inserted": "bool", "id": "row id"},
                },
            },
        }
