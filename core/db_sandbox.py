import sqlite3
import time
import threading
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field


from core.explain_guard import ExplainGuard, PlanAnalysisResult, RiskLevel


class ExecutionResult(BaseModel):
    """Execution result returned by DBSandbox."""
    success: bool
    columns: List[str] = Field(default_factory=list)
    rows: List[Tuple[Any, ...]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    is_empty: bool = False
    plan_analysis: Optional[PlanAnalysisResult] = None


class TableColumnInfo(BaseModel):
    name: str
    data_type: str
    not_null: bool
    primary_key: bool


class TableSchema(BaseModel):
    table_name: str
    columns: List[TableColumnInfo]
    foreign_keys: List[Dict[str, str]] = Field(default_factory=list)
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list)


class DBSandbox:
    """
    Hardened Database Sandbox.
    Features:
      1. Mandatory Read-Only Connection (URI mode=ro)
      2. Session-level execution timeout (timer interrupt)
      3. Read-only Schema exploration tools (sample_column_values, schema_search)
      4. Safe query execution with metrics (latency, row count, error capturing)
    """

    def __init__(
        self,
        db_path: str,
        timeout_seconds: float = 5.0,
        explain_guard: Optional[ExplainGuard] = None,
        enforce_explain: bool = True
    ):
        self.db_path = Path(db_path).resolve()
        self.timeout_seconds = timeout_seconds
        self.explain_guard = explain_guard or ExplainGuard()
        self.enforce_explain = enforce_explain

    def _get_readonly_connection(self) -> sqlite3.Connection:
        """Opens a strictly read-only SQLite connection using URI mode=ro."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        # uri=True with mode=ro forbids any write operations at the SQLite engine level
        conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def execute_query(self, sql_str: str, check_explain: bool = True) -> ExecutionResult:
        """
        Executes SQL query in sandbox with strict timeout, read-only isolation, and ExplainGuard pre-filtering.
        """
        start_time = time.perf_counter()
        conn = None
        timer = None
        try:
            conn = self._get_readonly_connection()

            # Cost-based pre-filter: inspect query plan before running execution
            plan_analysis = None
            if self.enforce_explain and check_explain:
                plan_analysis = self.explain_guard.analyze_plan(conn, sql_str)
                if not plan_analysis.is_safe:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    return ExecutionResult(
                        success=False,
                        execution_time_ms=round(elapsed_ms, 2),
                        error=f"Cost Pre-Filter Intercepted: {plan_analysis.error_message}",
                        plan_analysis=plan_analysis
                    )

            # Set up hard execution timeout
            timed_out = [False]

            def interrupt_conn():
                timed_out[0] = True
                try:
                    conn.interrupt()
                except Exception:
                    pass

            timer = threading.Timer(self.timeout_seconds, interrupt_conn)
            timer.start()

            cursor = conn.cursor()
            cursor.execute(sql_str)
            raw_rows = cursor.fetchall()

            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = [tuple(r) for r in raw_rows]
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ExecutionResult(
                success=True,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=round(elapsed_ms, 2),
                is_empty=(len(rows) == 0),
                plan_analysis=plan_analysis
            )
        except sqlite3.OperationalError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = str(e)
            if "interrupted" in err_msg.lower() or "timeout" in err_msg.lower():
                return ExecutionResult(
                    success=False,
                    execution_time_ms=round(elapsed_ms, 2),
                    error=f"Query Execution Timeout: Exceeded hard limit of {self.timeout_seconds}s."
                )
            if "readonly database" in err_msg.lower():
                return ExecutionResult(
                    success=False,
                    execution_time_ms=round(elapsed_ms, 2),
                    error="Security Exception: Write operations rejected by read-only database connection."
                )
            return ExecutionResult(
                success=False,
                execution_time_ms=round(elapsed_ms, 2),
                error=f"SQLite OperationalError: {err_msg}"
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                success=False,
                execution_time_ms=round(elapsed_ms, 2),
                error=f"Database Execution Failure: {type(e).__name__}: {str(e)}"
            )
        finally:
            if timer:
                timer.cancel()
            if conn:
                conn.close()

    def get_all_tables(self) -> List[str]:
        """Returns list of all user table names."""
        conn = self._get_readonly_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

    def get_table_schema(self, table_name: str) -> TableSchema:
        """Retrieves comprehensive table schema including columns and foreign keys."""
        conn = self._get_readonly_connection()
        try:
            cur = conn.cursor()
            # Columns: cid, name, type, notnull, dflt_value, pk
            cur.execute(f"PRAGMA table_info({table_name});")
            col_rows = cur.fetchall()
            columns = [
                TableColumnInfo(
                    name=r["name"],
                    data_type=r["type"],
                    not_null=bool(r["notnull"]),
                    primary_key=bool(r["pk"])
                )
                for r in col_rows
            ]

            # Foreign keys: id, seq, table, from, to, on_update, on_delete, match
            cur.execute(f"PRAGMA foreign_key_list({table_name});")
            fk_rows = cur.fetchall()
            foreign_keys = [
                {"from_column": r["from"], "to_table": r["table"], "to_column": r["to"]}
                for r in fk_rows
            ]

            # Sample 2 rows
            cur.execute(f"SELECT * FROM {table_name} LIMIT 2;")
            sample_data = [dict(r) for r in cur.fetchall()]

            return TableSchema(
                table_name=table_name,
                columns=columns,
                foreign_keys=foreign_keys,
                sample_rows=sample_data
            )
        finally:
            conn.close()

    def sample_column_values(self, table_name: str, column_name: str, limit: int = 5) -> List[Any]:
        """Exploration tool: samples non-null distinct values for a given column."""
        conn = self._get_readonly_connection()
        try:
            cur = conn.cursor()
            # Sanitize table and column name against alphanumeric/underscore
            if not table_name.isidentifier() or not column_name.isidentifier():
                raise ValueError(f"Invalid identifier: {table_name}.{column_name}")

            query = f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT {int(limit)}"
            cur.execute(query)
            return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()
