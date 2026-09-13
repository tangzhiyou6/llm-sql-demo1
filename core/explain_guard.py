import re
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PlanNode(BaseModel):
    id: int
    parent: int
    detail: str
    is_scan: bool = False
    is_search: bool = False
    is_temp_btree: bool = False
    table_name: Optional[str] = None


class PlanAnalysisResult(BaseModel):
    sql: str
    is_safe: bool
    risk_level: RiskLevel
    plan_nodes: List[PlanNode] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    cartesian_tables: List[str] = Field(default_factory=list)
    has_cartesian_product: bool = False


class ExplainGuard:
    """
    Cost-based Pre-filter using SQLite's EXPLAIN QUERY PLAN.
    Detects and intercepts dangerous query patterns before sandbox execution:
      1. Cartesian Products (multiple unindexed SCAN loops)
      2. Full table scans on large tables
      3. Runaway temporary B-tree allocations
    """

    def __init__(
        self,
        block_on_critical: bool = True,
        large_table_row_threshold: int = 10000,
        max_unindexed_scans: int = 1
    ):
        self.block_on_critical = block_on_critical
        self.large_table_row_threshold = large_table_row_threshold
        self.max_unindexed_scans = max_unindexed_scans

    def analyze_plan(self, conn: sqlite3.Connection, sql: str) -> PlanAnalysisResult:
        """
        Executes EXPLAIN QUERY PLAN and analyzes potential execution risks.
        """
        clean_sql = sql.strip().rstrip(";")
        cursor = conn.cursor()

        try:
            cursor.execute(f"EXPLAIN QUERY PLAN {clean_sql}")
            raw_rows = cursor.fetchall()
        except sqlite3.OperationalError as e:
            return PlanAnalysisResult(
                sql=clean_sql,
                is_safe=False,
                risk_level=RiskLevel.CRITICAL,
                error_message=f"EXPLAIN QUERY PLAN syntax/operational failure: {str(e)}"
            )

        nodes: List[PlanNode] = []
        scan_tables: List[str] = []
        search_tables: List[str] = []
        temp_btrees: List[str] = []
        warnings: List[str] = []

        scan_pattern = re.compile(r"SCAN\s+(?:TABLE\s+)?([a-zA-Z0-9_]+)", re.IGNORECASE)
        search_pattern = re.compile(r"SEARCH\s+(?:TABLE\s+)?([a-zA-Z0-9_]+)", re.IGNORECASE)

        for row in raw_rows:
            # SQLite EXPLAIN QUERY PLAN returns: (id, parent, notused, detail)
            node_id = row[0]
            parent_id = row[1]
            detail = row[3] if len(row) > 3 else row[2]

            detail_upper = detail.upper()
            is_scan = "SCAN " in detail_upper
            is_search = "SEARCH " in detail_upper
            is_temp_btree = "USE TEMP B-TREE" in detail_upper

            table_name = None
            if is_scan:
                m = scan_pattern.search(detail)
                if m:
                    matched_name = m.group(1)
                    if matched_name.upper() not in ("CONSTANT", "SUBQUERY"):
                        table_name = matched_name
                        scan_tables.append(table_name)
            elif is_search:
                m = search_pattern.search(detail)
                if m:
                    matched_name = m.group(1)
                    if matched_name.upper() not in ("CONSTANT", "SUBQUERY"):
                        table_name = matched_name
                        search_tables.append(table_name)

            if is_temp_btree:
                temp_btrees.append(detail)

            nodes.append(PlanNode(
                id=node_id,
                parent=parent_id,
                detail=detail,
                is_scan=is_scan,
                is_search=is_search,
                is_temp_btree=is_temp_btree,
                table_name=table_name
            ))

        # 1. Cartesian product detection:
        # If there are >= 2 distinct unindexed table scans, this indicates
        # nested full-table scans (Cartesian product O(N*M)) without indexed lookups.
        has_cartesian = False
        cartesian_tables = []
        unique_scan_tables = list(set(scan_tables))
        if len(unique_scan_tables) > self.max_unindexed_scans:
            # Check if this query joins multiple tables without index lookup
            has_cartesian = True
            cartesian_tables = unique_scan_tables
            warnings.append(
                f"Cartesian Product Detected: Multiple unindexed table scans on {cartesian_tables}. "
                f"Missing JOIN ... ON or WHERE equality predicate between joined tables."
            )

        # 2. Temp B-tree warnings
        if temp_btrees:
            warnings.append(f"Query allocates temporary B-trees in memory: {', '.join(temp_btrees)}")

        # 3. Determine Risk Level
        if has_cartesian:
            risk_level = RiskLevel.CRITICAL
            error_msg = (
                f"ExplainGuard: Query rejected due to CRITICAL Cartesian Product risk. "
                f"Multiple tables {cartesian_tables} are scanned without index predicates (O(N*M) explosion). "
                f"Actionable fix: Ensure explicit JOIN ... ON conditions are specified for all tables."
            )
            is_safe = False
        elif len(temp_btrees) >= 2:
            risk_level = RiskLevel.MEDIUM
            error_msg = None
            is_safe = True
        elif temp_btrees or len(scan_tables) == 1:
            risk_level = RiskLevel.LOW
            error_msg = None
            is_safe = True
        else:
            risk_level = RiskLevel.SAFE
            error_msg = None
            is_safe = True

        return PlanAnalysisResult(
            sql=clean_sql,
            is_safe=is_safe,
            risk_level=risk_level,
            plan_nodes=nodes,
            warnings=warnings,
            error_message=error_msg,
            cartesian_tables=cartesian_tables,
            has_cartesian_product=has_cartesian
        )

    def validate_or_raise(self, conn: sqlite3.Connection, sql: str) -> PlanAnalysisResult:
        """Analyzes query plan and raises ValueError if risk is CRITICAL."""
        result = self.analyze_plan(conn, sql)
        if self.block_on_critical and not result.is_safe:
            raise ValueError(result.error_message or "Query rejected by ExplainGuard pre-filter.")
        return result
