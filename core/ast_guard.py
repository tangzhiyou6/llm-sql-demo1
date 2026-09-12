from typing import List, Dict, Set, Optional, Tuple
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class ASTValidationError(Exception):
    """Raised when SQL AST validation detects safety or hallucination issues."""
    def __init__(self, message: str, error_type: str = "VALIDATION_ERROR", location: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.location = location


class SQLASTGuard:
    """
    SQL AST Guardian powered by sqlglot.
    Provides:
      1. Structural hallucination & safety interception (rejecting non-SELECT, unknown tables/columns)
      2. Dynamic AST rewrite (forcing LIMIT capping)
      3. Dialect transpilation (e.g. standard postgres -> target sqlite/mysql/clickhouse)
      4. Structured error diagnostics for LLM self-healing reflection
    """

    def __init__(self, default_limit: int = 100, target_dialect: str = "sqlite"):
        self.default_limit = default_limit
        self.target_dialect = target_dialect

    def parse_sql(self, sql_str: str, read_dialect: Optional[str] = None) -> exp.Expression:
        """Parse raw SQL string into sqlglot AST, capturing syntax errors with precise locations."""
        clean_sql = sql_str.strip().rstrip(";")
        try:
            parsed = sqlglot.parse_one(clean_sql, read=read_dialect or self.target_dialect)
            return parsed
        except ParseError as e:
            err_details = []
            for err in getattr(e, "errors", []):
                line = err.get("line")
                col = err.get("col")
                highlight = err.get("highlight")
                description = err.get("description", str(e))
                err_details.append(
                    f"Line {line}, Column {col}: {description}\nContext: {highlight}"
                )
            formatted_msg = "\n".join(err_details) if err_details else str(e)
            raise ASTValidationError(
                message=f"SQL Syntax Parse Error:\n{formatted_msg}",
                error_type="SYNTAX_ERROR"
            ) from e

    def validate_and_extract(
        self,
        sql_str: str,
        allowed_tables: Optional[Set[str]] = None,
        allowed_columns_map: Optional[Dict[str, Set[str]]] = None,
        read_dialect: Optional[str] = None,
    ) -> Tuple[exp.Expression, List[str], List[str]]:
        """
        Interception for high-risk operations, unrecognized tables, and phantom columns.
        Returns:
            (parsed_ast, list_of_referenced_tables, list_of_referenced_columns)
        """
        parsed = self.parse_sql(sql_str, read_dialect=read_dialect)

        # 1. Reject non-query operations
        # Allowed roots: exp.Select, exp.Union, exp.Subquery
        if not isinstance(parsed, (exp.Select, exp.Union)):
            raise ASTValidationError(
                f"Security violation: Only SELECT/UNION queries are permitted. Found AST node: {type(parsed).__name__}",
                error_type="DISALLOWED_OPERATION"
            )

        # 2. Extract Common Table Expressions (CTEs) so they aren't mistaken for real DB tables
        cte_names = set()
        with_clause = parsed.args.get("with")
        if with_clause:
            for cte in with_clause.expressions:
                cte_names.add(cte.alias_or_name.lower())

        # 3. Extract and check referenced tables
        referenced_tables = []
        for table_node in parsed.find_all(exp.Table):
            table_name = table_node.name.lower()
            if table_name and table_name not in cte_names:
                referenced_tables.append(table_name)
                if allowed_tables is not None:
                    allowed_lower = {t.lower() for t in allowed_tables}
                    if table_name not in allowed_lower:
                        raise ASTValidationError(
                            f"Hallucination error: Table '{table_name}' does not exist in schema. "
                            f"Allowed tables: {sorted(list(allowed_lower))}",
                            error_type="UNKNOWN_TABLE"
                        )

        # 4. Extract referenced columns and validate if allowed_columns_map is provided
        referenced_columns = []
        for col_node in parsed.find_all(exp.Column):
            col_name = col_node.name.lower()
            table_qualifier = col_node.table.lower() if col_node.table else None
            referenced_columns.append(f"{table_qualifier + '.' if table_qualifier else ''}{col_name}")

            # If column map is available and column is table-qualified, verify it belongs to that table
            if allowed_columns_map and table_qualifier and table_qualifier not in cte_names:
                table_cols = allowed_columns_map.get(table_qualifier.lower())
                if table_cols:
                    lower_table_cols = {c.lower() for c in table_cols}
                    if col_name != "*" and col_name not in lower_table_cols:
                        raise ASTValidationError(
                            f"Hallucination error: Column '{col_name}' does not exist on table '{table_qualifier}'. "
                            f"Available columns: {sorted(list(lower_table_cols))}",
                            error_type="UNKNOWN_COLUMN"
                        )

        return parsed, referenced_tables, referenced_columns

    def inject_or_cap_limit(self, parsed_ast: exp.Expression, max_limit: Optional[int] = None) -> exp.Expression:
        """
        Rewrites AST to ensure outermost query contains a LIMIT clause <= max_limit.
        Prevents full table dumps and frontend memory crashes.
        """
        limit_val = max_limit or self.default_limit
        if not isinstance(parsed_ast, (exp.Select, exp.Union)):
            return parsed_ast

        # Check existing limit
        existing_limit = parsed_ast.args.get("limit")
        if existing_limit is None:
            parsed_ast = parsed_ast.limit(limit_val)
        else:
            # If limit expression is a literal number, clamp it
            expr = existing_limit.expression
            if isinstance(expr, exp.Literal) and expr.is_int:
                cur_val = int(expr.this)
                if cur_val > limit_val:
                    existing_limit.set("expression", exp.Literal.number(limit_val))
        return parsed_ast

    def transpile(
        self,
        sql_str: str,
        read_dialect: Optional[str] = None,
        write_dialect: Optional[str] = None
    ) -> str:
        """Transpile SQL dialect (e.g. postgres -> sqlite/clickhouse)."""
        target = write_dialect or self.target_dialect
        transpiled = sqlglot.transpile(sql_str, read=read_dialect, write=target, pretty=True)
        if transpiled:
            return transpiled[0]
        return sql_str

    def format_error_for_reflection(self, error: Exception, raw_sql: str) -> str:
        """Formats structured AST error diagnostics for model self-healing prompt."""
        if isinstance(error, ASTValidationError):
            return (
                f"### [AST Validation Error]\n"
                f"Error Type: {error.error_type}\n"
                f"Issue: {error.message}\n"
                f"Offending SQL:\n```sql\n{raw_sql}\n```\n"
                f"Correction Directive: Fix the specific table/column or statement type identified above."
            )
        return (
            f"### [SQL Parsing Failure]\n"
            f"Error: {str(error)}\n"
            f"Offending SQL:\n```sql\n{raw_sql}\n```\n"
            f"Correction Directive: Ensure valid SQL syntax adhering strictly to the database dialect."
        )
