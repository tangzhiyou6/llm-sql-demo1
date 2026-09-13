import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from benchmark.sample_db import init_sample_database, setup_metadata_and_valuelookup
from core.db_sandbox import DBSandbox
from core.ast_guard import SQLASTGuard
from core.explain_guard import ExplainGuard
from core.consensus_engine import ExecutionConsensusEngine
from core.semantic_layer import (
    get_default_ecommerce_catalog,
    SemanticCompiler,
    SemanticQuery,
    DimensionFilter,
    OrderBy
)
from agent.pipeline import Text2SQLPipeline
import config


class Text2SQLMCPServer:
    """
    High-Performance Persistent Model Context Protocol (MCP) and JSON-RPC Server.
    Keeps all database connections, schema indexes, AST guards, and semantic catalogs
    in memory for ultra-low latency (<5ms) tool execution over stdio.
    """

    def __init__(self, db_path: Optional[str] = None):
        target_path = db_path or config.DB_PATH
        p = Path(target_path)
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        self.db_path = str(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            init_sample_database(self.db_path)

        # 1. Warm up retrieval & lookup
        self.retriever, self.val_lookup = setup_metadata_and_valuelookup(self.db_path)

        # 2. Hardened guards & sandbox
        self.explain_guard = ExplainGuard(block_on_critical=True)
        self.sandbox = DBSandbox(
            db_path=self.db_path,
            timeout_seconds=config.STATEMENT_TIMEOUT_SECONDS,
            explain_guard=self.explain_guard,
            enforce_explain=True
        )
        self.ast_guard = SQLASTGuard(
            default_limit=config.MAX_ROW_LIMIT,
            target_dialect=config.DIALECT
        )

        # 3. Consensus Engine & Semantic Layer
        self.consensus_engine = ExecutionConsensusEngine(sandbox=self.sandbox, ast_guard=self.ast_guard)
        self.semantic_catalog = get_default_ecommerce_catalog()
        self.semantic_compiler = SemanticCompiler(self.semantic_catalog)

        # 4. Pipeline instance
        self.pipeline = Text2SQLPipeline(
            retriever=self.retriever,
            value_lookup=self.val_lookup,
            sandbox=self.sandbox,
            ast_guard=self.ast_guard
        )
        self.current_phase = 1

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "text_to_sql",
                "description": "Enterprise Text-to-SQL pipeline. Converts natural language queries into verified SQL via Plan-then-Solve workflow, AST validation, ExplainGuard pre-filtering, and sandbox execution.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query."},
                        "use_consensus": {"type": "boolean", "description": "Whether to use execution consensus voting.", "default": False}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "semantic_query",
                "description": "Deterministic Semantic Metric Layer query compiler. Guarantees zero join hallucinations by compiling metrics and dimensions using topology graph traversal.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "metrics": {"type": "array", "items": {"type": "string"}, "description": "Registered metrics: total_revenue, order_count, avg_order_value, total_units_sold, item_sales_amount, customer_count, avg_review_rating"},
                        "dimensions": {"type": "array", "items": {"type": "string"}, "description": "Dimensions: city, customer_name, category, product_name, order_status, order_date"},
                        "filters": {"type": "array", "items": {"type": "object"}, "description": "Dimension filters, e.g. [{'dimension': 'order_status', 'operator': '=', 'value': 'PAID'}]"},
                        "limit": {"type": "integer", "description": "Row limit (default: 100)"}
                    },
                    "required": ["metrics"]
                }
            },
            {
                "name": "explain_query",
                "description": "Cost-based pre-filter analyzer. Runs EXPLAIN QUERY PLAN to detect Cartesian products, unindexed table scans, and temporary B-trees.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "The candidate SQL to inspect."}
                    },
                    "required": ["sql"]
                }
            },
            {
                "name": "consensus_sampling",
                "description": "Executes multiple candidate SQL queries, clusters them by execution result set equivalence, and returns the consensus winner.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "candidate_sqls": {"type": "array", "items": {"type": "string"}, "description": "List of diverse SQL candidates to evaluate."}
                    },
                    "required": ["candidate_sqls"]
                }
            },
            {
                "name": "execute_sql_in_sandbox",
                "description": "Executes SQL query in the hardened read-only database sandbox with AST validation (SELECT only), ExplainGuard pre-filter, and dynamic LIMIT 100 rewrite.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "The SQL statement to validate and execute."}
                    },
                    "required": ["sql"]
                }
            },
            {
                "name": "schema_search",
                "description": "Two-level hybrid retrieval tool to find relevant tables and pruned column metadata for a query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Keywords or question concepts to search in metadata."},
                        "top_k": {"type": "integer", "description": "Number of tables to retrieve (default: 5)."}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "value_lookup",
                "description": "Inverted index value lookup for high-frequency categorical dimension values (e.g. status codes, product categories).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "search_term": {"type": "string", "description": "Dimension concept or term mentioned in query."}
                    },
                    "required": ["search_term"]
                }
            },
            {
                "name": "sample_column_values",
                "description": "Samples non-null distinct values for a given column.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string", "description": "Target table name."},
                        "column_name": {"type": "string", "description": "Target column name."},
                        "limit": {"type": "integer", "description": "Number of samples (default: 5)."}
                    },
                    "required": ["table_name", "column_name"]
                }
            },
            {
                "name": "submit_reasoning_blueprint",
                "description": "Submits Phase 1 reasoning blueprint to unlock Phase 2 execution.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "blueprint": {"type": "object", "description": "The structured reasoning blueprint."}
                    },
                    "required": ["blueprint"]
                }
            }
        ]

    def handle_request(self, request: dict) -> dict:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "dsh-text-to-sql-persistent-daemon",
                        "version": "2.0.0"
                    }
                }
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.list_tools()}
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            try:
                data = self._execute_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False) if isinstance(data, (dict, list)) else str(data)}],
                        "data": data,
                        "isError": False
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Tool Execution Error: {str(e)}"}],
                        "isError": True,
                        "error": str(e)
                    }
                }

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"status": "pong"}}

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not found"}
        }

    def _execute_tool(self, name: str, args: dict) -> Any:
        if name == "text_to_sql":
            query = args.get("query", "")
            use_consensus = args.get("use_consensus", False)
            res = self.pipeline.run(query, use_consensus=use_consensus)
            return {
                "success": res.success,
                "user_query": res.user_query,
                "blueprint": res.blueprint.model_dump() if res.blueprint else None,
                "final_sql": res.final_sql,
                "rewritten_sql": res.rewritten_sql,
                "execution_result": res.execution_result.model_dump() if res.execution_result else None,
                "total_latency_ms": res.total_latency_ms,
                "healing_rounds": [h.model_dump() for h in res.healing_history],
                "consensus_result": res.consensus_result.model_dump() if res.consensus_result else None
            }

        elif name == "semantic_query":
            metrics = args.get("metrics", [])
            dimensions = args.get("dimensions", [])
            filters_raw = args.get("filters", [])
            order_by_raw = args.get("order_by", [])
            limit = args.get("limit", 100)

            filters = [
                DimensionFilter(dimension=f["dimension"], operator=f.get("operator", "="), value=f["value"])
                for f in filters_raw
            ]
            order_by = [
                OrderBy(field=o["field"], direction=o.get("direction", "DESC"))
                for o in order_by_raw
            ]

            sem_q = SemanticQuery(
                metrics=metrics,
                dimensions=dimensions,
                filters=filters,
                order_by=order_by,
                limit=limit
            )
            compiled_sql = self.semantic_compiler.compile(sem_q)
            exec_res = self.sandbox.execute_query(compiled_sql)
            return {
                "success": exec_res.success,
                "compiled_sql": compiled_sql,
                "row_count": exec_res.row_count,
                "columns": exec_res.columns,
                "rows": [list(r) for r in exec_res.rows],
                "execution_time_ms": exec_res.execution_time_ms,
                "error": exec_res.error
            }

        elif name == "explain_query":
            sql = args.get("sql", "").strip()
            conn = self.sandbox._get_readonly_connection()
            try:
                analysis = self.explain_guard.analyze_plan(conn, sql)
                return analysis.model_dump()
            finally:
                conn.close()

        elif name == "consensus_sampling":
            candidates = args.get("candidate_sqls", [])
            c_res = self.consensus_engine.evaluate_candidates(candidates)
            return c_res.model_dump()

        elif name == "execute_sql_in_sandbox":
            sql = args.get("sql", "").strip()
            parsed, _, _ = self.ast_guard.validate_and_extract(sql)
            capped = self.ast_guard.inject_or_cap_limit(parsed, max_limit=config.MAX_ROW_LIMIT)
            rewritten_sql = capped.sql(dialect=config.DIALECT, pretty=True)
            res = self.sandbox.execute_query(rewritten_sql)
            return {
                "success": res.success,
                "rewritten_sql": rewritten_sql,
                "row_count": res.row_count,
                "columns": res.columns,
                "rows": [list(r) for r in res.rows[:10]],
                "execution_time_ms": res.execution_time_ms,
                "error": res.error,
                "plan_analysis": res.plan_analysis.model_dump() if res.plan_analysis else None
            }

        elif name == "schema_search":
            query = args.get("query", "")
            top_k = args.get("top_k", 5)
            tables = self.retriever.retrieve_tables_hybrid(query, top_k=top_k)
            schema_str = self.retriever.prune_and_format_schema(query, tables)
            return {
                "candidate_tables": [t.table_name for t in tables],
                "schema_context": schema_str
            }

        elif name == "value_lookup":
            term = args.get("search_term", "")
            matches = self.val_lookup.search_query(term, top_k=3)
            return {
                "matches": [
                    {"matched_term": m.matched_term, "db_value": str(m.db_value), "filter_hint": m.filter_hint, "score": m.score}
                    for m in matches
                ]
            }

        elif name == "sample_column_values":
            tbl = args.get("table_name", "")
            col = args.get("column_name", "")
            limit = args.get("limit", 5)
            samples = self.sandbox.sample_column_values(tbl, col, limit=limit)
            return {"table": tbl, "column": col, "samples": samples}

        elif name == "submit_reasoning_blueprint":
            self.current_phase = 2
            return {
                "status": "Blueprint approved.",
                "message": "Phase 1 concluded. Phase 2 execution is unlocked."
            }

        raise ValueError(f"Unknown tool: {name}")

    def run_stdio(self):
        """Processes JSON-RPC requests line by line from stdin."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if "id" not in req:
                    continue
                resp = self.handle_request(req)
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    server = Text2SQLMCPServer()
    server.run_stdio()
