import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from benchmark.sample_db import init_sample_database, setup_metadata_and_valuelookup
from core.db_sandbox import DBSandbox
from core.ast_guard import SQLASTGuard
from core.deepseek_harness import DEEPSEEK_DB_TOOLS
import config


class Text2SQLMCPServer:
    """
    Standard Model Context Protocol (MCP) Server for DeepSeek Harness (dsh).
    Communicates via standard input/output (stdio JSON-RPC 2.0).
    Exposes the 5 hardened database tools directly to dsh or any MCP client.
    """

    def __init__(self, db_path: str = config.DB_PATH):
        if not Path(db_path).exists():
            init_sample_database(db_path)

        self.retriever, self.val_lookup = setup_metadata_and_valuelookup(db_path)
        self.sandbox = DBSandbox(db_path=db_path, timeout_seconds=config.STATEMENT_TIMEOUT_SECONDS)
        self.ast_guard = SQLASTGuard(default_limit=config.MAX_ROW_LIMIT, target_dialect=config.DIALECT)
        self.current_phase = 1

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
                        "name": "dsh-text-to-sql-mcp",
                        "version": "1.0.0"
                    }
                }
            }

        elif method == "tools/list":
            # Convert DeepSeek tool format to MCP tools format
            mcp_tools = []
            for dt in DEEPSEEK_DB_TOOLS:
                fn = dt["function"]
                mcp_tools.append({
                    "name": fn["name"],
                    "description": fn["description"],
                    "inputSchema": fn["parameters"]
                })
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": mcp_tools}
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            try:
                res_text = self._execute_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": res_text}],
                        "isError": False
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Tool Execution Error: {str(e)}"}],
                        "isError": True
                    }
                }

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not found"}
        }

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "schema_search":
            query = args.get("query", "")
            top_k = args.get("top_k", 5)
            tables = self.retriever.retrieve_tables_hybrid(query, top_k=top_k)
            schema_str = self.retriever.prune_and_format_schema(query, tables)
            return json.dumps({
                "candidate_tables": [t.table_name for t in tables],
                "schema_context": schema_str
            }, ensure_ascii=False)

        elif name == "sample_column_values":
            tbl = args.get("table_name", "")
            col = args.get("column_name", "")
            limit = args.get("limit", 5)
            samples = self.sandbox.sample_column_values(tbl, col, limit=limit)
            return json.dumps({"table": tbl, "column": col, "samples": samples}, ensure_ascii=False)

        elif name == "value_lookup":
            term = args.get("search_term", "")
            matches = self.val_lookup.search_query(term, top_k=3)
            return json.dumps({
                "matches": [
                    {"matched_term": m.matched_term, "db_value": str(m.db_value), "filter_hint": m.filter_hint, "score": m.score}
                    for m in matches
                ]
            }, ensure_ascii=False)

        elif name == "submit_reasoning_blueprint":
            self.current_phase = 2
            return json.dumps({
                "status": "Blueprint approved.",
                "message": "Phase 1 concluded. Phase 2 (SQL Generation & execute_sql_in_sandbox) is unlocked."
            }, ensure_ascii=False)

        elif name == "execute_sql_in_sandbox":
            sql = args.get("sql", "").strip()
            # AST Validation & Rewrite
            parsed, _, _ = self.ast_guard.validate_and_extract(sql)
            capped = self.ast_guard.inject_or_cap_limit(parsed, max_limit=config.MAX_ROW_LIMIT)
            rewritten_sql = capped.sql(dialect=config.DIALECT, pretty=True)

            res = self.sandbox.execute_query(rewritten_sql)
            if res.success:
                return json.dumps({
                    "status": "SUCCESS",
                    "rewritten_sql": rewritten_sql,
                    "row_count": res.row_count,
                    "columns": res.columns,
                    "rows": [list(r) for r in res.rows[:10]],
                    "execution_time_ms": res.execution_time_ms
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "status": "ERROR",
                    "error": res.error,
                    "rewritten_sql": rewritten_sql
                }, ensure_ascii=False)

        raise ValueError(f"Unknown tool: {name}")

    def run_stdio(self):
        """Processes JSON-RPC requests line by line from stdin."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                # If notification (no id), ignore response
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
