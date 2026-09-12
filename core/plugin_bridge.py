import sys
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from benchmark.sample_db import init_sample_database, setup_metadata_and_valuelookup
from core.db_sandbox import DBSandbox
from core.ast_guard import SQLASTGuard
from agent.pipeline import Text2SQLPipeline
import config


def get_bridge_services(db_path: str = config.DB_PATH):
    if not Path(db_path).exists():
        init_sample_database(db_path)
    retriever, val_lookup = setup_metadata_and_valuelookup(db_path)
    sandbox = DBSandbox(db_path=db_path, timeout_seconds=config.STATEMENT_TIMEOUT_SECONDS)
    ast_guard = SQLASTGuard(default_limit=config.MAX_ROW_LIMIT, target_dialect=config.DIALECT)
    return retriever, val_lookup, sandbox, ast_guard


def handle_tool_call(tool_name: str, args: dict) -> dict:
    retriever, val_lookup, sandbox, ast_guard = get_bridge_services()

    if tool_name == "text_to_sql":
        query = args.get("query", "")
        pipeline = Text2SQLPipeline(
            retriever=retriever,
            value_lookup=val_lookup,
            sandbox=sandbox,
            ast_guard=ast_guard
        )
        res = pipeline.run(query)
        return {
            "success": res.success,
            "user_query": res.user_query,
            "blueprint": res.blueprint.model_dump() if res.blueprint else None,
            "final_sql": res.final_sql,
            "rewritten_sql": res.rewritten_sql,
            "execution_result": res.execution_result.model_dump() if res.execution_result else None,
            "total_latency_ms": res.total_latency_ms,
            "healing_rounds": [h.model_dump() for h in res.healing_history]
        }

    elif tool_name == "execute_sql_in_sandbox":
        sql = args.get("sql", "").strip()
        parsed, _, _ = ast_guard.validate_and_extract(sql)
        capped = ast_guard.inject_or_cap_limit(parsed, max_limit=config.MAX_ROW_LIMIT)
        rewritten_sql = capped.sql(dialect=config.DIALECT, pretty=True)
        exec_res = sandbox.execute_query(rewritten_sql)
        return {
            "success": exec_res.success,
            "rewritten_sql": rewritten_sql,
            "row_count": exec_res.row_count,
            "columns": exec_res.columns,
            "rows": [list(r) for r in exec_res.rows[:10]],
            "execution_time_ms": exec_res.execution_time_ms,
            "error": exec_res.error
        }

    elif tool_name == "schema_search":
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        candidate_tables = retriever.retrieve_tables_hybrid(query, top_k=top_k)
        pruned_schema = retriever.prune_and_format_schema(query, candidate_tables)
        return {
            "candidate_tables": [t.table_name for t in candidate_tables],
            "schema_context": pruned_schema
        }

    elif tool_name == "value_lookup":
        term = args.get("search_term", "")
        top_k = args.get("top_k", 5)
        matches = val_lookup.search_query(term, top_k=top_k)
        return {
            "matches": [m.model_dump() for m in matches]
        }

    raise ValueError(f"Unknown tool name: {tool_name}")


def main():
    parser = argparse.ArgumentParser(description="Bridge between Node.js dsh plugin and Python engine")
    parser.add_argument("--tool", required=True, help="Tool name to execute")
    parser.add_argument("--args", required=True, help="JSON-encoded arguments")
    parsed_cli = parser.parse_args()

    try:
        call_args = json.loads(parsed_cli.args)
        result = handle_tool_call(parsed_cli.tool, call_args)
        sys.stdout.write(json.dumps(result, ensure_ascii=False))
        sys.stdout.flush()
    except Exception as e:
        err_res = {"success": False, "error": str(e)}
        sys.stdout.write(json.dumps(err_res, ensure_ascii=False))
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
