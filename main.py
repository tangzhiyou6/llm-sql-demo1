import argparse
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from benchmark.sample_db import init_sample_database, setup_metadata_and_valuelookup
from core.db_sandbox import DBSandbox
from core.ast_guard import SQLASTGuard
from agent.pipeline import Text2SQLPipeline
from benchmark.run_benchmark import run_benchmark_suite
import config


def print_banner():
    print("""
========================================================================
       Enterprise Text-to-SQL Architecture (DeepSeek + sqlglot)
    Plan-then-Solve | AST Guardian | Hybrid Search | Read-Only Sandbox
========================================================================
""")


def display_pipeline_result(result, val_lookup, user_query):
    print("\n" + "=" * 75)
    print(f"  User Query: \"{user_query}\"")
    print("=" * 75)

    # 1. Value Lookup Matches
    v_matches = val_lookup.search_query(user_query, top_k=3)
    print("\n[Step 1: Value-Lookup Alignment]")
    if v_matches:
        for m in v_matches:
            print(f"  • Matched: '{m.matched_term}' -> Filter: `{m.filter_hint}` (Confidence: {m.score:.2f})")
    else:
        print("  • No explicit categorical dimension values detected.")

    # 2. Phase 1 Blueprint
    print("\n[Step 2: Phase 1 Reasoning Blueprint]")
    if result.blueprint:
        bp = result.blueprint
        print(f"  • Intent       : {bp.user_intent}")
        print(f"  • Tables       : {bp.selected_tables}")
        print(f"  • Join Paths   : {bp.join_paths or 'None (Single Table)'}")
        print(f"  • Filters      : {bp.filter_conditions or 'None'}")
        print(f"  • Aggregations : {bp.aggregations or 'None'}")
        print(f"  • CoT Summary  : {bp.reasoning_summary}")

    # 3. Phase 2 Generation & AST Rewrite
    print("\n[Step 3: Phase 2 Code Generation & AST Guard]")
    print(f"  • Initial SQL  :\n    {result.final_sql}")
    print(f"  • AST Rewritten (LIMIT capped) :\n    {result.rewritten_sql}")

    # 4. Self-Healing Trace
    if result.healing_history:
        print(f"\n[Step 4: Self-Healing Trace ({len(result.healing_history)} round(s))]")
        for h in result.healing_history:
            print(f"  - Round {h.round_number} [{h.failure_type}]: {h.error_message}")
            print(f"    Model Used : {h.model_used}")
    else:
        print("\n[Step 4: Self-Healing Trace]")
        print("  • Zero errors encountered: Passed AST guard and execution on first attempt.")

    # 5. Database Sandbox Execution
    print("\n[Step 5: Database Sandbox Execution]")
    if result.execution_result:
        er = result.execution_result
        if er.success:
            print(f"  • Status : SUCCESS (Returned {er.row_count} row(s) in {er.execution_time_ms} ms)")
            print(f"  • Columns: {er.columns}")
            print("  • Rows   :")
            for row in er.rows[:5]:
                print(f"    {row}")
            if er.row_count > 5:
                print(f"    ... and {er.row_count - 5} more row(s)")
        else:
            print(f"  • Status : FAILED ({er.error})")

    print(f"\n[Summary]: Total Pipeline Latency = {result.total_latency_ms:.2f} ms | Status = {'PASSED' if result.success else 'FAILED'}")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Enterprise Text-to-SQL Execution Engine")
    parser.add_argument("--query", "-q", type=str, help="Single natural language query to process")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run the golden evaluation benchmark suite")
    parser.add_argument("--init-db", action="store_true", help="Initialize and re-seed the sample database")
    parser.add_argument("--harness", action="store_true", help="Run via DeepSeek Agent Tool-Calling Harness")
    args = parser.parse_args()

    db_path = config.DB_PATH

    # Initialize DB if requested or missing
    if args.init_db or not Path(db_path).exists():
        print(f"[*] Initializing sample e-commerce database at {db_path}...")
        init_sample_database(db_path)
        print("[+] Database initialized successfully.")
        if args.init_db and not args.query and not args.benchmark:
            return

    if args.benchmark:
        run_benchmark_suite(db_path=db_path)
        return

    # Setup core services
    retriever, val_lookup = setup_metadata_and_valuelookup(db_path)
    sandbox = DBSandbox(db_path=db_path, timeout_seconds=config.STATEMENT_TIMEOUT_SECONDS)
    ast_guard = SQLASTGuard(default_limit=config.MAX_ROW_LIMIT, target_dialect=config.DIALECT)

    if args.harness:
        from core.deepseek_harness import DeepSeekHarness
        from harness_runner import display_harness_result
        harness = DeepSeekHarness(retriever=retriever, value_lookup=val_lookup, sandbox=sandbox, ast_guard=ast_guard)
        if args.query:
            result = harness.run(args.query)
            display_harness_result(result, args.query)
            return
        from harness_runner import run_harness_cli
        run_harness_cli()
        return

    pipeline = Text2SQLPipeline(
        retriever=retriever,
        value_lookup=val_lookup,
        sandbox=sandbox,
        ast_guard=ast_guard
    )

    if args.query:
        result = pipeline.run(args.query)
        display_pipeline_result(result, val_lookup, args.query)
        return

    # Interactive CLI Mode
    print_banner()
    print("Interactive Mode. Type your question (or 'exit' to quit, 'benchmark' to run tests):\n")
    sample_queries = [
        "查询所有处于PAID状态的订单ID和金额",
        "查找购买过100%纯果汁的客户姓名和订单日期",
        "统计每个城市注册的客户总数量",
        "找出累计消费金额大于200元的高价值客户"
    ]
    print("Example Queries:")
    for sq in sample_queries:
        print(f"  - {sq}")
    print()

    while True:
        try:
            user_input = input("Text2SQL > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break
            if user_input.lower() == "benchmark":
                run_benchmark_suite(db_path=db_path)
                continue

            result = pipeline.run(user_input)
            display_pipeline_result(result, val_lookup, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
