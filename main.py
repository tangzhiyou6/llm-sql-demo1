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
    parser.add_argument("--consensus", "-c", action="store_true", help="Enable Execution Consensus clustering across candidate queries (P1)")
    parser.add_argument("--explain", type=str, help="Run Cost-Based ExplainGuard pre-filter analysis on given SQL (P2)")
    parser.add_argument("--semantic", action="store_true", help="Run query through Semantic Metric Layer Compiler (P3)")
    parser.add_argument("--metrics", type=str, default="total_revenue", help="Comma-separated metric names for semantic query")
    parser.add_argument("--dimensions", type=str, default="city", help="Comma-separated dimension names for semantic query")
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
    from core.explain_guard import ExplainGuard
    from core.semantic_layer import get_default_ecommerce_catalog, SemanticCompiler, SemanticQuery, DimensionFilter

    explain_guard = ExplainGuard(block_on_critical=True)
    retriever, val_lookup = setup_metadata_and_valuelookup(db_path)
    sandbox = DBSandbox(db_path=db_path, timeout_seconds=config.STATEMENT_TIMEOUT_SECONDS, explain_guard=explain_guard)
    ast_guard = SQLASTGuard(default_limit=config.MAX_ROW_LIMIT, target_dialect=config.DIALECT)

    # P2 Explain CLI
    if args.explain:
        conn = sandbox._get_readonly_connection()
        try:
            analysis = explain_guard.analyze_plan(conn, args.explain)
            print("\n" + "=" * 70)
            print(f"  EXPLAIN QUERY PLAN Analysis (P2 ExplainGuard)")
            print("=" * 70)
            print(f"  • SQL        : {args.explain}")
            print(f"  • Risk Level : {analysis.risk_level.value}")
            print(f"  • Safe       : {analysis.is_safe}")
            if analysis.has_cartesian_product:
                print(f"  • Cartesian  : DETECTED on {analysis.cartesian_tables}")
            if analysis.error_message:
                print(f"  • Error      : {analysis.error_message}")
            if analysis.warnings:
                print(f"  • Warnings   : {analysis.warnings}")
            print("\n  • Plan Nodes :")
            for node in analysis.plan_nodes:
                print(f"    [{node.id}] (parent={node.parent}) {node.detail}")
            print("=" * 70 + "\n")
            return
        finally:
            conn.close()

    # P3 Semantic Query CLI
    if args.semantic:
        catalog = get_default_ecommerce_catalog()
        compiler = SemanticCompiler(catalog)
        m_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
        d_list = [d.strip() for d in args.dimensions.split(",") if d.strip()]
        q = SemanticQuery(metrics=m_list, dimensions=d_list, limit=10)
        compiled_sql = compiler.compile(q)
        print("\n" + "=" * 70)
        print("  Semantic Metric Layer Compilation (P3)")
        print("=" * 70)
        print(f"  • Requested Metrics   : {m_list}")
        print(f"  • Requested Dimensions: {d_list}")
        print(f"\n  • Compiled SQL:\n{compiled_sql}\n")
        exec_res = sandbox.execute_query(compiled_sql)
        print(f"  • Execution Result    : {'SUCCESS' if exec_res.success else 'FAILED'} ({exec_res.row_count} rows)")
        print(f"  • Columns             : {exec_res.columns}")
        for r in exec_res.rows[:5]:
            print(f"    {r}")
        print("=" * 70 + "\n")
        return

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
        result = pipeline.run(args.query, use_consensus=args.consensus)
        display_pipeline_result(result, val_lookup, args.query)
        if result.consensus_result:
            cr = result.consensus_result
            print(f"[Consensus Details]: Total Candidates = {cr.total_candidates} | Valid = {cr.valid_candidates} | Confidence = {cr.confidence * 100:.1f}%")
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
