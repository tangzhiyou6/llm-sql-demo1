import argparse
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from benchmark.sample_db import init_sample_database, setup_metadata_and_valuelookup
from core.db_sandbox import DBSandbox
from core.ast_guard import SQLASTGuard
from core.deepseek_harness import DeepSeekHarness, DEEPSEEK_DB_TOOLS
import config


def print_harness_banner():
    print("""
========================================================================
             DeepSeek Agent Harness for Text-to-SQL
    Native Tool Calling | Phase 1/2 Guardrails | Sandbox Self-Healing
========================================================================
""")


def display_harness_result(result, user_query):
    print("\n" + "=" * 78)
    print(f"  [DeepSeek Harness Execution] Query: \"{user_query}\"")
    print("=" * 78)

    # 1. Phase 1 Blueprint
    print("\n[Harness State 1: Exploration & Reasoning Blueprint]")
    if result.blueprint:
        bp = result.blueprint
        print(f"  • Intent       : {bp.user_intent}")
        print(f"  • Chosen Tables: {bp.selected_tables}")
        print(f"  • Join Paths   : {bp.join_paths or 'None'}")
        print(f"  • Filters      : {bp.filter_conditions or 'None'}")
        print(f"  • Reasoning CoT: {bp.reasoning_summary}")
    else:
        print("  • No blueprint generated.")

    # 2. Phase 2 Sandbox Run
    print("\n[Harness State 2: Sandboxed SQL & AST Rewriting]")
    print(f"  • Candidate SQL  : {result.final_sql}")
    print(f"  • AST Rewritten  : {result.rewritten_sql}")

    # 3. Self-Healing Trace
    print("\n[Harness State 3: Self-Healing & Repair Trace]")
    if result.healing_history:
        for h in result.healing_history:
            print(f"  - Round {h.round_number} [{h.failure_type}]: {h.error_message}")
            print(f"    Dispatch Model: {h.model_used}")
    else:
        print("  • Zero errors: Model converged on the first sandbox execution attempt.")

    # 4. Sandbox Results
    print("\n[Harness State 4: Hardened DB Sandbox Output]")
    if result.execution_result:
        er = result.execution_result
        if er.success:
            print(f"  • Status : SUCCESS (Returned {er.row_count} row(s) in {er.execution_time_ms} ms)")
            print(f"  • Columns: {er.columns}")
            print("  • Data Sample:")
            for row in er.rows[:5]:
                print(f"    {row}")
            if er.row_count > 5:
                print(f"    ... and {er.row_count - 5} more row(s)")
        else:
            print(f"  • Status : FAILED ({er.error})")

    models_str = ", ".join(result.models_used) if result.models_used else "Default"
    print(f"\n[Harness Summary]: Latency = {result.total_latency_ms:.2f} ms | Status = {'PASSED' if result.success else 'FAILED'} | Models = {models_str}")
    print("=" * 78 + "\n")


def run_harness_cli():
    parser = argparse.ArgumentParser(description="DeepSeek Text-to-SQL Harness Runner")
    parser.add_argument("--query", "-q", type=str, help="Natural language query to process via DeepSeek Harness")
    parser.add_argument("--tools", action="store_true", help="Print all registered DeepSeek DB Tool definitions (JSON Schema)")
    args = parser.parse_args()

    db_path = config.DB_PATH
    if not Path(db_path).exists():
        print(f"[*] Initializing sample database at {db_path}...")
        init_sample_database(db_path)

    if args.tools:
        print(json.dumps(DEEPSEEK_DB_TOOLS, indent=2, ensure_ascii=False))
        return

    retriever, val_lookup = setup_metadata_and_valuelookup(db_path)
    sandbox = DBSandbox(db_path=db_path, timeout_seconds=config.STATEMENT_TIMEOUT_SECONDS)
    ast_guard = SQLASTGuard(default_limit=config.MAX_ROW_LIMIT, target_dialect=config.DIALECT)

    harness = DeepSeekHarness(
        retriever=retriever,
        value_lookup=val_lookup,
        sandbox=sandbox,
        ast_guard=ast_guard
    )

    if args.query:
        result = harness.run(args.query)
        display_harness_result(result, args.query)
        return

    # Interactive Loop
    print_harness_banner()
    print("DeepSeek Harness Interactive Session. Enter question (or 'exit' / 'q' to quit):\n")
    sample_queries = [
        "查询所有处于PAID状态的订单ID和金额",
        "查找购买过100%纯果汁的客户姓名和订单日期",
        "统计每个商品品类的商品数量和平均价格",
        "统计总销售额排名前3的商品名称与销售总金额"
    ]
    print("Try these sample questions:")
    for sq in sample_queries:
        print(f"  • {sq}")
    print()

    while True:
        try:
            user_input = input("DeepSeek-Harness > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting DeepSeek Harness.")
                break
            result = harness.run(user_input)
            display_harness_result(result, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    run_harness_cli()
