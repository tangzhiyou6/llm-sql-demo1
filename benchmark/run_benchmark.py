import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.db_sandbox import DBSandbox
from core.ast_guard import SQLASTGuard
from benchmark.sample_db import init_sample_database, setup_metadata_and_valuelookup
from benchmark.evaluator import Text2SQLEvaluator, TestCaseResult
from agent.pipeline import Text2SQLPipeline
import config


def run_benchmark_suite(db_path: str = config.DB_PATH, dataset_path: str = None) -> None:
    dataset_file = Path(dataset_path or (BASE_DIR / "benchmark" / "dataset.json"))
    if not dataset_file.exists():
        print(f"Error: Dataset file not found at {dataset_file}")
        return

    with open(dataset_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # 1. Initialize sample database if needed
    if not Path(db_path).exists():
        print(f"[*] Seeding benchmark database at {db_path}...")
        init_sample_database(db_path)

    # 2. Build metadata and value-lookup engines
    retriever, val_lookup = setup_metadata_and_valuelookup(db_path)
    sandbox = DBSandbox(db_path=db_path, timeout_seconds=config.STATEMENT_TIMEOUT_SECONDS)
    ast_guard = SQLASTGuard(default_limit=config.MAX_ROW_LIMIT, target_dialect="sqlite")

    pipeline = Text2SQLPipeline(
        retriever=retriever,
        value_lookup=val_lookup,
        sandbox=sandbox,
        ast_guard=ast_guard
    )

    evaluator = Text2SQLEvaluator(float_tolerance=1e-4)
    results = []

    print("=" * 80)
    print(f"  Running Text-to-SQL Benchmark ({len(cases)} cases across Simple/Moderate/Challenging)")
    print("=" * 80)

    for idx, tc in enumerate(cases, 1):
        q_id = tc["id"]
        difficulty = tc["difficulty"]
        question = tc["question"]
        gt_sql = tc["ground_truth_sql"]

        print(f"[{idx:02d}/{len(cases):02d}] ({difficulty:11s}) {question[:45]}...", end=" ", flush=True)

        # Execute Ground Truth SQL in sandbox
        gt_res = sandbox.execute_query(gt_sql)

        # Run pipeline
        pipe_res = pipeline.run(question)

        # If offline simulation mode and API not configured, for test integrity evaluate predicted query or gt
        pred_res = pipe_res.execution_result
        pred_sql = pipe_res.rewritten_sql or pipe_res.final_sql

        is_ex = evaluator.compare_results(gt_res, pred_res)
        syntax_valid = pipe_res.execution_result.success if pipe_res.execution_result else False

        results.append(
            TestCaseResult(
                test_id=q_id,
                difficulty=difficulty,
                question=question,
                ground_truth_sql=gt_sql,
                generated_sql=pred_sql,
                syntax_valid=syntax_valid,
                execution_success=syntax_valid,
                is_empty=pred_res.is_empty if pred_res else True,
                execution_accuracy=is_ex,
                latency_ms=pipe_res.total_latency_ms,
                error=pipe_res.execution_result.error if pipe_res.execution_result else None
            )
        )

        status_flag = "✓ EX PASS" if is_ex else ("~ SR PASS" if syntax_valid else "✗ FAIL")
        print(f"[{status_flag}] ({pipe_res.total_latency_ms:.1f}ms)")

    # 3. Compute 4D metrics
    metrics = evaluator.compute_metrics(results)

    print("\n" + "=" * 80)
    print("                      4D METRIC RADAR EVALUATION REPORT                     ")
    print("=" * 80)
    print(f"  • Total Test Cases           : {metrics.total_cases}")
    print(f"  • EX (Execution Accuracy)    : {metrics.ex_rate:.1f}% ({metrics.ex_count}/{metrics.total_cases})")
    print(f"  • SR (Syntax Success Rate)   : {metrics.sr_rate:.1f}% ({metrics.syntax_success_count}/{metrics.total_cases})")
    print(f"  • ERR (Empty Result Rate)    : {metrics.err_rate:.1f}% ({metrics.empty_result_count}/{metrics.total_cases})")
    print(f"  • Avg Latency (End-to-End)   : {metrics.avg_latency_ms:.2f} ms")
    print("-" * 80)
    print("  Tier Breakdown:")
    for tier, stat in metrics.tier_stats.items():
        print(f"    - {tier:12s} : EX {stat['ex_rate']:5.1f}% | SR {stat['sr_rate']:5.1f}% ({stat['ex_count']}/{stat['total']})")
    print("=" * 80)

    # Save summary report markdown
    report_file = BASE_DIR / "benchmark" / "BENCHMARK_REPORT.md"
    with open(report_file, "w", encoding="utf-8") as rf:
        rf.write(f"# Text-to-SQL Benchmark Evaluation Report\n\n")
        rf.write(f"| Metric | Result |\n|---|---|\n")
        rf.write(f"| **Total Cases** | `{metrics.total_cases}` |\n")
        rf.write(f"| **EX (Execution Accuracy)** | `{metrics.ex_rate}%` ({metrics.ex_count}/{metrics.total_cases}) |\n")
        rf.write(f"| **SR (Syntax Success Rate)** | `{metrics.sr_rate}%` ({metrics.syntax_success_count}/{metrics.total_cases}) |\n")
        rf.write(f"| **ERR (Empty Result Rate)** | `{metrics.err_rate}%` ({metrics.empty_result_count}/{metrics.total_cases}) |\n")
        rf.write(f"| **Avg Latency** | `{metrics.avg_latency_ms} ms` |\n\n")
        rf.write("## Stratified Tier Breakdown\n\n")
        rf.write("| Difficulty | Total | EX Count | EX Rate | SR Rate |\n|---|---|---|---|---|\n")
        for tier, stat in metrics.tier_stats.items():
            rf.write(f"| {tier} | {stat['total']} | {stat['ex_count']} | {stat['ex_rate']}% | {stat['sr_rate']}% |\n")
    print(f"\n[+] Detailed benchmark report saved to: {report_file}")


if __name__ == "__main__":
    run_benchmark_suite()
