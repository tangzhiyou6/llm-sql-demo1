import math
from typing import List, Tuple, Any, Dict, Optional
from pydantic import BaseModel, Field
from core.db_sandbox import ExecutionResult


class TestCaseResult(BaseModel):
    test_id: str
    difficulty: str
    question: str
    ground_truth_sql: str
    generated_sql: str
    syntax_valid: bool
    execution_success: bool
    is_empty: bool
    execution_accuracy: bool  # EX match
    latency_ms: float
    error: Optional[str] = None


class BenchmarkMetrics(BaseModel):
    total_cases: int = 0
    syntax_success_count: int = 0
    sr_rate: float = 0.0     # Syntax & Execution Success Rate (SR)
    ex_count: int = 0
    ex_rate: float = 0.0     # Execution Accuracy (EX)
    empty_result_count: int = 0
    err_rate: float = 0.0    # Empty Result Rate (ERR)
    avg_latency_ms: float = 0.0
    tier_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class Text2SQLEvaluator:
    """
    Evaluation Engine calculating the 4D Metric Radar:
      1. EX (Execution Accuracy): Result set equivalence with float epsilon tolerance (< 1e-4)
      2. SR (Syntax Success Rate): AST and execution pass rate without DB errors
      3. ERR (Empty Result Rate): Non-error queries yielding 0 rows
      4. Execution Time (Latency): End-to-end processing latency
    """

    def __init__(self, float_tolerance: float = 1e-4):
        self.float_tolerance = float_tolerance

    def compare_results(
        self,
        gt_result: ExecutionResult,
        pred_result: ExecutionResult
    ) -> bool:
        """
        Compares ground truth execution result against candidate execution result.
        Permits column order and row order insensitivity when applicable, with float tolerance.
        """
        if not gt_result.success or not pred_result.success:
            return False

        gt_rows = gt_result.rows
        pred_rows = pred_result.rows

        # If row counts differ, not equivalent
        if len(gt_rows) != len(pred_rows):
            return False

        if len(gt_rows) == 0:
            return True

        # Check column count
        if len(gt_rows[0]) != len(pred_rows[0]):
            return False

        # Normalize and sort rows for order-invariant comparison
        def normalize_val(val: Any) -> Any:
            if val is None:
                return "NULL"
            if isinstance(val, (float, int)):
                # Round to 4 decimal places for stable hash/sort
                return round(float(val), 4)
            return str(val).strip()

        def normalize_row(row: Tuple[Any, ...]) -> Tuple[Any, ...]:
            return tuple(normalize_val(v) for v in row)

        try:
            sorted_gt = sorted([normalize_row(r) for r in gt_rows], key=lambda x: str(x))
            sorted_pred = sorted([normalize_row(r) for r in pred_rows], key=lambda x: str(x))

            for r_gt, r_pred in zip(sorted_gt, sorted_pred):
                for v_gt, v_pred in zip(r_gt, r_pred):
                    if isinstance(v_gt, float) and isinstance(v_pred, float):
                        if not math.isclose(v_gt, v_pred, rel_tol=self.float_tolerance, abs_tol=self.float_tolerance):
                            return False
                    elif v_gt != v_pred:
                        return False
            return True
        except Exception:
            return False

    def compute_metrics(self, results: List[TestCaseResult]) -> BenchmarkMetrics:
        """Computes aggregate 4-dimension radar metrics and stratified tier breakdowns."""
        total = len(results)
        if total == 0:
            return BenchmarkMetrics()

        sr_count = sum(1 for r in results if r.execution_success)
        ex_count = sum(1 for r in results if r.execution_accuracy)
        empty_count = sum(1 for r in results if r.execution_success and r.is_empty)
        avg_latency = sum(r.latency_ms for r in results) / total

        # Breakdown by tier (Simple, Moderate, Challenging)
        tiers = ["Simple", "Moderate", "Challenging"]
        tier_stats = {}
        for t in tiers:
            sub = [r for r in results if r.difficulty == t]
            sub_total = len(sub)
            if sub_total > 0:
                sub_ex = sum(1 for r in sub if r.execution_accuracy)
                sub_sr = sum(1 for r in sub if r.execution_success)
                tier_stats[t] = {
                    "total": sub_total,
                    "ex_count": sub_ex,
                    "ex_rate": round(sub_ex / sub_total * 100, 2),
                    "sr_rate": round(sub_sr / sub_total * 100, 2)
                }

        return BenchmarkMetrics(
            total_cases=total,
            syntax_success_count=sr_count,
            sr_rate=round(sr_count / total * 100, 2),
            ex_count=ex_count,
            ex_rate=round(ex_count / total * 100, 2),
            empty_result_count=empty_count,
            err_rate=round(empty_count / total * 100, 2),
            avg_latency_ms=round(avg_latency, 2),
            tier_stats=tier_stats
        )
