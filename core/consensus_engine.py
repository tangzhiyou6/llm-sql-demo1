import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from core.db_sandbox import DBSandbox, ExecutionResult
from core.ast_guard import SQLASTGuard
from core.llm_router import LLMRouter, LLMResponse
import config


class CandidateQuery(BaseModel):
    sql: str
    rewritten_sql: Optional[str] = None
    success: bool = False
    execution_result: Optional[ExecutionResult] = None
    result_hash: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class ExecutionCluster(BaseModel):
    cluster_hash: str
    vote_count: int
    canonical_sql: str
    rewritten_sql: str
    sample_result: ExecutionResult
    candidate_indices: List[int] = Field(default_factory=list)


class ConsensusResult(BaseModel):
    success: bool
    winning_sql: Optional[str] = None
    winning_rewritten_sql: Optional[str] = None
    winning_result: Optional[ExecutionResult] = None
    confidence: float = 0.0
    total_candidates: int = 0
    valid_candidates: int = 0
    clusters: List[ExecutionCluster] = Field(default_factory=list)
    candidates: List[CandidateQuery] = Field(default_factory=list)
    error: Optional[str] = None


class ExecutionConsensusEngine:
    """
    Execution Consensus & Result Clustering Engine.
    Executes multiple diverse candidate SQL queries in DBSandbox,
    clusters them by their normalized result sets, and selects the consensus winner.
    """

    def __init__(self, sandbox: DBSandbox, ast_guard: Optional[SQLASTGuard] = None):
        self.sandbox = sandbox
        self.ast_guard = ast_guard or SQLASTGuard(
            default_limit=config.MAX_ROW_LIMIT,
            target_dialect=config.DIALECT
        )

    @staticmethod
    def normalize_cell(val: Any) -> Any:
        """Normalizes individual data cell values for consistent hashing."""
        if val is None:
            return None
        if isinstance(val, float):
            return round(val, 4)
        if isinstance(val, (int, bool)):
            return val
        if isinstance(val, str):
            return val.strip()
        return str(val)

    @classmethod
    def compute_result_hash(cls, rows: List[Tuple[Any, ...]], order_sensitive: bool = False) -> str:
        """
        Computes an invariant cryptographic hash of query execution rows.
        Normalizes numeric floats, handles NULLs, and sorts rows unless order is meaningful.
        """
        normalized_rows = [
            tuple(cls.normalize_cell(cell) for cell in row)
            for row in rows
        ]

        if not order_sensitive:
            # Sort deterministically so queries with arbitrary row order match
            try:
                normalized_rows = sorted(normalized_rows, key=lambda r: str(r))
            except Exception:
                pass

        serialized = json.dumps(normalized_rows, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def evaluate_candidates(
        self,
        candidate_sqls: List[str],
        order_sensitive: bool = False
    ) -> ConsensusResult:
        """
        Executes a list of candidate SQL queries, computes their result sets,
        clusters by execution equivalence, and selects the majority winner.
        """
        candidates: List[CandidateQuery] = []
        clusters_map: Dict[str, ExecutionCluster] = {}

        for idx, raw_sql in enumerate(candidate_sqls):
            cand = CandidateQuery(sql=raw_sql)
            try:
                # 1. AST Validation and Limit Injection
                parsed, _, _ = self.ast_guard.validate_and_extract(raw_sql)
                capped = self.ast_guard.inject_or_cap_limit(parsed, max_limit=config.MAX_ROW_LIMIT)
                rewritten_sql = capped.sql(dialect=config.DIALECT, pretty=True)
                cand.rewritten_sql = rewritten_sql

                # 2. DBSandbox Execution (includes ExplainGuard pre-filtering)
                exec_res = self.sandbox.execute_query(rewritten_sql)
                cand.execution_result = exec_res
                cand.success = exec_res.success
                cand.execution_time_ms = exec_res.execution_time_ms

                if exec_res.success:
                    # 3. Compute Result Set Hash
                    r_hash = self.compute_result_hash(exec_res.rows, order_sensitive=order_sensitive)
                    cand.result_hash = r_hash

                    # 4. Cluster by result hash
                    if r_hash not in clusters_map:
                        clusters_map[r_hash] = ExecutionCluster(
                            cluster_hash=r_hash,
                            vote_count=1,
                            canonical_sql=raw_sql,
                            rewritten_sql=rewritten_sql,
                            sample_result=exec_res,
                            candidate_indices=[idx]
                        )
                    else:
                        cluster = clusters_map[r_hash]
                        cluster.vote_count += 1
                        cluster.candidate_indices.append(idx)
                        # Favor simpler or faster query as canonical
                        if exec_res.execution_time_ms < cluster.sample_result.execution_time_ms:
                            cluster.canonical_sql = raw_sql
                            cluster.rewritten_sql = rewritten_sql
                            cluster.sample_result = exec_res
                else:
                    cand.error = exec_res.error

            except Exception as e:
                cand.success = False
                cand.error = str(e)

            candidates.append(cand)

        total_candidates = len(candidates)
        valid_candidates = sum(1 for c in candidates if c.success)

        if not clusters_map:
            # All candidates failed
            return ConsensusResult(
                success=False,
                total_candidates=total_candidates,
                valid_candidates=0,
                candidates=candidates,
                error="All candidate queries failed execution or were blocked by guards."
            )

        # Sort clusters by vote count descending, then by fastest execution time
        sorted_clusters = sorted(
            clusters_map.values(),
            key=lambda cl: (cl.vote_count, -cl.sample_result.execution_time_ms),
            reverse=True
        )

        winner = sorted_clusters[0]
        confidence = round(winner.vote_count / total_candidates, 3)

        return ConsensusResult(
            success=True,
            winning_sql=winner.canonical_sql,
            winning_rewritten_sql=winner.rewritten_sql,
            winning_result=winner.sample_result,
            confidence=confidence,
            total_candidates=total_candidates,
            valid_candidates=valid_candidates,
            clusters=sorted_clusters,
            candidates=candidates
        )

    def generate_and_vote(
        self,
        router: LLMRouter,
        prompt: str,
        n: int = 3,
        system_prompt: Optional[str] = None
    ) -> ConsensusResult:
        """
        Generates N diverse SQL candidates via temperature sampling,
        then executes consensus clustering to select the optimal query.
        """
        candidate_sqls: List[str] = []
        temperatures = [0.0, 0.4, 0.7] if n == 3 else [0.0] + [0.2 * i for i in range(1, n)]

        for t in temperatures[:n]:
            try:
                resp = router.route_for_task(
                    task_type="generate",
                    system_prompt=system_prompt or "You are a professional SQLite Text-to-SQL expert.",
                    user_prompt=f"{prompt}\n(Generation strategy: temperature={t})",
                    temperature=t
                )
                raw_text = resp.text.strip()
                # Extract SQL code block if wrapped in markdown
                if "```sql" in raw_text:
                    raw_text = raw_text.split("```sql")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()
                candidate_sqls.append(raw_text)
            except Exception as e:
                # If LLM call fails, skip
                continue

        if not candidate_sqls:
            return ConsensusResult(
                success=False,
                error="Failed to generate any candidate queries from LLM router."
            )

        return self.evaluate_candidates(candidate_sqls)
