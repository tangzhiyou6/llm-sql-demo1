import time
from typing import Optional, List
from core.hybrid_retriever import HybridRetriever
from core.value_lookup import ValueLookupEngine
from core.db_sandbox import DBSandbox
from core.ast_guard import SQLASTGuard
from core.llm_router import LLMRouter
from agent.schemas import PipelineResult, ReasoningBlueprint
from agent.phase1_planner import Phase1Planner
from agent.phase2_generator import Phase2Generator
import config


class Text2SQLPipeline:
    """
    End-to-End Enterprise Text-to-SQL Pipeline implementing:
      1. Level 1 & 2 Hybrid Metadata Retrieval + Value-Lookup Alignment
      2. Phase 1: Read-Only Exploration & Reasoning Blueprint (DeepSeek Dual-Model)
      3. Phase 2: SQL Generation, sqlglot AST Safety Check, Limit Rewriting
      4. Hardened Read-Only DB Sandbox Execution with bounded Self-Healing (<= 2-3 rounds)
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        value_lookup: ValueLookupEngine,
        sandbox: DBSandbox,
        ast_guard: Optional[SQLASTGuard] = None,
        router: Optional[LLMRouter] = None,
        max_healing_rounds: int = config.MAX_SELF_HEAL_ROUNDS
    ):
        self.retriever = retriever
        self.value_lookup = value_lookup
        self.sandbox = sandbox
        self.ast_guard = ast_guard or SQLASTGuard(default_limit=config.MAX_ROW_LIMIT, target_dialect=config.DIALECT)
        self.router = router or LLMRouter()

        self.phase1_planner = Phase1Planner(
            retriever=self.retriever,
            value_lookup=self.value_lookup,
            sandbox=self.sandbox,
            router=self.router
        )
        self.phase2_generator = Phase2Generator(
            ast_guard=self.ast_guard,
            sandbox=self.sandbox,
            router=self.router,
            max_healing_rounds=max_healing_rounds
        )

    def run(self, user_query: str) -> PipelineResult:
        """Executes the two-phase pipeline on a user natural language query."""
        start_time = time.perf_counter()
        models_used = set()

        # --- Phase 1: Read-Only Exploration & Blueprint Generation ---
        blueprint, schema_str, candidate_tables = self.phase1_planner.plan(user_query)

        # --- Phase 2: Generation, AST Guard, and Self-Healing Execution ---
        final_sql, rewritten_sql, exec_result, healing_history = self.phase2_generator.generate_and_execute(
            blueprint=blueprint,
            schema_str=schema_str,
            candidate_tables=candidate_tables
        )

        for step in healing_history:
            models_used.add(step.model_used)

        total_latency = (time.perf_counter() - start_time) * 1000.0

        return PipelineResult(
            user_query=user_query,
            blueprint=blueprint,
            final_sql=final_sql,
            rewritten_sql=rewritten_sql,
            execution_result=exec_result,
            healing_history=healing_history,
            success=exec_result.success if exec_result else False,
            total_latency_ms=round(total_latency, 2),
            models_used=list(models_used)
        )
