import re
from typing import List, Dict, Any, Set, Tuple, Optional
from core.ast_guard import SQLASTGuard, ASTValidationError
from core.db_sandbox import DBSandbox, ExecutionResult
from core.llm_router import LLMRouter, LLMResponse
from core.hybrid_retriever import TableMetadata
from agent.schemas import ReasoningBlueprint, SelfHealingRound
import config


SYSTEM_PROMPT_GENERATOR = """You are a Senior SQL Engineer.
Generate an accurate, standard SQL query strictly matching the provided Reasoning Blueprint and Database Schema.
Rules:
1. Output ONLY the raw SQL query inside a ```sql ... ``` markdown code block.
2. Use valid standard SQL syntax compatible with the target database.
3. Strictly adhere to the tables, join paths, and filter literals specified in the Blueprint.
4. Do NOT hallucinate tables or columns not defined in the schema.
"""

SYSTEM_PROMPT_HEALER = """You are an Expert SQL Debugger and Database Engine Specialist.
Your mission is to analyze the failed SQL query and the exact runtime/AST error, perform step-by-step root-cause reflection, and output the corrected SQL query.
Rules:
1. Carefully inspect the line/column error or database operational exception.
2. Output your reasoning thought, followed by the corrected SQL query enclosed in ```sql ... ```.
3. Ensure no invalid column references, syntax mistakes, or mismatched types remain.
"""


class Phase2Generator:
    """
    Phase 2: Code Generation, AST Guard Validation, Dynamic Rewriting, and Constrained Self-Healing Execution.
    Limits self-healing to a strict maximum (default 2-3 rounds) to eliminate infinite loops and token blowup.
    """

    def __init__(
        self,
        ast_guard: SQLASTGuard,
        sandbox: DBSandbox,
        router: LLMRouter,
        max_healing_rounds: int = 3
    ):
        self.ast_guard = ast_guard
        self.sandbox = sandbox
        self.router = router
        self.max_healing_rounds = max_healing_rounds

    def generate_candidate_sql(
        self,
        blueprint: ReasoningBlueprint,
        schema_str: str,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Generates a candidate SQL query from blueprint with specified temperature."""
        prompt = (
            f"Reasoning Blueprint:\n"
            f"- Intent: {blueprint.user_intent}\n"
            f"- Tables: {blueprint.selected_tables}\n"
            f"- Join Paths: {blueprint.join_paths}\n"
            f"- Filters: {blueprint.filter_conditions}\n"
            f"- Aggregations: {blueprint.aggregations}\n"
            f"- Ordering/Limit: {blueprint.ordering_and_limit}\n"
            f"- Reasoning: {blueprint.reasoning_summary}\n\n"
            f"{schema_str}\n\n"
            f"Generate the standard SQL query matching this blueprint."
        )
        resp = self.router.call_fast_model(
            system_prompt=SYSTEM_PROMPT_GENERATOR,
            user_prompt=prompt,
            temperature=temperature
        )
        return self._extract_sql(resp.content)

    _generate_candidate_sql = generate_candidate_sql

    def generate_and_execute(
        self,
        blueprint: ReasoningBlueprint,
        schema_str: str,
        candidate_tables: List[TableMetadata]
    ) -> Tuple[str, str, ExecutionResult, List[SelfHealingRound]]:
        """
        Executes Phase 2 workflow:
          Generate -> AST Guard -> Limit Rewrite -> Sandbox Run -> (Self-Heal Loop <= max_rounds)
        """
        allowed_tables = {t.table_name for t in candidate_tables}
        # Also include blueprint tables if valid
        allowed_tables.update([t.lower() for t in blueprint.selected_tables])

        allowed_columns_map = {}
        for t in candidate_tables:
            allowed_columns_map[t.table_name.lower()] = {c.name.lower() for c in t.columns}

        healing_history: List[SelfHealingRound] = []

        # 1. Round 0: Initial SQL generation
        current_sql = self.generate_candidate_sql(blueprint, schema_str, temperature=0.0)
        current_model = self.router.fast_model

        # 2. Execution & Self-Healing Loop
        for round_idx in range(self.max_healing_rounds + 1):
            rewritten_sql = ""
            ast_error_msg = None
            exec_result = None

            # Step A: AST Validation & Hallucination Check
            try:
                parsed_ast, _, _ = self.ast_guard.validate_and_extract(
                    current_sql,
                    allowed_tables=allowed_tables,
                    allowed_columns_map=allowed_columns_map
                )

                # Step B: AST Rewriting (Inject/Cap LIMIT 100)
                capped_ast = self.ast_guard.inject_or_cap_limit(parsed_ast, max_limit=config.MAX_ROW_LIMIT)
                rewritten_sql = capped_ast.sql(dialect=config.DIALECT, pretty=True)

            except Exception as e:
                ast_error_msg = self.ast_guard.format_error_for_reflection(e, current_sql)

            # Step C: If AST passed, execute in DB Sandbox
            if not ast_error_msg:
                exec_result = self.sandbox.execute_query(rewritten_sql)
                # Success condition: query ran without error
                if exec_result.success:
                    return current_sql, rewritten_sql, exec_result, healing_history

            # If we reached here, an error occurred (AST error or Execution error)
            # If we've exhausted retry rounds, break out
            if round_idx >= self.max_healing_rounds:
                break

            # Construct Reflection Error Diagnostic
            failure_type = "AST_ERROR" if ast_error_msg else "EXECUTION_ERROR"
            error_description = ast_error_msg if ast_error_msg else f"Database Error: {exec_result.error}"

            healing_prompt = (
                f"### Previous SQL Attempt:\n```sql\n{current_sql}\n```\n\n"
                f"### Error Encountered ({failure_type}):\n{error_description}\n\n"
                f"### Blueprint Reference:\n"
                f"- Intent: {blueprint.user_intent}\n"
                f"- Join Paths: {blueprint.join_paths}\n"
                f"- Filters: {blueprint.filter_conditions}\n\n"
                f"{schema_str}\n\n"
                f"Please reflect on the error and output the corrected SQL in ```sql ... ``` block."
            )

            # Route to Deep Reasoning Model (DeepSeek-R1) for self-healing
            heal_resp = self.router.call_reasoning_model(
                system_prompt=SYSTEM_PROMPT_HEALER,
                user_prompt=healing_prompt
            )

            healing_history.append(
                SelfHealingRound(
                    round_number=round_idx + 1,
                    attempted_sql=current_sql,
                    failure_type=failure_type,
                    error_message=error_description,
                    reflection_and_fix=heal_resp.thinking_cot or "Applied targeted fix to address error.",
                    model_used=heal_resp.model
                )
            )

            current_sql = self._extract_sql(heal_resp.content)
            current_model = heal_resp.model

        # Fallback return when max healing rounds exceeded
        final_exec_res = exec_result or ExecutionResult(
            success=False,
            error=f"Failed to converge after {self.max_healing_rounds} self-healing rounds. Last error: {ast_error_msg or (exec_result.error if exec_result else 'Unknown')}"
        )
        return current_sql, rewritten_sql or current_sql, final_exec_res, healing_history

    def _extract_sql(self, text: str) -> str:
        """Extracts SQL string from markdown fences or raw response, stripping any <think> tags."""
        clean = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", clean, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # If no code block, return stripped text
        return clean.strip()
