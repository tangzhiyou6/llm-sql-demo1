import json
import re
from typing import Dict, Any, List, Optional
from core.hybrid_retriever import HybridRetriever, TableMetadata
from core.value_lookup import ValueLookupEngine
from core.db_sandbox import DBSandbox
from core.llm_router import LLMRouter, LLMResponse
from agent.schemas import ReasoningBlueprint


SYSTEM_PROMPT_PHASE1 = """You are a Principal Database Architect & Text-to-SQL Planner.
Your sole goal in Phase 1 is to analyze the user request and produce a structured "Reasoning Blueprint".
DO NOT write full SQL code yet. Focus exclusively on understanding the schema, identifying join paths, and locking down filters.

Output MUST be a valid JSON object matching this schema:
{
  "user_intent": "Brief description of business intent",
  "selected_tables": ["table1", "table2"],
  "join_paths": ["table1.col = table2.col"],
  "filter_conditions": ["table.column = 'exact_val'"],
  "aggregations": ["COUNT(table.id)"],
  "ordering_and_limit": "ORDER BY ... LIMIT ...",
  "reasoning_summary": "Concise step-by-step logic explaining the join graph and predicate choices"
}
Only output the raw JSON object, without markdown quotes or explanation.
"""


class Phase1Planner:
    """
    Phase 1: Read-Only Exploration & Reasoning Blueprint Generation.
    Guarantees that SQL generation is preceded by schema grounding, join path analysis, and literal alignment.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        value_lookup: ValueLookupEngine,
        sandbox: DBSandbox,
        router: LLMRouter
    ):
        self.retriever = retriever
        self.value_lookup = value_lookup
        self.sandbox = sandbox
        self.router = router

    def plan(self, user_query: str) -> Tuple[ReasoningBlueprint, str, List[TableMetadata]]:
        """
        Executes Phase 1 exploration and returns:
          (ReasoningBlueprint, pruned_schema_str, candidate_tables)
        """
        # 1. Level 1: Table Hybrid Retrieval (Top-5 candidate tables per specification)
        candidate_tables = self.retriever.retrieve_tables_hybrid(user_query, top_k=5)

        # 2. Level 2: Column Pruning inside candidate tables
        pruned_schema_str = self.retriever.prune_and_format_schema(user_query, candidate_tables)

        # 3. Value-Lookup: Entity alignment for categorical values
        value_matches = self.value_lookup.search_query(user_query, top_k=5)
        value_alignment_str = self.value_lookup.format_matches_for_prompt(value_matches)

        # 4. Determine complexity for routing
        is_complex = len(candidate_tables) >= 2 or any(kw in user_query for kw in ["每个", "排名", "平均", "总计", "环比", "最高", "top"])

        # 5. Build user prompt
        user_prompt = (
            f"User Question: {user_query}\n\n"
            f"{pruned_schema_str}\n\n"
            f"{value_alignment_str}\n\n"
            f"Task: Generate the Reasoning Blueprint JSON. Ensure join_paths use the exact foreign keys specified in the schema."
        )

        response: LLMResponse = self.router.route_for_task(
            task_type="plan",
            is_complex=is_complex,
            system_prompt=SYSTEM_PROMPT_PHASE1,
            user_prompt=user_prompt
        )

        # 6. Parse JSON output
        blueprint = self._parse_blueprint(response.content, user_query, candidate_tables, value_matches)
        return blueprint, pruned_schema_str, candidate_tables

    def _parse_blueprint(
        self,
        raw_content: str,
        user_query: str,
        candidate_tables: List[TableMetadata],
        value_matches: List[Any]
    ) -> ReasoningBlueprint:
        """Parses model response into validated ReasoningBlueprint with fallback defaults."""
        try:
            clean = re.sub(r"<think>[\s\S]*?</think>", "", raw_content).strip()
            match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", clean, re.IGNORECASE)
            if match:
                json_str = match.group(1).strip()
            else:
                json_match = re.search(r"\{[\s\S]*\}", clean)
                json_str = json_match.group(0).strip() if json_match else clean

            data = json.loads(json_str)
            return ReasoningBlueprint(**data)
        except Exception:
            # Safe heuristic fallback blueprint
            selected_tables = [t.table_name for t in candidate_tables[:2]]
            join_paths = []
            if len(selected_tables) > 1:
                # Look up foreign key path
                t1 = candidate_tables[0]
                for fk in t1.foreign_keys:
                    if fk.get("to_table") in selected_tables:
                        join_paths.append(f"{t1.table_name}.{fk['from_column']} = {fk['to_table']}.{fk['to_column']}")

            filters = [m.filter_hint for m in value_matches]
            return ReasoningBlueprint(
                user_intent=user_query,
                selected_tables=selected_tables,
                join_paths=join_paths,
                filter_conditions=filters,
                aggregations=[],
                ordering_and_limit="LIMIT 100",
                reasoning_summary="Auto-synthesized fallback blueprint based on hybrid retrieval and value matches."
            )
