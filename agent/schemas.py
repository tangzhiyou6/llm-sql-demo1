from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from core.db_sandbox import ExecutionResult
from core.consensus_engine import ConsensusResult


class ReasoningBlueprint(BaseModel):
    """
    Phase 1 output: Structured reasoning blueprint.
    Forbids writing complex SQL in Phase 1, focusing entirely on schema grounding & join path validation.
    """
    user_intent: str = Field(description="Normalized business intent of user query")
    selected_tables: List[str] = Field(description="Tables required to satisfy the query")
    join_paths: List[str] = Field(default_factory=list, description="Explicit foreign key join conditions")
    filter_conditions: List[str] = Field(default_factory=list, description="Filter predicates with precise DB values")
    aggregations: List[str] = Field(default_factory=list, description="Aggregations or group by requirements")
    ordering_and_limit: str = Field(default="", description="Ordering columns and row limits")
    reasoning_summary: str = Field(default="", description="Step-by-step CoT chain of thought")


class SelfHealingRound(BaseModel):
    round_number: int
    attempted_sql: str
    failure_type: str  # AST_ERROR, EXECUTION_ERROR, EMPTY_RESULT
    error_message: str
    reflection_and_fix: str
    model_used: str


class PipelineResult(BaseModel):
    user_query: str
    blueprint: Optional[ReasoningBlueprint] = None
    final_sql: str = ""
    rewritten_sql: str = ""
    execution_result: Optional[ExecutionResult] = None
    healing_history: List[SelfHealingRound] = Field(default_factory=list)
    success: bool = False
    total_latency_ms: float = 0.0
    models_used: List[str] = Field(default_factory=list)
    consensus_result: Optional[ConsensusResult] = None
