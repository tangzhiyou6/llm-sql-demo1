---
name: text-to-sql
description: Enterprise-grade Text-to-SQL workflow using two-phase Plan-then-Solve, AST safety guardrails, hybrid metadata retrieval, read-only database sandbox, Execution Consensus (P1), ExplainGuard (P2), and Semantic Metric Layer (P3).
whenToUse: Use whenever the user asks questions about business data, sales, orders, customers, products, or requires generating and verifying database SQL queries.
---

# Enterprise Text-to-SQL Skill for DeepSeek Harness (dsh)

This skill provides an industrial-grade Text-to-SQL architecture that eliminates hallucinations, prevents ReAct infinite loops, caps output rows, intercepts Cartesian products, and isolates database execution in a hardened read-only environment.

## 1. Quick Execution via dsh / CLI

### Standard Text-to-SQL Pipeline:
```bash
.venv/bin/python main.py --query "<User Question>"
```

### P1: Multi-Candidate Execution Consensus & Result Clustering:
```bash
.venv/bin/python main.py --query "<User Question>" --consensus
```

### P2: Cost-Based ExplainGuard Pre-Filter Analysis:
```bash
.venv/bin/python main.py --explain "SELECT * FROM customers, orders"
```

### P3: Deterministic Semantic Metric Layer Query:
```bash
.venv/bin/python main.py --semantic --metrics total_revenue,order_count --dimensions city
```

### Benchmark Suite (15 Golden Cases):
```bash
.venv/bin/python main.py --benchmark
```

### Node.js Custom Plugin Verification:
```bash
node test_plugin.js
```

---

## 2. Advanced Architecture Enhancements (P0 ~ P3)

### P0: Persistent Daemon & Sub-Millisecond Tool Calling
- The DSH plugin maintains a long-lived persistent JSON-RPC stdio daemon (`mcp_server.py`), eliminating per-call Python cold starts (~300ms) down to **< 1ms**.

### P1: Execution Consensus & Clustering Engine (`core/consensus_engine.py`)
- Generates $N$ diverse SQL candidates using temperature sampling and prompt variations.
- Normalizes rows (float precision rounding, NULL mapping, deterministic sorting) and clusters candidates by exact execution result set equivalence.
- Majority voting selects the winning canonical SQL and reports confidence.

### P2: Cost-Based Pre-Filter (`core/explain_guard.py`)
- Intercepts dangerous SQL before execution using `EXPLAIN QUERY PLAN`.
- Detects and blocks Cartesian Products (multiple unindexed `SCAN` loops) with clear actionable diagnostic errors (`CRITICAL`).
- Emits advisory warnings for unindexed scans or temporary B-tree allocations.

### P3: Semantic Metric Layer Compiler (`core/semantic_layer.py`)
- Declarative semantic modeling of business metrics (`total_revenue`, `order_count`, `avg_order_value`, `total_units_sold`) and dimensions (`city`, `product_name`, `order_status`).
- Automatic topology graph BFS traversal resolves minimal join paths (e.g. `order_items` $\to$ `orders` $\to$ `customers`) with **zero join hallucination**.

---

## 3. Two-Phase Constrained Workflow Rules

### Phase 1: Read-Only Exploration & Reasoning Blueprint
1. **Schema Retrieval**: Inspect metadata using `schema_search` or hybrid retrieval.
2. **Entity & Enum Alignment**: Use `value_lookup` to check actual values (e.g. "果汁" $\to$ `'100%纯果汁'`).
3. **Mandatory Output**: Formulate a structured Reasoning Blueprint (`selected_tables`, `join_paths`, `filter_conditions`).
4. **Hard Guardrail**: NEVER write or execute complex SQL in Phase 1.

### Phase 2: Generation, AST Guard, Cost Pre-filter & Execution
1. **AST Guardian (`sqlglot`)**: Only `SELECT` statements are permitted. Limits are dynamically injected or capped (`LIMIT 100`).
2. **ExplainGuard Pre-filter**: Verifies query plan before running; rejects Cartesian explosions.
3. **Hardened Read-Only Sandbox**: Strictly read-only connection (`mode=ro`) with hard 5-second timeout.
4. **Bounded Self-Healing**: Self-healing reflection capped at 2~3 rounds max.
