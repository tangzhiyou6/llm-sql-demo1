---
name: text-to-sql
description: Enterprise-grade Text-to-SQL workflow using two-phase Plan-then-Solve, AST safety guardrails, hybrid metadata retrieval, and read-only database sandbox.
whenToUse: Use whenever the user asks questions about business data, sales, orders, customers, products, or requires generating and verifying database SQL queries.
---

# Enterprise Text-to-SQL Skill for DeepSeek Harness (dsh)

This skill provides an industrial-grade Text-to-SQL architecture that eliminates hallucinations, prevents ReAct infinite loops, caps output rows, and isolates database execution in a hardened read-only environment.

## 1. Quick Execution via dsh / CLI

When you need to answer a natural language question or execute a Text-to-SQL task, run the runner in your terminal:

```bash
.venv/bin/python harness_runner.py --query "<User Question>"
```

Or run via the main CLI:
```bash
.venv/bin/python main.py --harness --query "<User Question>"
```

For benchmark evaluation across 15 stratified Golden cases (EX / SR / ERR / Latency):
```bash
.venv/bin/python main.py --benchmark
```

---

## 2. Core Architecture & Workflow Rules

When planning or generating SQL, strictly adhere to the **Two-Phase Constrained Workflow**:

### Phase 1: Read-Only Exploration & Reasoning Blueprint
1. **Schema Retrieval**: Do not guess tables or columns. Inspect metadata using `schema_search` or run Level 1/2 hybrid retrieval.
2. **Entity & Enum Alignment**: Always use `value_lookup` to check actual values (e.g., query mentions "果汁" $\to$ align to `'100%纯果汁'`; "已支付" $\to$ `'PAID'`).
3. **Mandatory Output**: Formulate a structured **Reasoning Blueprint** containing:
   - `selected_tables`: exact table names (e.g. `['orders', 'customers']`)
   - `join_paths`: verified foreign keys (e.g. `orders.customer_id = customers.id`)
   - `filter_conditions`: exact literal predicates (e.g. `orders.status = 'PAID'`)
   - `ordering_and_limit`: standard ordering and limits.
4. **Hard Guardrail**: **NEVER write or execute complex SQL in Phase 1.**

### Phase 2: Code Generation, AST Guard & Sandbox Execution
1. **SQL Generation**: Generate standard SQL strictly according to the approved Blueprint.
2. **AST Guardian (`sqlglot`)**:
   - Only `SELECT` statements are permitted (`DROP`, `UPDATE`, `DELETE`, `ALTER` are blocked).
   - Dynamic Rewrite: An outer `LIMIT 100` clause is automatically injected or capped.
   - Table and column references are validated against the schema whitelist.
3. **Hardened DB Sandbox**:
   - Connection is strictly read-only (`mode=ro`).
   - Hard execution timeout is enforced (5.0s) to prevent full table scans from hanging.
4. **Bounded Self-Healing**:
   - If AST validation fails or execution throws an error, inspect the precise line/column error or database message.
   - Self-healing is strictly capped at **2~3 rounds maximum** to prevent infinite loops and token explosion.

---

## 3. Database Schema Overview (`ecommerce.db`)

- **`customers`**: `id` (PK), `name`, `city`, `email`, `created_at`
- **`products`**: `id` (PK), `product_name`, `category`, `price`, `stock`
- **`orders`**: `id` (PK), `customer_id` (FK $\to$ `customers.id`), `order_date`, `status`, `total_amount`
- **`order_items`**: `id` (PK), `order_id` (FK $\to$ `orders.id`), `product_id` (FK $\to$ `products.id`), `quantity`, `unit_price`
- **`customer_reviews`**: `id` (PK), `customer_id` (FK $\to$ `customers.id`), `product_id` (FK $\to$ `products.id`), `rating`, `comment`, `review_date`
