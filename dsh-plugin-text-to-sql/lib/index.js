import { defineTool } from "@deepseek-ai/dsh-tools";
import { spawn } from "node:child_process";
import * as readline from "node:readline";
import { resolve } from "node:path";

const name = "plugin-text-to-sql";
const inject = ["tools"];

const PROJECT_DIR = "/Users/tangzhiyou/study-codes/llm-sql-demo1";
const PYTHON_PATH = resolve(PROJECT_DIR, ".venv/bin/python");
const DAEMON_SCRIPT = resolve(PROJECT_DIR, "mcp_server.py");

/**
 * P0 Architecture: Persistent Stdio Daemon Client.
 * Maintains a persistent Python MCP process over stdin/stdout,
 * eliminating the 300ms Python cold-start latency down to <5ms per call.
 */
class PythonDaemonClient {
  constructor(pythonPath, scriptPath, cwd) {
    this.pythonPath = pythonPath;
    this.scriptPath = scriptPath;
    this.cwd = cwd;
    this.proc = null;
    this.rl = null;
    this.pending = new Map();
    this.seqId = 1;
  }

  ensureStarted() {
    if (this.proc && !this.proc.killed) {
      return;
    }

    this.proc = spawn(this.pythonPath, [this.scriptPath], {
      cwd: this.cwd,
      stdio: ["pipe", "pipe", "inherit"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" }
    });

    this.rl = readline.createInterface({
      input: this.proc.stdout,
      terminal: false
    });

    this.rl.on("line", (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      try {
        const resp = JSON.parse(trimmed);
        if (resp.id && this.pending.has(resp.id)) {
          const { resolve, reject, timer } = this.pending.get(resp.id);
          clearTimeout(timer);
          this.pending.delete(resp.id);

          if (resp.error) {
            reject(new Error(resp.error.message || JSON.stringify(resp.error)));
          } else {
            const data = resp.result?.data !== undefined ? resp.result.data : resp.result;
            resolve(data);
          }
        }
      } catch (_err) {
        // Ignore unparseable stream noise
      }
    });

    this.proc.on("exit", (code) => {
      for (const [id, { reject, timer }] of this.pending) {
        clearTimeout(timer);
        reject(new Error(`Python daemon closed with exit code ${code}`));
      }
      this.pending.clear();
      this.proc = null;
    });

    process.on("exit", () => {
      this.stop();
    });
  }

  async callTool(toolName, args, timeoutMs = 45000) {
    this.ensureStarted();
    const id = this.seqId++;

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`Tool '${toolName}' call timed out after ${timeoutMs}ms`));
        }
      }, timeoutMs);

      this.pending.set(id, { resolve, reject, timer });

      const request = {
        jsonrpc: "2.0",
        id,
        method: "tools/call",
        params: {
          name: toolName,
          arguments: args
        }
      };

      try {
        this.proc.stdin.write(JSON.stringify(request) + "\n");
      } catch (err) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(err);
      }
    });
  }

  stop() {
    if (this.proc && !this.proc.killed) {
      this.proc.kill();
      this.proc = null;
    }
  }
}

const daemon = new PythonDaemonClient(PYTHON_PATH, DAEMON_SCRIPT, PROJECT_DIR);

export function apply(ctx) {
  // 1. All-in-one text_to_sql pipeline tool (with optional consensus)
  ctx.tools.register(defineTool({
    name: "text_to_sql",
    description: "Enterprise Text-to-SQL pipeline. Converts natural language queries into verified SQL through Plan-then-Solve workflow, AST safety checks, ExplainGuard pre-filtering, and executes in a read-only sandbox.",
    parameters: {
      query: {
        type: "string",
        required: true,
        description: "Natural language query about business metrics, orders, customers, or products."
      },
      use_consensus: {
        type: "boolean",
        description: "Enable multi-candidate execution consensus clustering (P1)."
      }
    },
    output: {
      schema: { type: "object", additionalProperties: true },
      render: (_args, val) => [{ type: "text", text: typeof val === "string" ? val : JSON.stringify(val, null, 2) }]
    },
    async execute(args) {
      return await daemon.callTool("text_to_sql", args);
    }
  }));

  // 2. Semantic Metric Layer Compiler tool (P3)
  ctx.tools.register(defineTool({
    name: "semantic_query",
    description: "Deterministic Semantic Metric Layer query compiler (P3). Guarantees zero join hallucinations by compiling business metrics and dimensions using topology graph traversal.",
    parameters: {
      metrics: {
        type: "array",
        items: { type: "string" },
        required: true,
        description: "Metrics: total_revenue, order_count, avg_order_value, total_units_sold, item_sales_amount, customer_count, avg_review_rating"
      },
      dimensions: {
        type: "array",
        items: { type: "string" },
        description: "Dimensions: city, customer_name, category, product_name, order_status, order_date"
      },
      filters: {
        type: "array",
        items: { type: "object", additionalProperties: true },
        description: "Filters: e.g. [{'dimension': 'order_status', 'operator': '=', 'value': 'PAID'}]"
      },
      limit: {
        type: "integer",
        description: "Row limit (default: 100)"
      }
    },
    output: {
      schema: { type: "object", additionalProperties: true },
      render: (_args, val) => [{ type: "text", text: typeof val === "string" ? val : JSON.stringify(val, null, 2) }]
    },
    async execute(args) {
      return await daemon.callTool("semantic_query", args);
    }
  }));

  // 3. Explain Query Cost Pre-filter tool (P2)
  ctx.tools.register(defineTool({
    name: "explain_query",
    description: "Cost-based pre-filter analyzer (P2). Runs EXPLAIN QUERY PLAN to detect Cartesian products, unindexed table scans, and temporary B-trees before execution.",
    parameters: {
      sql: {
        type: "string",
        required: true,
        description: "Candidate SQL to inspect for query plan risks."
      }
    },
    output: {
      schema: { type: "object", additionalProperties: true },
      render: (_args, val) => [{ type: "text", text: typeof val === "string" ? val : JSON.stringify(val, null, 2) }]
    },
    async execute(args) {
      return await daemon.callTool("explain_query", args);
    }
  }));

  // 4. Consensus Sampling tool (P1)
  ctx.tools.register(defineTool({
    name: "consensus_sampling",
    description: "Execution Consensus & Result Set Clustering engine (P1). Executes diverse SQL candidates in sandbox, clusters by normalized result sets, and returns the consensus winner.",
    parameters: {
      candidate_sqls: {
        type: "array",
        items: { type: "string" },
        required: true,
        description: "List of candidate SQL strings to vote on."
      }
    },
    output: {
      schema: { type: "object", additionalProperties: true },
      render: (_args, val) => [{ type: "text", text: typeof val === "string" ? val : JSON.stringify(val, null, 2) }]
    },
    async execute(args) {
      return await daemon.callTool("consensus_sampling", args);
    }
  }));

  // 5. execute_sql_in_sandbox tool
  ctx.tools.register(defineTool({
    name: "execute_sql_in_sandbox",
    description: "Executes candidate SQL query in the hardened read-only database sandbox with AST validation (SELECT only), ExplainGuard pre-filtering (P2), and dynamic LIMIT 100 rewrite.",
    parameters: {
      sql: {
        type: "string",
        required: true,
        description: "The SQL statement to validate and execute."
      }
    },
    output: {
      schema: { type: "object", additionalProperties: true },
      render: (_args, val) => [{ type: "text", text: typeof val === "string" ? val : JSON.stringify(val, null, 2) }]
    },
    async execute(args) {
      return await daemon.callTool("execute_sql_in_sandbox", args);
    }
  }));

  // 6. schema_search tool
  ctx.tools.register(defineTool({
    name: "schema_search",
    description: "Two-level hybrid retrieval tool to find relevant tables and pruned column metadata for a query.",
    parameters: {
      query: {
        type: "string",
        required: true,
        description: "Keywords or question concepts to search in metadata."
      },
      top_k: {
        type: "integer",
        description: "Number of tables to retrieve (default: 5)."
      }
    },
    output: {
      schema: { type: "object", additionalProperties: true },
      render: (_args, val) => [{ type: "text", text: typeof val === "string" ? val : JSON.stringify(val, null, 2) }]
    },
    async execute(args) {
      return await daemon.callTool("schema_search", args);
    }
  }));

  // 7. value_lookup tool
  ctx.tools.register(defineTool({
    name: "value_lookup",
    description: "Inverted index value lookup for high-frequency categorical dimension values (e.g. status codes, product categories).",
    parameters: {
      search_term: {
        type: "string",
        required: true,
        description: "Dimension concept or term mentioned in query."
      }
    },
    output: {
      schema: { type: "object", additionalProperties: true },
      render: (_args, val) => [{ type: "text", text: typeof val === "string" ? val : JSON.stringify(val, null, 2) }]
    },
    async execute(args) {
      return await daemon.callTool("value_lookup", args);
    }
  }));
}

export { name, inject, daemon };
