import { defineTool } from "@deepseek-ai/dsh-tools";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { resolve } from "node:path";

const execFileAsync = promisify(execFile);

const name = "plugin-text-to-sql";
const inject = ["tools"];

const PROJECT_DIR = "/Users/tangzhiyou/study-codes/llm-sql-demo1";
const PYTHON_PATH = resolve(PROJECT_DIR, ".venv/bin/python");
const BRIDGE_PATH = resolve(PROJECT_DIR, "core/plugin_bridge.py");

async function callPythonBridge(toolName, args) {
  try {
    const { stdout } = await execFileAsync(
      PYTHON_PATH,
      [BRIDGE_PATH, "--tool", toolName, "--args", JSON.stringify(args)],
      { cwd: PROJECT_DIR, timeout: 30000 }
    );
    return JSON.parse(stdout.trim());
  } catch (err) {
    return {
      success: false,
      error: err.stdout ? JSON.parse(err.stdout).error : err.message
    };
  }
}

export function apply(ctx) {
  // 1. All-in-one text_to_sql pipeline tool
  ctx.tools.register(defineTool({
    name: "text_to_sql",
    description: "Enterprise Text-to-SQL tool. Converts natural language queries into verified SQL through Plan-then-Solve workflow, AST safety checks, dynamic LIMIT 100 rewrite, and executes in a read-only sandbox.",
    parameters: {
      query: {
        type: "string",
        required: true,
        description: "Natural language query about business metrics, orders, customers, or products."
      }
    },
    output: {
      schema: { type: "object", additionalProperties: true },
      render: (_args, val) => [{ type: "text", text: typeof val === "string" ? val : JSON.stringify(val, null, 2) }]
    },
    async execute(args) {
      return await callPythonBridge("text_to_sql", args);
    }
  }));

  // 2. execute_sql_in_sandbox tool
  ctx.tools.register(defineTool({
    name: "execute_sql_in_sandbox",
    description: "Executes candidate SQL query in the hardened read-only database sandbox with AST validation (SELECT only), dynamic LIMIT 100 rewrite, and 5-second timeout.",
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
      return await callPythonBridge("execute_sql_in_sandbox", args);
    }
  }));

  // 3. schema_search tool
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
      return await callPythonBridge("schema_search", args);
    }
  }));

  // 4. value_lookup tool
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
      return await callPythonBridge("value_lookup", args);
    }
  }));
}

export { name, inject };
