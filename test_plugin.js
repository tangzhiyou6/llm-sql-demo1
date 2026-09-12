/**
 * Automated test suite for dsh-plugin-text-to-sql.
 * Tests plugin registration, tool definitions, python bridge, and sandbox security.
 */

import assert from "node:assert/strict";
import { resolve } from "node:path";

const PLUGIN_PATH = resolve(process.cwd(), "dsh-plugin-text-to-sql/lib/index.js");

console.log("========================================================================");
console.log("             Testing dsh-plugin-text-to-sql Custom Plugin               ");
console.log("========================================================================");

async function runTests() {
  // 1. Load plugin module
  console.log("\n[Test 1] Importing plugin module...");
  const plugin = await import(PLUGIN_PATH);
  assert.equal(plugin.name, "plugin-text-to-sql", "Plugin name must be plugin-text-to-sql");
  assert.deepEqual(plugin.inject, ["tools"], "Plugin must inject ['tools']");
  console.log("  ✓ Plugin metadata & inject contract verified.");

  // 2. Mock Cordis Context and apply plugin
  console.log("\n[Test 2] Registering tools on Cordis ctx.tools...");
  const registeredTools = [];
  const mockCtx = {
    tools: {
      register: (tool) => {
        registeredTools.push(tool);
      }
    }
  };
  plugin.apply(mockCtx);

  const toolNames = registeredTools.map((t) => t.name);
  console.log("  ✓ Registered tools:", toolNames);
  assert.ok(toolNames.includes("text_to_sql"), "Must register text_to_sql");
  assert.ok(toolNames.includes("execute_sql_in_sandbox"), "Must register execute_sql_in_sandbox");
  assert.ok(toolNames.includes("schema_search"), "Must register schema_search");
  assert.ok(toolNames.includes("value_lookup"), "Must register value_lookup");

  // 3. Test text_to_sql (All-in-one pipeline)
  console.log("\n[Test 3] Testing 'text_to_sql' tool (End-to-End)...");
  const text2sqlTool = registeredTools.find((t) => t.name === "text_to_sql");
  const t2sRes = await text2sqlTool.execute({
    query: "查询所有处于PAID状态的订单ID和金额"
  });
  console.log("  • Success     :", t2sRes.success);
  console.log("  • Rewritten SQL:\n   ", t2sRes.rewritten_sql.replace(/\n/g, "\n    "));
  console.log("  • Rows returned:", t2sRes.execution_result?.row_count);
  assert.equal(t2sRes.success, true, "Pipeline execution must succeed");
  assert.ok(t2sRes.execution_result.row_count > 0, "Must return rows");
  console.log("  ✓ text_to_sql pipeline passed.");

  // 4. Test execute_sql_in_sandbox (SELECT + Dynamic LIMIT)
  console.log("\n[Test 4] Testing 'execute_sql_in_sandbox' (Safe SELECT)...");
  const sandboxTool = registeredTools.find((t) => t.name === "execute_sql_in_sandbox");
  const sbRes = await sandboxTool.execute({
    sql: "SELECT product_name, price FROM products WHERE price > 50;"
  });
  console.log("  • Columns:", sbRes.columns);
  console.log("  • Rows   :", sbRes.rows);
  assert.equal(sbRes.success, true);
  assert.equal(sbRes.row_count, 2);
  assert.ok(sbRes.rewritten_sql.includes("LIMIT 100"), "Must inject LIMIT 100");
  console.log("  ✓ execute_sql_in_sandbox SELECT passed with LIMIT injection.");

  // 5. Test execute_sql_in_sandbox security rejection (DROP TABLE)
  console.log("\n[Test 5] Testing 'execute_sql_in_sandbox' security rejection (DROP TABLE)...");
  const dropRes = await sandboxTool.execute({
    sql: "DROP TABLE products;"
  });
  console.log("  • Status :", dropRes.success ? "UNEXPECTED_PASS" : "BLOCKED");
  console.log("  • Error  :", dropRes.error);
  assert.equal(dropRes.success, false, "DROP TABLE must be blocked");
  assert.ok(dropRes.error.includes("Security violation") || dropRes.error.includes("SELECT"), "Must raise security violation");
  console.log("  ✓ Security rejection verified: High-risk statement intercepted.");

  // 6. Test schema_search
  console.log("\n[Test 6] Testing 'schema_search' tool...");
  const schemaTool = registeredTools.find((t) => t.name === "schema_search");
  const scRes = await schemaTool.execute({
    query: "订单与客户",
    top_k: 3
  });
  console.log("  • Candidate tables:", scRes.candidate_tables);
  assert.ok(scRes.candidate_tables.length > 0, "Must return candidate tables");
  console.log("  ✓ schema_search passed.");

  // 7. Test value_lookup
  console.log("\n[Test 7] Testing 'value_lookup' tool (Enum/Synonym mapping)...");
  const valTool = registeredTools.find((t) => t.name === "value_lookup");
  const valRes = await valTool.execute({
    search_term: "果汁",
    top_k: 5
  });
  console.log("  • Matches:", valRes.matches.map((m) => `${m.matched_term} -> ${m.filter_hint}`));
  assert.ok(valRes.matches.length > 0, "Must match '果汁'");
  assert.ok(valRes.matches.some((m) => m.db_value === "100%纯果汁"), "Must map to 100%纯果汁");
  console.log("  ✓ value_lookup passed: Concept '果汁' mapped to '100%纯果汁'.");

  console.log("\n========================================================================");
  console.log("  🎉 ALL 7 PLUGIN TESTS PASSED SUCCESSFULLY!");
  console.log("========================================================================\n");
}

runTests().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
