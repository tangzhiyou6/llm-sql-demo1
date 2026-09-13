import unittest
import sys
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.explain_guard import ExplainGuard, RiskLevel
from core.consensus_engine import ExecutionConsensusEngine
from core.semantic_layer import (
    get_default_ecommerce_catalog,
    SemanticCompiler,
    SemanticQuery,
    DimensionFilter,
    OrderBy
)
from core.db_sandbox import DBSandbox
from core.ast_guard import SQLASTGuard
from benchmark.sample_db import init_sample_database
from mcp_server import Text2SQLMCPServer
import config


class TestModernText2SQLSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = config.BASE_DIR / "test_modern.db"
        init_sample_database(str(cls.db_path))
        cls.explain_guard = ExplainGuard(block_on_critical=True)
        cls.sandbox = DBSandbox(
            db_path=str(cls.db_path),
            timeout_seconds=2.0,
            explain_guard=cls.explain_guard,
            enforce_explain=True
        )
        cls.ast_guard = SQLASTGuard(default_limit=100, target_dialect="sqlite")
        cls.consensus_engine = ExecutionConsensusEngine(sandbox=cls.sandbox, ast_guard=cls.ast_guard)
        cls.catalog = get_default_ecommerce_catalog()
        cls.compiler = SemanticCompiler(cls.catalog)

    @classmethod
    def tearDownClass(cls):
        if cls.db_path.exists():
            cls.db_path.unlink()

    # =========================================================================
    # P2: Cost-Based Pre-Filter & Cartesian Product Tests
    # =========================================================================
    def test_p2_explain_guard_detects_cartesian_product(self):
        conn = self.sandbox._get_readonly_connection()
        try:
            bad_sql = "SELECT * FROM customers, orders"
            analysis = self.explain_guard.analyze_plan(conn, bad_sql)
            self.assertFalse(analysis.is_safe)
            self.assertEqual(analysis.risk_level, RiskLevel.CRITICAL)
            self.assertTrue(analysis.has_cartesian_product)
            self.assertIn("Cartesian Product", analysis.error_message)
        finally:
            conn.close()

    def test_p2_sandbox_blocks_cartesian_product(self):
        bad_sql = "SELECT * FROM customers, orders"
        res = self.sandbox.execute_query(bad_sql)
        self.assertFalse(res.success)
        self.assertIn("Cost Pre-Filter Intercepted", res.error)
        self.assertIsNotNone(res.plan_analysis)

    def test_p2_explain_guard_allows_indexed_join(self):
        good_sql = "SELECT c.name, o.total_amount FROM customers c JOIN orders o ON c.id = o.customer_id"
        res = self.sandbox.execute_query(good_sql)
        self.assertTrue(res.success)
        self.assertGreater(res.row_count, 0)
        self.assertTrue(res.plan_analysis.is_safe)

    # =========================================================================
    # P1: Execution Consensus & Clustering Tests
    # =========================================================================
    def test_p1_consensus_result_clustering_and_voting(self):
        candidates = [
            "SELECT city, COUNT(*) as cnt FROM customers GROUP BY city ORDER BY cnt DESC",
            "SELECT c.city, count(c.id) FROM customers c GROUP BY c.city ORDER BY 2 DESC",
            "SELECT city, COUNT(*) FROM customers WHERE city = '上海' GROUP BY city",
            "SELECT * FROM customers, orders"  # Disallowed Cartesian product
        ]
        result = self.consensus_engine.evaluate_candidates(candidates)
        self.assertTrue(result.success)
        self.assertEqual(result.total_candidates, 4)
        self.assertEqual(result.valid_candidates, 3)
        self.assertEqual(len(result.clusters), 2)
        # Winning cluster has 2 votes
        self.assertEqual(result.clusters[0].vote_count, 2)
        self.assertAlmostEqual(result.confidence, 0.5, places=2)
        self.assertIn("customers", result.winning_sql)

    def test_p1_consensus_handles_all_failing_candidates(self):
        bad_candidates = [
            "SELECT * FROM customers, orders",
            "SELECT syntax error here",
            "DROP TABLE customers"
        ]
        result = self.consensus_engine.evaluate_candidates(bad_candidates)
        self.assertFalse(result.success)
        self.assertEqual(result.valid_candidates, 0)
        self.assertIn("failed execution", result.error)

    # =========================================================================
    # P3: Semantic Metric Layer Compiler Tests
    # =========================================================================
    def test_p3_semantic_compiler_single_hop_join(self):
        q = SemanticQuery(
            metrics=["total_revenue", "order_count"],
            dimensions=["city"],
            filters=[DimensionFilter(dimension="order_status", operator="=", value="PAID")],
            limit=5
        )
        sql = self.compiler.compile(q)
        self.assertIn("FROM orders", sql)
        self.assertIn("JOIN customers ON orders.customer_id = customers.id", sql)
        self.assertIn("orders.status != 'CANCELLED'", sql)
        self.assertIn("orders.status = 'PAID'", sql)
        self.assertIn("GROUP BY", sql)
        self.assertIn("LIMIT 5", sql)

        # Execution in sandbox
        res = self.sandbox.execute_query(sql)
        self.assertTrue(res.success)
        self.assertGreater(res.row_count, 0)

    def test_p3_semantic_compiler_multi_hop_join(self):
        # order_items -> orders -> customers
        q = SemanticQuery(
            metrics=["total_units_sold"],
            dimensions=["city"],
            limit=10
        )
        sql = self.compiler.compile(q)
        self.assertIn("FROM order_items", sql)
        self.assertIn("JOIN orders ON order_items.order_id = orders.id", sql)
        self.assertIn("JOIN customers ON orders.customer_id = customers.id", sql)

        res = self.sandbox.execute_query(sql)
        self.assertTrue(res.success)
        self.assertGreater(res.row_count, 0)

    # =========================================================================
    # P0: MCP / Persistent Daemon In-Memory Execution
    # =========================================================================
    def test_p0_mcp_server_in_memory_tools(self):
        server = Text2SQLMCPServer(db_path=str(self.db_path))

        # 1. tools/list
        list_req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        resp = server.handle_request(list_req)
        tool_names = [t["name"] for t in resp["result"]["tools"]]
        self.assertIn("text_to_sql", tool_names)
        self.assertIn("semantic_query", tool_names)
        self.assertIn("explain_query", tool_names)
        self.assertIn("consensus_sampling", tool_names)
        self.assertIn("execute_sql_in_sandbox", tool_names)

        # 2. tools/call -> semantic_query
        call_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "semantic_query",
                "arguments": {
                    "metrics": ["customer_count"],
                    "dimensions": ["city"]
                }
            }
        }
        resp2 = server.handle_request(call_req)
        self.assertFalse(resp2["result"]["isError"])
        data = resp2["result"]["data"]
        self.assertTrue(data["success"])
        self.assertGreater(data["row_count"], 0)


if __name__ == "__main__":
    unittest.main()
