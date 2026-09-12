import unittest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.ast_guard import SQLASTGuard, ASTValidationError
from core.db_sandbox import DBSandbox
from benchmark.sample_db import init_sample_database
import config


class TestText2SQLGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = config.BASE_DIR / "test_sandbox.db"
        init_sample_database(str(cls.db_path))
        cls.sandbox = DBSandbox(db_path=str(cls.db_path), timeout_seconds=2.0)
        cls.ast_guard = SQLASTGuard(default_limit=100, target_dialect="sqlite")

    @classmethod
    def tearDownClass(cls):
        if cls.db_path.exists():
            cls.db_path.unlink()

    def test_ast_rejects_drop_table(self):
        sql = "DROP TABLE customers;"
        with self.assertRaises(ASTValidationError) as ctx:
            self.ast_guard.validate_and_extract(sql)
        self.assertEqual(ctx.exception.error_type, "DISALLOWED_OPERATION")

    def test_ast_rejects_update_and_delete(self):
        with self.assertRaises(ASTValidationError):
            self.ast_guard.validate_and_extract("UPDATE orders SET total_amount = 0;")
        with self.assertRaises(ASTValidationError):
            self.ast_guard.validate_and_extract("DELETE FROM products WHERE id = 1;")

    def test_ast_catches_table_hallucination(self):
        sql = "SELECT * FROM phantom_table;"
        with self.assertRaises(ASTValidationError) as ctx:
            self.ast_guard.validate_and_extract(sql, allowed_tables={"customers", "orders"})
        self.assertEqual(ctx.exception.error_type, "UNKNOWN_TABLE")

    def test_ast_catches_column_hallucination(self):
        sql = "SELECT customers.phantom_col FROM customers;"
        with self.assertRaises(ASTValidationError) as ctx:
            self.ast_guard.validate_and_extract(
                sql,
                allowed_tables={"customers"},
                allowed_columns_map={"customers": {"id", "name", "city"}}
            )
        self.assertEqual(ctx.exception.error_type, "UNKNOWN_COLUMN")

    def test_limit_injection(self):
        sql = "SELECT id, name FROM customers"
        parsed = self.ast_guard.parse_sql(sql)
        rewritten = self.ast_guard.inject_or_cap_limit(parsed, max_limit=100)
        sql_out = rewritten.sql()
        self.assertIn("LIMIT 100", sql_out)

    def test_limit_capping(self):
        sql = "SELECT id, name FROM customers LIMIT 5000"
        parsed = self.ast_guard.parse_sql(sql)
        rewritten = self.ast_guard.inject_or_cap_limit(parsed, max_limit=100)
        sql_out = rewritten.sql()
        self.assertIn("LIMIT 100", sql_out)
        self.assertNotIn("LIMIT 5000", sql_out)

    def test_db_sandbox_readonly_enforcement(self):
        # Even if someone executes a write directly against sandbox connection, mode=ro refuses it
        res = self.sandbox.execute_query("INSERT INTO customers (name, city, email, created_at) VALUES ('Fake', 'Nowhere', 'fake@test.com', '2024-01-01');")
        self.assertFalse(res.success)
        self.assertIn("read-only", res.error.lower())

    def test_db_sandbox_timeout(self):
        # Recursive CTE creating infinite loop
        runaway_sql = """
        WITH RECURSIVE infinite_loop(x) AS (
            SELECT 1
            UNION ALL
            SELECT x + 1 FROM infinite_loop
        )
        SELECT * FROM infinite_loop;
        """
        quick_sandbox = DBSandbox(db_path=str(self.db_path), timeout_seconds=1.0)
        res = quick_sandbox.execute_query(runaway_sql)
        self.assertFalse(res.success)
        self.assertIn("Timeout", res.error)


if __name__ == "__main__":
    unittest.main()
