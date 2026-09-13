import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import deque
from pydantic import BaseModel, Field


class Dimension(BaseModel):
    name: str
    table: str
    column: str
    data_type: str = "TEXT"
    description: str = ""


class Metric(BaseModel):
    name: str
    table: str
    aggregation: str  # "SUM", "COUNT", "AVG", "MIN", "MAX", etc.
    expression: str   # e.g. "orders.total_amount" or "order_items.quantity * order_items.unit_price"
    default_filter: Optional[str] = None  # e.g. "orders.status != 'CANCELLED'"
    description: str = ""


class JoinEdge(BaseModel):
    from_table: str
    to_table: str
    join_type: str = "JOIN"  # "JOIN", "LEFT JOIN"
    condition: str          # e.g. "orders.customer_id = customers.id"


class DimensionFilter(BaseModel):
    dimension: str
    operator: str = "="     # "=", "!=", ">", "<", ">=", "<=", "LIKE", "IN"
    value: Any


class OrderBy(BaseModel):
    field: str
    direction: str = "DESC" # "ASC", "DESC"


class SemanticQuery(BaseModel):
    metrics: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    filters: List[DimensionFilter] = Field(default_factory=list)
    order_by: List[OrderBy] = Field(default_factory=list)
    limit: Optional[int] = 100


class SemanticCatalog:
    """
    Semantic Model Catalog holding business metrics, dimensions, and table join topologies.
    """

    def __init__(self):
        self.dimensions: Dict[str, Dimension] = {}
        self.metrics: Dict[str, Metric] = {}
        self.joins: List[JoinEdge] = []
        self._graph: Dict[str, List[Tuple[str, JoinEdge]]] = {}

    def register_dimension(self, dim: Dimension):
        self.dimensions[dim.name] = dim
        # Also allow lookup by table.column
        full_key = f"{dim.table}.{dim.column}"
        if full_key not in self.dimensions:
            self.dimensions[full_key] = dim

    def register_metric(self, metric: Metric):
        self.metrics[metric.name] = metric

    def register_join(self, edge: JoinEdge):
        self.joins.append(edge)
        if edge.from_table not in self._graph:
            self._graph[edge.from_table] = []
        if edge.to_table not in self._graph:
            self._graph[edge.to_table] = []

        # Undirected navigation for shortest join path resolution
        self._graph[edge.from_table].append((edge.to_table, edge))
        # Reverse edge
        reverse_edge = JoinEdge(
            from_table=edge.to_table,
            to_table=edge.from_table,
            join_type=edge.join_type,
            condition=edge.condition
        )
        self._graph[edge.to_table].append((edge.from_table, reverse_edge))

    def resolve_dimension(self, name_or_col: str) -> Dimension:
        if name_or_col in self.dimensions:
            return self.dimensions[name_or_col]
        # Check by column name only
        for dim in self.dimensions.values():
            if dim.column == name_or_col or dim.name == name_or_col:
                return dim
        raise KeyError(f"Dimension '{name_or_col}' not found in semantic catalog.")

    def resolve_metric(self, name: str) -> Metric:
        if name in self.metrics:
            return self.metrics[name]
        raise KeyError(f"Metric '{name}' not found in semantic catalog.")

    def find_shortest_path(self, start_table: str, target_table: str) -> List[JoinEdge]:
        """Finds shortest join path between two tables using Breadth-First Search."""
        if start_table == target_table:
            return []

        queue = deque([(start_table, [])])
        visited = {start_table}

        while queue:
            current, path = queue.popleft()
            if current == target_table:
                return path

            for neighbor, edge in self._graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [edge]))

        raise ValueError(f"No join path found connecting '{start_table}' and '{target_table}'.")

    def resolve_join_tree(self, root_table: str, required_tables: Set[str]) -> List[JoinEdge]:
        """
        Builds a minimal spanning join sequence starting from root_table to connect all required_tables.
        """
        joined_tables = {root_table}
        active_joins: List[JoinEdge] = []
        tables_to_connect = set(required_tables) - {root_table}

        while tables_to_connect:
            best_path = None
            best_target = None
            min_len = float("inf")

            for target in tables_to_connect:
                # Find shortest path from ANY already joined table to target
                for joined in joined_tables:
                    try:
                        p = self.find_shortest_path(joined, target)
                        if 0 < len(p) < min_len:
                            min_len = len(p)
                            best_path = p
                            best_target = target
                    except ValueError:
                        continue

            if not best_path:
                raise ValueError(f"Could not connect remaining tables {tables_to_connect} to {joined_tables}")

            for edge in best_path:
                # If to_table not joined yet, add to joins
                if edge.to_table not in joined_tables:
                    active_joins.append(edge)
                    joined_tables.add(edge.to_table)
                elif edge.from_table not in joined_tables:
                    active_joins.append(edge)
                    joined_tables.add(edge.from_table)

            tables_to_connect -= joined_tables

        return active_joins


class SemanticCompiler:
    """
    Compiles a high-level SemanticQuery into dialect-compliant, deterministic SQL.
    Guarantees zero join hallucinations and enforces standard metric definitions.
    """

    def __init__(self, catalog: SemanticCatalog):
        self.catalog = catalog

    def compile(self, sem_query: SemanticQuery) -> str:
        # 1. Resolve all requested metrics & dimensions
        metric_objs: List[Metric] = [self.catalog.resolve_metric(m) for m in sem_query.metrics]
        dim_objs: List[Dimension] = [self.catalog.resolve_dimension(d) for d in sem_query.dimensions]

        # 2. Collect all required tables
        required_tables: Set[str] = set()
        for m in metric_objs:
            required_tables.add(m.table)
        for d in dim_objs:
            required_tables.add(d.table)
        for f in sem_query.filters:
            f_dim = self.catalog.resolve_dimension(f.dimension)
            required_tables.add(f_dim.table)

        if not required_tables:
            raise ValueError("SemanticQuery requires at least one metric or dimension.")

        # 3. Determine base (root) table
        # Prioritize metric table if available, else first dimension table
        if metric_objs:
            root_table = metric_objs[0].table
        else:
            root_table = dim_objs[0].table

        # 4. Resolve minimal join path
        join_edges = self.catalog.resolve_join_tree(root_table, required_tables)

        # 5. Build SELECT clause
        select_parts = []
        for d in dim_objs:
            select_parts.append(f"{d.table}.{d.column} AS {d.name.replace('.', '_')}")

        for m in metric_objs:
            if m.aggregation.upper() == "COUNT" and m.expression.strip() == "*":
                agg_expr = "COUNT(*)"
            elif m.aggregation.upper() in ("SUM", "AVG", "MIN", "MAX", "COUNT"):
                agg_expr = f"{m.aggregation.upper()}({m.expression})"
            else:
                agg_expr = m.expression
            select_parts.append(f"{agg_expr} AS {m.name}")

        select_sql = "SELECT\n  " + ",\n  ".join(select_parts)

        # 6. Build FROM & JOIN clauses
        from_sql = f"FROM {root_table}"
        join_lines = []
        for edge in join_edges:
            # edge.to_table is joined onto joined_tables
            join_lines.append(f"{edge.join_type} {edge.to_table} ON {edge.condition}")
        join_sql = "\n".join(join_lines)

        # 7. Build WHERE clause (metric default filters + user filters)
        where_conditions: List[str] = []
        for m in metric_objs:
            if m.default_filter and m.default_filter not in where_conditions:
                where_conditions.append(m.default_filter)

        for f in sem_query.filters:
            f_dim = self.catalog.resolve_dimension(f.dimension)
            if isinstance(f.value, str):
                val_repr = f"'{f.value}'"
            elif f.value is None:
                val_repr = "NULL"
            else:
                val_repr = str(f.value)

            if f.operator.upper() == "LIKE":
                where_conditions.append(f"{f_dim.table}.{f_dim.column} LIKE {val_repr}")
            elif f.operator.upper() == "IN" and isinstance(f.value, (list, tuple)):
                items = ", ".join(f"'{x}'" if isinstance(x, str) else str(x) for x in f.value)
                where_conditions.append(f"{f_dim.table}.{f_dim.column} IN ({items})")
            else:
                where_conditions.append(f"{f_dim.table}.{f_dim.column} {f.operator} {val_repr}")

        where_sql = ""
        if where_conditions:
            where_sql = "WHERE\n  " + "\n  AND ".join(where_conditions)

        # 8. Build GROUP BY clause
        group_sql = ""
        if metric_objs and dim_objs:
            group_cols = [f"{d.table}.{d.column}" for d in dim_objs]
            group_sql = "GROUP BY\n  " + ", ".join(group_cols)

        # 9. Build ORDER BY clause
        order_sql = ""
        if sem_query.order_by:
            order_parts = [f"{o.field} {o.direction.upper()}" for o in sem_query.order_by]
            order_sql = "ORDER BY " + ", ".join(order_parts)
        elif metric_objs and dim_objs:
            # Default order by first metric descending
            order_sql = f"ORDER BY {metric_objs[0].name} DESC"

        # 10. Build LIMIT
        limit_sql = f"LIMIT {sem_query.limit}" if sem_query.limit else ""

        # Assemble full query
        components = [select_sql, from_sql]
        if join_sql:
            components.append(join_sql)
        if where_sql:
            components.append(where_sql)
        if group_sql:
            components.append(group_sql)
        if order_sql:
            components.append(order_sql)
        if limit_sql:
            components.append(limit_sql)

        return "\n".join(components)


def get_default_ecommerce_catalog() -> SemanticCatalog:
    """
    Creates and populates the production semantic catalog for ecommerce.db.
    """
    cat = SemanticCatalog()

    # Dimensions
    cat.register_dimension(Dimension(name="customer_id", table="customers", column="id", description="客户ID"))
    cat.register_dimension(Dimension(name="customer_name", table="customers", column="name", description="客户姓名"))
    cat.register_dimension(Dimension(name="city", table="customers", column="city", description="客户所在城市"))
    cat.register_dimension(Dimension(name="email", table="customers", column="email", description="客户邮箱"))

    cat.register_dimension(Dimension(name="product_id", table="products", column="id", description="商品ID"))
    cat.register_dimension(Dimension(name="product_name", table="products", column="product_name", description="商品名称"))
    cat.register_dimension(Dimension(name="category", table="products", column="category", description="商品类目"))
    cat.register_dimension(Dimension(name="price", table="products", column="price", data_type="REAL", description="商品单价"))

    cat.register_dimension(Dimension(name="order_id", table="orders", column="id", description="订单ID"))
    cat.register_dimension(Dimension(name="order_date", table="orders", column="order_date", description="下单时间"))
    cat.register_dimension(Dimension(name="order_status", table="orders", column="status", description="订单状态"))

    # Metrics
    cat.register_metric(Metric(
        name="total_revenue",
        table="orders",
        aggregation="SUM",
        expression="orders.total_amount",
        default_filter="orders.status != 'CANCELLED'",
        description="成交总额 (剔除已取消订单)"
    ))

    cat.register_metric(Metric(
        name="order_count",
        table="orders",
        aggregation="COUNT",
        expression="DISTINCT orders.id",
        description="总订单笔数"
    ))

    cat.register_metric(Metric(
        name="avg_order_value",
        table="orders",
        aggregation="AVG",
        expression="orders.total_amount",
        default_filter="orders.status != 'CANCELLED'",
        description="平均客单价"
    ))

    cat.register_metric(Metric(
        name="total_units_sold",
        table="order_items",
        aggregation="SUM",
        expression="order_items.quantity",
        description="商品销售总件数"
    ))

    cat.register_metric(Metric(
        name="item_sales_amount",
        table="order_items",
        aggregation="SUM",
        expression="order_items.quantity * order_items.unit_price",
        description="明细销售总金额"
    ))

    cat.register_metric(Metric(
        name="customer_count",
        table="customers",
        aggregation="COUNT",
        expression="DISTINCT customers.id",
        description="注册客户总量"
    ))

    cat.register_metric(Metric(
        name="avg_review_rating",
        table="customer_reviews",
        aggregation="AVG",
        expression="customer_reviews.rating",
        description="商品平均评分"
    ))

    # Join Topologies
    # orders -> customers
    cat.register_join(JoinEdge(
        from_table="orders",
        to_table="customers",
        join_type="JOIN",
        condition="orders.customer_id = customers.id"
    ))

    # order_items -> orders
    cat.register_join(JoinEdge(
        from_table="order_items",
        to_table="orders",
        join_type="JOIN",
        condition="order_items.order_id = orders.id"
    ))

    # order_items -> products
    cat.register_join(JoinEdge(
        from_table="order_items",
        to_table="products",
        join_type="JOIN",
        condition="order_items.product_id = products.id"
    ))

    # customer_reviews -> customers
    cat.register_join(JoinEdge(
        from_table="customer_reviews",
        to_table="customers",
        join_type="JOIN",
        condition="customer_reviews.customer_id = customers.id"
    ))

    # customer_reviews -> products
    cat.register_join(JoinEdge(
        from_table="customer_reviews",
        to_table="products",
        join_type="JOIN",
        condition="customer_reviews.product_id = products.id"
    ))

    return cat
