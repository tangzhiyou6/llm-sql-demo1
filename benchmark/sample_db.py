import sqlite3
from pathlib import Path
from typing import Tuple
from core.hybrid_retriever import HybridRetriever, TableMetadata, ColumnMetadata
from core.value_lookup import ValueLookupEngine


def init_sample_database(db_path: str) -> None:
    """Initializes a realistic e-commerce SQLite database with foreign keys and sample rows."""
    p = Path(db_path)
    if p.exists():
        p.unlink()

    conn = sqlite3.connect(p)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. customers
    cursor.execute("""
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        city TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at TEXT NOT NULL
    );
    """)

    # 2. products
    cursor.execute("""
    CREATE TABLE products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        stock INTEGER NOT NULL
    );
    """)

    # 3. orders
    cursor.execute("""
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        order_date TEXT NOT NULL,
        status TEXT NOT NULL,
        total_amount REAL NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    );
    """)

    # 4. order_items
    cursor.execute("""
    CREATE TABLE order_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    """)

    # 5. customer_reviews
    cursor.execute("""
    CREATE TABLE customer_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        rating INTEGER NOT NULL,
        comment TEXT,
        review_date TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    """)

    # Seed data
    customers = [
        (1, "张伟", "北京", "zhangwei@example.com", "2023-01-15"),
        (2, "李娜", "上海", "lina@example.com", "2023-02-20"),
        (3, "王强", "深圳", "wangqiang@example.com", "2023-03-10"),
        (4, "赵敏", "杭州", "zhaomin@example.com", "2023-04-05"),
        (5, "陈杰", "北京", "chenjie@example.com", "2023-05-18"),
        (6, "刘芳", "广州", "liufang@example.com", "2023-06-22"),
        (7, "杨洋", "上海", "yangyang@example.com", "2023-07-01"),
        (8, "孙丽", "成都", "sunli@example.com", "2023-08-14"),
    ]
    cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?);", customers)

    products = [
        (1, "100%纯果汁", "果汁饮料", 18.5, 250),
        (2, "低糖乌龙茶", "茶饮料", 6.0, 500),
        (3, "手冲精品咖啡豆", "咖啡", 88.0, 120),
        (4, "无糖苏打水", "碳酸饮料", 4.5, 600),
        (5, "有机燕麦乳", "乳制品", 22.0, 300),
        (6, "鲜橙复合果汁", "果汁饮料", 15.0, 180),
        (7, "冻干黑咖啡粉", "咖啡", 45.0, 200),
        (8, "特级龙井茶", "茶饮料", 158.0, 50),
    ]
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?);", products)

    orders = [
        (101, 1, "2024-01-10", "PAID", 224.5),
        (102, 2, "2024-01-12", "PAID", 88.0),
        (103, 3, "2024-01-15", "CANCELLED", 45.0),
        (104, 1, "2024-02-01", "PAID", 107.0),
        (105, 4, "2024-02-14", "SHIPPED", 158.0),
        (106, 5, "2024-02-20", "PAID", 37.0),
        (107, 2, "2024-03-05", "PAID", 176.0),
        (108, 6, "2024-03-12", "PENDING", 22.0),
        (109, 7, "2024-03-18", "PAID", 90.0),
        (110, 8, "2024-03-25", "PAID", 316.0),
    ]
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?);", orders)

    order_items = [
        (1, 101, 1, 5, 18.5),
        (2, 101, 3, 1, 88.0),
        (3, 101, 5, 2, 22.0),
        (4, 102, 3, 1, 88.0),
        (5, 103, 7, 1, 45.0),
        (6, 104, 1, 4, 18.5),
        (7, 104, 2, 5, 6.0),
        (8, 105, 8, 1, 158.0),
        (9, 106, 1, 2, 18.5),
        (10, 107, 3, 2, 88.0),
        (11, 108, 5, 1, 22.0),
        (12, 109, 6, 6, 15.0),
        (13, 110, 8, 2, 158.0),
    ]
    cursor.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?);", order_items)

    reviews = [
        (1, 1, 1, 5, "果汁非常纯正，口感极佳！", "2024-01-15"),
        (2, 2, 3, 5, "咖啡豆香气浓郁，回味甘甜", "2024-01-18"),
        (3, 1, 3, 4, "烘焙度适中，油脂丰富", "2024-02-05"),
        (4, 4, 8, 5, "特级龙井，清香扑鼻", "2024-02-20"),
        (5, 5, 1, 5, "100%纯果汁确实好喝，推荐", "2024-02-25"),
        (6, 7, 6, 3, "橙汁略偏酸，一般", "2024-03-22"),
    ]
    cursor.executemany("INSERT INTO customer_reviews VALUES (?, ?, ?, ?, ?, ?);", reviews)

    conn.commit()
    conn.close()


def setup_metadata_and_valuelookup(db_path: str) -> Tuple[HybridRetriever, ValueLookupEngine]:
    """Builds two-level metadata and independent value lookup index for the e-commerce database."""
    retriever = HybridRetriever()
    val_lookup = ValueLookupEngine()

    # Table 1: customers
    customers_meta = TableMetadata(
        table_name="customers",
        business_name="客户表",
        description="存储注册客户的基本档案，包括姓名、所在城市、邮箱与注册日期",
        columns=[
            ColumnMetadata(name="id", data_type="INTEGER", business_name="客户ID", description="主键唯一标识", is_primary_key=True),
            ColumnMetadata(name="name", data_type="TEXT", business_name="客户姓名", description="客户真实姓名", sample_values=["张伟", "李娜", "王强"]),
            ColumnMetadata(name="city", data_type="TEXT", business_name="所在城市", description="客户常住城市", sample_values=["北京", "上海", "深圳", "杭州"]),
            ColumnMetadata(name="email", data_type="TEXT", business_name="电子邮箱", description="客户唯一邮箱地址", sample_values=["zhangwei@example.com"]),
            ColumnMetadata(name="created_at", data_type="TEXT", business_name="注册时间", description="账号注册日期", sample_values=["2023-01-15"]),
        ]
    )
    retriever.register_table(customers_meta)

    # Table 2: products
    products_meta = TableMetadata(
        table_name="products",
        business_name="商品表",
        description="存储商城销售的商品目录，包括商品名称、类别、单价和库存量",
        columns=[
            ColumnMetadata(name="id", data_type="INTEGER", business_name="商品ID", description="主键唯一标识", is_primary_key=True),
            ColumnMetadata(name="product_name", data_type="TEXT", business_name="商品名称", description="商品全称", sample_values=["100%纯果汁", "手冲精品咖啡豆"]),
            ColumnMetadata(name="category", data_type="TEXT", business_name="商品品类", description="商品所属分类", sample_values=["果汁饮料", "茶饮料", "咖啡", "乳制品"]),
            ColumnMetadata(name="price", data_type="REAL", business_name="商品售价", description="商品标牌单价", sample_values=[18.5, 88.0, 158.0]),
            ColumnMetadata(name="stock", data_type="INTEGER", business_name="当前库存", description="仓库现存件数", sample_values=[250, 500]),
        ]
    )
    retriever.register_table(products_meta)

    # Table 3: orders
    orders_meta = TableMetadata(
        table_name="orders",
        business_name="订单表",
        description="记录客户提交的交易订单信息，包含支付状态、下单日期与订单总金额",
        columns=[
            ColumnMetadata(name="id", data_type="INTEGER", business_name="订单ID", description="订单主键ID", is_primary_key=True),
            ColumnMetadata(name="customer_id", data_type="INTEGER", business_name="客户ID", description="下单客户外键", is_foreign_key=True, foreign_key_target="customers.id"),
            ColumnMetadata(name="order_date", data_type="TEXT", business_name="下单日期", description="订单创建年月日", sample_values=["2024-01-10", "2024-02-14"]),
            ColumnMetadata(name="status", data_type="TEXT", business_name="订单状态", description="订单履约状态代码", sample_values=["PAID", "CANCELLED", "SHIPPED", "PENDING"]),
            ColumnMetadata(name="total_amount", data_type="REAL", business_name="订单总金额", description="该笔订单结算总价", sample_values=[224.5, 88.0, 316.0]),
        ],
        foreign_keys=[{"from_column": "customer_id", "to_table": "customers", "to_column": "id"}]
    )
    retriever.register_table(orders_meta)

    # Table 4: order_items
    order_items_meta = TableMetadata(
        table_name="order_items",
        business_name="订单商品明细表",
        description="订单所包含的具体商品行明细，记录每个商品的购买数量、成交单价与销售额明细，用于统计商品销售总额与销量",
        columns=[
            ColumnMetadata(name="id", data_type="INTEGER", business_name="明细ID", description="主键ID", is_primary_key=True),
            ColumnMetadata(name="order_id", data_type="INTEGER", business_name="订单ID", description="关联订单ID", is_foreign_key=True, foreign_key_target="orders.id"),
            ColumnMetadata(name="product_id", data_type="INTEGER", business_name="商品ID", description="关联商品ID", is_foreign_key=True, foreign_key_target="products.id"),
            ColumnMetadata(name="quantity", data_type="INTEGER", business_name="购买数量", description="购买件数", sample_values=[1, 2, 5]),
            ColumnMetadata(name="unit_price", data_type="REAL", business_name="成交单价", description="下单时锁定的单价", sample_values=[18.5, 88.0]),
        ],
        foreign_keys=[
            {"from_column": "order_id", "to_table": "orders", "to_column": "id"},
            {"from_column": "product_id", "to_table": "products", "to_column": "id"}
        ]
    )
    retriever.register_table(order_items_meta)

    # Table 5: customer_reviews
    reviews_meta = TableMetadata(
        table_name="customer_reviews",
        business_name="商品评价表",
        description="客户对购买商品的星级打分和评论内容",
        columns=[
            ColumnMetadata(name="id", data_type="INTEGER", business_name="评价ID", description="主键ID", is_primary_key=True),
            ColumnMetadata(name="customer_id", data_type="INTEGER", business_name="客户ID", description="评价人ID", is_foreign_key=True, foreign_key_target="customers.id"),
            ColumnMetadata(name="product_id", data_type="INTEGER", business_name="商品ID", description="被评价商品ID", is_foreign_key=True, foreign_key_target="products.id"),
            ColumnMetadata(name="rating", data_type="INTEGER", business_name="星级评分", description="1-5分评级", sample_values=[1, 2, 3, 4, 5]),
            ColumnMetadata(name="comment", data_type="TEXT", business_name="评价文本", description="客户留言反馈", sample_values=["果汁非常纯正", "香气浓郁"]),
            ColumnMetadata(name="review_date", data_type="TEXT", business_name="评价日期", description="撰写评价日期", sample_values=["2024-01-15"]),
        ],
        foreign_keys=[
            {"from_column": "customer_id", "to_table": "customers", "to_column": "id"},
            {"from_column": "product_id", "to_table": "products", "to_column": "id"}
        ]
    )
    retriever.register_table(reviews_meta)

    # --- Index High-Frequency Categorical Dimension Values ---
    # 1. Cities
    val_lookup.index_column_values("customers", "city", ["北京", "上海", "深圳", "杭州", "广州", "成都"])
    # 2. Status with synonyms
    val_lookup.index_column_values(
        "orders", "status",
        ["PAID", "CANCELLED", "SHIPPED", "PENDING"],
        synonyms_map={
            "PAID": ["已支付", "已付款", "付款完成", "支付成功"],
            "CANCELLED": ["已取消", "作废", "取消订单"],
            "SHIPPED": ["已发货", "运输中"],
            "PENDING": ["待支付", "待付款", "未付款"]
        }
    )
    # 3. Product categories and names
    val_lookup.index_column_values("products", "category", ["果汁饮料", "茶饮料", "咖啡", "碳酸饮料", "乳制品"])
    val_lookup.index_column_values(
        "products", "product_name",
        ["100%纯果汁", "低糖乌龙茶", "手冲精品咖啡豆", "无糖苏打水", "有机燕麦乳", "鲜橙复合果汁", "冻干黑咖啡粉", "特级龙井茶"]
    )

    return retriever, val_lookup
