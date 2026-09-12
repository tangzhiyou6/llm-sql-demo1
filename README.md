# 工业级 Text-to-SQL 架构工程脚手架 (DeepSeek + sqlglot)

本工程基于**两阶段有约束流水线 (Plan-then-Solve)**、**sqlglot AST 深度改写与安全网**、**两级元数据混合检索 + Value-Lookup** 与 **四维评测雷达基准** 落地构建，杜绝纯 Agent 自由探索引发的死循环与 Token 爆炸，并在底层保障数据库安全只读与超时截断。

---

## 一、 核心架构特性与实现映射

| 业务痛点 | 架构解决方案 | 核心模块实现 |
|---|---|---|
| **Agent 自由探索死循环 / Token 爆炸** | 两阶段流水线：Phase 1 仅产出《推理蓝图》，Phase 2 生成与自愈（严格限制 $\le 2\sim3$ 轮） | [`agent/phase1_planner.py`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/agent/phase1_planner.py)<br>[`agent/phase2_generator.py`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/agent/phase2_generator.py) |
| **单模型成本与推理延迟矛盾** | 双模型分工（Dual-Model Routing）：轻量模型（DeepSeek-V3）负责 Schema 剪枝，深度推理模型（DeepSeek-R1）负责多表 JOIN 与 CoT 自愈 | [`core/llm_router.py`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/core/llm_router.py) |
| **全表扫描拖垮数据库 / 注入破坏风险** | 只读连接池（URI `mode=ro`）硬拒绝所有修改 + Session 级超时（5s）强行截断 | [`core/db_sandbox.py`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/core/db_sandbox.py) |
| **前端大数据量打崩 / 目标数仓语法差异** | AST 动态改写：外层无条件注入/钳制 `LIMIT 100`；sqlglot 统一转译为目标方言 | [`core/ast_guard.py`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/core/ast_guard.py) |
| **幻觉表名与幻觉列名** | AST 结构级白名单校验，并在解析失败时高亮精准行号、列号供模型反思自愈 | [`core/ast_guard.py`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/core/ast_guard.py) |
| **纯向量检索专有名词弱、枚举值匹配缺失** | 混合检索（BM25 + 向量语义 + RRF）两级元数据剪枝 + 独立枚举值 Value-Lookup 倒排索引 | [`core/hybrid_retriever.py`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/core/hybrid_retriever.py)<br>[`core/value_lookup.py`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/core/value_lookup.py) |
| **缺乏针对内部库的科学评测与防回退机制** | 私有黄金测试集（分层 Simple/Moderate/Challenging）与四维雷达网（EX, SR, ERR, Latency） | [`benchmark/evaluator.py`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/benchmark/evaluator.py)<br>[`benchmark/run_benchmark.py`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/benchmark/run_benchmark.py) |

---

## 二、 工程目录结构

```
llm-sql-demo1/
├── config.py                 # 全局配置（超时阈值、Limit 上限、模型路由名称等）
├── requirements.txt          # 项目依赖（sqlglot, pydantic, httpx, python-dotenv）
├── .env.example              # API 密钥与环境变量模板
├── ecommerce.db              # 内置真实电商多表测试数据库
│
├── core/                     # 核心引擎层
│   ├── ast_guard.py          # sqlglot AST 校验、LIMIT 注入、方言转译与高亮定位
│   ├── db_sandbox.py         # 5秒超时截断、强制只读事务与 Schema 采样沙箱
│   ├── hybrid_retriever.py   # 两级元数据混合检索（BM25 + Dense + RRF）与列剪枝
│   ├── value_lookup.py       # 独立高频枚举值与同义词倒排索引（果汁 -> 100%纯果汁）
│   ├── llm_router.py         # 双模型调度（Fast Model vs Deep Reasoning Model）
│   └── deepseek_harness.py   # DeepSeek 原生 Tool Calling Harness 驱动与状态机
│
├── agent/                    # 编排与调度流水线
│   ├── schemas.py            # Pydantic 数据契约（ReasoningBlueprint, PipelineResult）
│   ├── phase1_planner.py     # Phase 1: 只读探索产出《推理蓝图》（表/外键路径/过滤逻辑）
│   ├── phase2_generator.py   # Phase 2: 代码生成、AST 校验改写与自愈（≤3轮）
│   └── pipeline.py           # 完整端到端流水线驱动
│
├── benchmark/                # 评测基准与回归框架
│   ├── sample_db.py          # 电商多表生成与元数据装载器
│   ├── dataset.json          # 私有黄金测试集（15个代表案例，覆盖三层复杂度）
│   ├── evaluator.py          # 四维指标计算（EX, SR, ERR, Latency，支持浮点容差比对）
│   └── run_benchmark.py      # 评测流水线与 Markdown 报告输出
│
├── tests/                    # 单元与防护网回归测试
│   ├── test_guards.py        # 拦截 DROP/UPDATE、超时熔断、LIMIT 改写与只读连接测试
│   └── test_harness.py       # DeepSeek Harness 状态流转与工具调用测试
│
├── harness_runner.py         # DeepSeek Harness 独立运行脚本
└── main.py                   # 交互式命令行与全功能入口
```

---

## 三、 快速上手与运行

### 1. 激活虚拟环境
本项目已创建 `.venv` 环境并安装必要依赖：
```bash
source .venv/bin/activate
```

### 2. 配置大模型 API（可选）
复制 `.env.example` 为 `.env`：
```bash
cp .env.example .env
```
配置你的 `DEEPSEEK_API_KEY`（如未配置，系统内置离线模拟 Fallback 引擎，依然可完整体验 AST 校验、沙箱运行与基准评测）。

### 3. 初始化样例数据库
```bash
python main.py --init-db
```

### 4. 单条自然语言查询执行
```bash
python main.py --query "查询所有处于PAID状态的订单ID和金额"
```
```bash
python main.py --query "查找购买过100%纯果汁的客户姓名和订单日期"
```

控制台将清晰打印流水线五个步骤的输出：
1. **[Value-Lookup]** 实体对齐（如将 `果汁` 精准对齐为 `products.product_name = '100%纯果汁'`）
2. **[Phase 1 Blueprint]** 输出选中的表、外键连接路径与过滤条件
3. **[Phase 2 AST Guard]** 生成 SQL 并无条件注入/钳制 `LIMIT 100`
4. **[Self-Healing]** 自纠错轨迹（若首轮语法或运行时出错，捕获结构化报错并反思自愈）
5. **[DBSandbox Execution]** 只读连接安全执行与延迟统计

### 5. 交互式对话体验
直接运行：
```bash
python main.py
```

### 6. 执行基准评测 (Benchmarking)
```bash
python main.py --benchmark
```
将对 `benchmark/dataset.json` 中的全部用例进行四维指标评测，并在 [`benchmark/BENCHMARK_REPORT.md`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/benchmark/BENCHMARK_REPORT.md) 输出报告。

### 7. 运行安全防护单元测试
```bash
python -m unittest discover tests
```
验证对 `DROP TABLE`、`UPDATE`、表名/列名幻觉、超时熔断、只读数据库写拒绝以及 DeepSeek Harness 状态流转的硬性拦截（11 个单元测试全部通过）。

---

## 四、 在 DeepSeek Harness 中运行

本项目专门封装了符合 OpenAI/DeepSeek 标准的 Agent Tool-Calling Harness（测试与执行挽具）：[`core/deepseek_harness.py`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/core/deepseek_harness.py)。

### 1. 查看 Harness 中注册的标准 DB Tools (JSON Schema)
```bash
python harness_runner.py --tools
```
工具集包含：
- `schema_search` (Phase 1 只读探索)
- `sample_column_values` (Phase 1 采样列数据)
- `value_lookup` (Phase 1 高频枚举倒排检索)
- `submit_reasoning_blueprint` (Phase 1 状态完结并解锁 Phase 2)
- `execute_sql_in_sandbox` (Phase 2 AST 校验 + LIMIT 注入 + 只读超时沙箱执行)

### 2. 通过 DeepSeek Harness 执行单条查询
```bash
python harness_runner.py --query "查询所有处于PAID状态的订单ID和金额"
# 或者使用 main.py 入口
python main.py --harness --query "查询所有处于PAID状态的订单ID和金额"
```

### 3. 进入 DeepSeek Harness 交互模式
```bash
python harness_runner.py
# 或
python main.py --harness
```
进入交互会话后，模型将在 Harness 的驱动下自主调用探索工具、生成蓝图、提交只读沙箱执行，并在出错时触发 DeepSeek-R1 深度自纠错。

---

## 五、 在 dsh (DeepSeek Harness CLI & Web UI) 中运行

你系统上已安装官方的 DeepSeek Harness 命令行工具 `dsh`（`/Users/tangzhiyou/.local/state/fnm_multishells/10768_1789223774496/bin/dsh`）。我们已为你无缝适配了 **原生 Workspace Skill** 与 **标准 MCP Server** 两种运行方式：

### 方式 1：启动 dsh Web UI（自动挂载 Skill，开箱即用）
本项目根目录下已配置好符合 `dsh` 规约的技能：[`.dsh/skills/text-to-sql/SKILL.md`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/.dsh/skills/text-to-sql/SKILL.md)。

只需在当前工程目录下启动：
```bash
dsh web
```
`dsh` 会自动通过 `@deepseek-ai/dsh-skill-filesystem` 扫描并加载当前工程的 `text-to-sql` 技能。在浏览器界面直接提问，Agent 将自动遵循两阶段蓝图规划、AST 改写与只读沙箱执行。

### 方式 2：通过 MCP (Model Context Protocol) 接入 dsh
本项目提供了符合标准规范的 JSON-RPC 2.0 stdio MCP Server：[`mcp_server.py`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/mcp_server.py)，并配置了 [`.dsh/mcp.json`](file:///Users/tangzhiyou/study-codes/llm-sql-demo1/.dsh/mcp.json)。

你可以在 `dsh`、Claude Desktop 或任何 MCP 客户端中直接连接该服务器，所有 5 个自定义 DB Tools 将直接作为原生工具被模型调用：
```bash
# 测试 MCP 服务是否正常
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | .venv/bin/python mcp_server.py
```


