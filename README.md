# LangChain Python 学习 Demo

一个循序渐进的 LangChain Python 入门项目，包含 **17 个学习阶段、18 个可运行脚本**。从模型调用开始，逐步学习多轮对话、流式输出、工具调用、结构化输出、RAG、Agent 会话持久化，以及异常处理、自动重试、历史摘要和调用次数限制。

示例以商店客服为场景：查询本地退款、配送和保修政策，调用工具查询模拟订单，并结合会话历史回答追问。项目使用 **uv 管理依赖、锁文件和虚拟环境**，所有命令在项目根目录执行。

## 环境与安装

项目支持 Python 3.12–3.14，`.python-version` 指定 Python 3.14；本次迁移使用 Python 3.14.10 验证。

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再运行：

```bash
uv sync --locked
```

uv 会按 `uv.lock` 安装依赖并创建项目内的 `.venv/`。使用 `uv run` 时会自动使用此虚拟环境，无需手动激活。

参考 [.env.example](./.env.example)，在项目根目录创建或修改 `.env`：

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=your-model-name
```

替换为实际 API Key 和账号可用的模型名称。工具调用和结构化输出需要模型支持相应能力。知识库使用 `text-embedding-3-small`，账号还需要能调用这个 Embedding 模型。

脚本会自动从项目根目录加载 `.env`，无需额外传入环境文件参数。运行实际模型与 Embedding 示例会产生 API 用量。

## 快速开始

先调用一次模型：

```bash
uv run 01-chat.py
```

再体验带知识库、工具和 SQLite 会话的流式 Agent：

```bash
uv run 08-build-index.py
uv run 13-agent-stream.py quickstart-demo "查询 A1001 发货了吗？申请退款需要什么材料？"
uv run 13-agent-stream.py quickstart-demo "我刚才查询的订单编号是什么？"
```

后两条命令使用同一个会话编号。即使每次命令都启动新的进程，第二次仍会恢复上一次的会话历史。

## 示例目录与学习顺序

| 阶段 | 文件 | 学习内容 |
| --- | --- | --- |
| 01 | [01-chat.py](./01-chat.py) | 创建模型客户端，使用 `invoke()` 获取回答 |
| 02 | [02-history.py](./02-history.py) | 手动保存并传递消息历史 |
| 03 | [03-stream.py](./03-stream.py) | 使用 `stream()` 逐段输出回答 |
| 04 | [04-tools.py](./04-tools.py) | 使用 `@tool` 定义订单查询工具，让 Agent 执行函数 |
| 05 | [05-structured.py](./05-structured.py) | 使用 Pydantic 和 `with_structured_output()` 提取信息 |
| 06 | [06-rag.py](./06-rag.py) | 向量化示例资料，完成基础 RAG |
| 07 | [07-file-rag.py](./07-file-rag.py) | 读取 Markdown，切分文本并检索回答 |
| 08 | [08-build-index.py](./08-build-index.py)、[08-query.py](./08-query.py) | 分离索引构建与查询，保存并复用文档向量 |
| 09 | [09-chat-rag.py](./09-chat-rag.py) | 结合历史改写追问，交互式检索并流式回答 |
| 10 | [10-agent-rag.py](./10-agent-rag.py) | Agent 按需选择政策检索、订单查询工具 |
| 11 | [11-agent-memory.py](./11-agent-memory.py) | 使用 `InMemorySaver` 和 `thread_id` 管理会话 |
| 12 | [12-agent-sqlite.py](./12-agent-sqlite.py) | SQLite 会话持久化与恢复 |
| 13 | [13-agent-stream.py](./13-agent-stream.py) | Agent 流式回答及工具消息过滤 |
| 14 | [14-tool-errors.py](./14-tool-errors.py) | 使用 `@wrap_tool_call` 处理已知工具异常 |
| 15 | [15-tool-retry.py](./15-tool-retry.py) | 使用 `ToolRetryMiddleware` 自动重试 |
| 16 | [16-agent-summary.py](./16-agent-summary.py) | 使用 `SummarizationMiddleware` 压缩历史 |
| 17 | [17-agent-limits.py](./17-agent-limits.py) | 使用 `ModelCallLimitMiddleware` 限制模型调用次数 |

01–06 的提问写在脚本中，可直接修改。07、08-query 和 10 接受一个问题参数；12–17 接受“会话编号、问题”两个参数；09 为交互式终端。

公共代码集中在两个文件中，方便比较各阶段增加了哪些能力：

- [demo_common.py](./demo_common.py)：环境配置、模型客户端、文档切分和索引读写。
- [agent_common.py](./agent_common.py)：订单及政策工具、中间件、SQLite 生命周期和回答输出。

## 技术栈与常用接口

- **LangChain**：`create_agent`、工具和 Agent 中间件。
- **langchain-openai**：`ChatOpenAI` 与 `OpenAIEmbeddings`。
- **Pydantic**：工具参数和结构化输出的数据校验。
- **langchain-text-splitters**：`RecursiveCharacterTextSplitter` 文本切分。
- **langchain-core**：`Document`、消息类型和 `InMemoryVectorStore`。
- **LangGraph**：`InMemorySaver` 与 `SqliteSaver` 会话状态保存。
- **uv**：依赖管理、版本锁定、虚拟环境和命令运行。

Python 消息的正文通过 `.text` 读取。流式输出时使用 `isinstance(chunk, AIMessage)`，同时覆盖完整 AI 消息和 AI 消息片段，过滤工具返回的原始数据。

## 知识库示例

### 读取与切分文档

知识库内容位于 [knowledge/shop.md](./knowledge/shop.md)，包含退款、配送和保修政策。

```bash
uv run 07-file-rag.py "申请保修需要准备什么？"
```

每次运行都会重新读取、切分并向量化文档。当前切分参数是 `chunk_size=100`、`chunk_overlap=20`，优先在段落、换行和中文标点处切分。长度默认按字符计算。

### 保存并复用索引

运行 `08-query.py` 及 09–17 阶段前，需要先有 `storage/index.json`：

```bash
uv run 08-build-index.py
uv run 08-query.py "退款审核通过后多久能到账？"
```

索引保存文档片段、来源信息、向量和 Embedding 模型名。加载时通过缓存适配器恢复已保存的文档向量，不再调用文档 Embedding API；每次提问仍需向量化问题并调用聊天模型。

修改文档、切分参数或 Embedding 模型后，重新运行 `08-build-index.py` 更新索引。加载器也支持已有索引中的 `embeddingModel` 和 `pageContent` 字段，因此之前生成的 JSON 可以直接复用。

### 多轮知识库聊天

```bash
uv run 09-chat-rag.py
```

连续输入：

```text
我想申请退款。
需要准备什么？
审核通过后多久能到账？
```

程序会显示改写后的检索问题、检索片段和流式回答。输入 `/exit`、按 Ctrl+C 或结束终端输入均可退出。此阶段的聊天历史只保存在当前进程内。

## Agent 与会话

### 工具组合

```bash
uv run 10-agent-rag.py "查询 A1001 的发货状态，再告诉我退款需要什么材料。"
```

Agent 可以选择两个工具：

- `get_order(order_id)`：查询模拟订单；`A1001` 为已发货的机械键盘，`A1002` 为待发货的无线鼠标。
- `search_policies(query)`：检索商店政策，并返回文件来源和片段编号。

### 内存会话

```bash
uv run 11-agent-memory.py
```

脚本演示同一个 `session-a` 的连续对话，以及 `session-b` 的独立历史。`InMemorySaver` 的状态会在进程退出后丢失。

### SQLite 会话

```bash
uv run 12-agent-sqlite.py session-a "我叫小明，我的订单编号是 A1001。"
uv run 12-agent-sqlite.py session-a "帮我查询之前那个订单的发货状态。"
uv run 12-agent-sqlite.py session-b "我叫什么？我的订单编号是什么？"
```

12–17 阶段共享 `storage/conversations-python.sqlite`。同一个数据库文件和 `thread_id` 可以恢复会话；新的会话编号拥有独立历史。实验时建议给不同阶段使用不同会话编号。

Python 版本使用独立的检查点数据库，已有 `storage/conversations.sqlite` 保留，但不会自动迁移其中的会话。文档索引与会话数据库相互独立，重建索引不会清空会话记录。

## Agent 中间件

### 13：流式输出

```bash
uv run 13-agent-stream.py stream-demo "查询 A1002 的发货状态。"
```

使用 `agent.stream(..., stream_mode="messages")` 获取消息片段和元数据。工具日志显示执行过程，AI 回答逐段显示。数据流消费完成后，SQLite 连接由上下文管理器关闭。

### 14：工具异常处理

```bash
uv run 14-tool-errors.py error-demo "查询 ERR001 的发货状态，同时告诉我退款需要什么材料。"
```

`ERR001` 会抛出 `OrderServiceUnavailable`。外层中间件捕获已知异常，通过包含对应 `tool_call_id` 的 `ToolMessage` 告知模型失败；其他异常继续向外传播。

| 订单号 | 14 阶段的结果 |
| --- | --- |
| `A1001` | 查询成功：机械键盘，已发货 |
| `A1002` | 查询成功：无线鼠标，待发货 |
| `A9999` | 查询正常完成，但订单不存在 |
| `ERR001` | 模拟服务故障，无法确定订单状态 |

### 15：自动重试

```bash
uv run 15-tool-retry.py retry-success "查询 A1001 的发货状态。"
uv run 15-tool-retry.py retry-failure "查询 ERR001 的发货状态。"
uv run 15-tool-retry.py retry-missing "查询 A9999 的发货状态。"
```

15–17 阶段将 `A1001` 的第一次查询设为失败，之后可以成功；这个模拟状态在每次启动进程时重置。`ERR001` 持续失败，`A9999` 正常返回订单不存在。

当前重试配置：

```python
ToolRetryMiddleware(
    tools=["get_order"],
    max_retries=2,
    retry_on=(OrderServiceUnavailable,),
    initial_delay=0.5,
    backoff_factor=2,
    jitter=False,
    on_failure="error",
)
```

首次执行之外最多重试两次，合计最多执行 3 次，等待时间分别为 0.5 秒和 1 秒。**Python 的延时单位为秒。**

异常处理层包裹重试层，顺序为 `[handle_tool_errors, order_retry()]`：先重试，耗尽后才将异常转换为工具失败消息。

### 16：历史摘要

```bash
uv run 16-agent-summary.py summary-demo "我叫小明，请记住。"
uv run 16-agent-summary.py summary-demo "请用中文简短回答。"
uv run 16-agent-summary.py summary-demo "我偏好使用列表阅读信息。"
uv run 16-agent-summary.py summary-demo "我叫什么？我有哪些回答偏好？"
```

当前使用 `trigger=("messages", 6)` 和 `keep=("messages", 2)`。触发检查发生在调用模型之前，消息数量包含用户、AI 和工具消息。中间件保护工具调用与结果的配对关系，实际保留数量可能调整。

程序打印当前状态消息数及摘要。内部摘要通过 `lc_source="summarization"` 标记，在流式客服回答中被过滤，在调试日志中单独展示。自定义 `summary_prompt` 要求保留姓名、订单编号、偏好等信息。

摘要属于有损压缩，会额外调用模型；SQLite 也可能保留旧检查点，因此当前上下文变短不代表数据库文件一定变小。

### 17：模型调用次数限制

```bash
uv run 17-agent-limits.py limits-demo "查询 A1002 的发货状态。"
```

当前配置为 `run_limit=3`、`thread_limit=6`、`exit_behavior="end"`：

- `run_limit`：单次 Agent 执行的模型调用上限。
- `thread_limit`：同一个会话累计的模型调用上限，随 SQLite 状态保存。
- `exit_behavior="end"`：准备超额调用时结束执行，并生成限制提示。

终端打印 `thread_model_call_count`。可临时把 `run_limit` 改成 `1`，使用新会话查询订单，观察最终回答前的第二次模型调用被拦截，然后恢复为 `3`。

限额统计 Agent 主循环的模型调用，摘要、Embedding 和工具内部重试不包含在内，因此不等于 API 总请求数或费用上限。

## 数据与目录

```text
01-chat.py … 17-agent-limits.py  # 18 个编号示例（08 包含构建和查询）
demo_common.py                 # 公共配置、文档与索引
agent_common.py                # 客服工具、会话与中间件
knowledge/shop.md              # 知识库原文
storage/
  index.json                   # 文档、向量及 Embedding 模型名
  conversations-python.sqlite  # Python 会话数据库
pyproject.toml                 # uv 项目配置与依赖
uv.lock                        # 锁定依赖版本
.python-version                # 项目默认 Python 版本
.env.example                   # 环境变量模板
tests/test_demos.py             # 不访问网络的验证用例
```

`.env`、`.venv/`、`storage/` 和工具缓存均由 `.gitignore` 忽略。新环境需要配置模型并构建索引。

## 开发与验证

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

测试替换了真实聊天模型与 Embedding，禁止网络连接，不消耗 API 额度。覆盖全部 18 个入口、旧索引兼容、检索来源、结构校验、异常重试、消息过滤、会话隔离与恢复、摘要和持久化调用限额。

维护依赖时使用：

```bash
uv add package-name
uv lock --upgrade
uv sync --locked
```

## 常见问题

- **找不到索引**：先执行 `uv run 08-build-index.py`。
- **回答与资料不符**：依次检查检索问题、检索片段与最终回答；文档更新后重建索引。
- **工具 JSON 混进回答**：检查 `isinstance(chunk, AIMessage)` 过滤；内部摘要还需检查 `lc_source`。
- **工具没有重新执行**：Agent 可能复用了历史结果。可换一个新会话编号观察；提示词要求不等于程序强制执行。
- **ERR001 总是失败**：这是故障测试编号，验证订单不存在请使用 `A9999`。
- **重试没有生效**：检查中间件顺序、异常类型，以及异常是否提前被转换成普通结果。
- **会话没有恢复**：使用相同数据库文件和会话编号；Python 会话与之前的检查点文件分开保存。
- **摘要没有立即出现**：阈值在模型调用前检查，达到 6 条消息后可能在下一轮才触发。
- **出现模型调用超限提示**：检查单轮上限和累计计数，重启进程不会清空同一会话的累计计数。

本项目用于学习和本地演示，订单为模拟数据，向量检索在内存中执行。模型回答及引用来源仍需结合实际资料核对。
