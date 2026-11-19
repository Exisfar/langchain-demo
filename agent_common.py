"""共享的客服工具、中间件与会话输出；各编号脚本负责组合学习阶段。"""

import json
import sys
from contextlib import contextmanager
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ToolRetryMiddleware,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

from demo_common import DATABASE_PATH

ORDERS = {
    "A1001": {"product": "机械键盘", "status": "已发货"},
    "A1002": {"product": "无线鼠标", "status": "待发货"},
}

SYSTEM_PROMPT = (
    "你是商店客服，结合当前会话历史理解用户的追问。"
    "查询具体订单时必须调用 get_order；查询政策时必须调用 search_policies。"
    "订单和政策事实以工具结果为依据，引用政策时标注 source。"
    "不知道的信息要明确说明，不要猜测。"
    "工具最终返回服务不可用时，本轮不要重复调用该工具，明确告知用户查询失败。"
)


class OrderServiceUnavailable(ConnectionError):
    """模拟可以重试的订单服务故障。"""

    code = "ORDER_SERVICE_UNAVAILABLE"


def make_order_tool(*, simulate_errors=False, fail_once=False):
    should_fail_once = fail_once

    @tool
    def get_order(order_id: str) -> str:
        """根据订单编号查询商品和发货状态，例如 A1001。"""
        nonlocal should_fail_once
        print("[查订单]", order_id)
        if should_fail_once and order_id == "A1001":
            should_fail_once = False
            print("[模拟故障] 第一次查询失败")
            raise OrderServiceUnavailable("模拟订单服务暂时不可用")
        if simulate_errors and order_id == "ERR001":
            raise OrderServiceUnavailable("模拟订单服务连接失败")
        order = ORDERS.get(order_id)
        result = (
            {"order_id": order_id, **order}
            if order
            else {"order_id": order_id, "error": "订单不存在"}
        )
        return json.dumps(result, ensure_ascii=False)

    return get_order


def make_policy_tool(vector_store):
    @tool
    def search_policies(query: str) -> str:
        """使用完整、明确的问题检索商店的退款、配送和保修政策。"""
        print("[查知识库]", query)
        documents = vector_store.similarity_search(query, k=3)
        return json.dumps(
            [
                {
                    "source": f"{doc.metadata['source']}#片段{doc.metadata['chunk']}",
                    "content": doc.page_content,
                }
                for doc in documents
            ],
            ensure_ascii=False,
        )

    return search_policies


@wrap_tool_call
def handle_tool_errors(request, handler):
    """只捕获已知服务故障，其他异常继续向外传播。"""
    try:
        return handler(request)
    except OrderServiceUnavailable as error:
        print(f"\n[工具异常] {request.tool_call['name']}：{error}", file=sys.stderr)
        return ToolMessage(
            content=json.dumps(
                {
                    "error": error.code,
                    "message": "订单服务暂时不可用，本轮不要重试，请告知用户稍后再试。",
                },
                ensure_ascii=False,
            ),
            tool_call_id=request.tool_call["id"],
            status="error",
        )


def order_retry() -> ToolRetryMiddleware:
    return ToolRetryMiddleware(
        tools=["get_order"],
        max_retries=2,
        retry_on=(OrderServiceUnavailable,),
        initial_delay=0.5,  # Python 版本使用秒，分别等待 0.5 秒、1 秒。
        backoff_factor=2,
        jitter=False,
        on_failure="error",
    )


def history_summary(model) -> SummarizationMiddleware:
    return SummarizationMiddleware(
        model=model,
        trigger=("messages", 6),
        keep=("messages", 2),
        summary_prompt=(
            "将以下对话压缩为简短中文摘要，保留姓名、订单编号、回答偏好、"
            "已确认事实和未完成事项。不要添加未知信息。以[对话摘要]开头。\n"
            "对话内容：\n{messages}"
        ),
    )


def create_shop_agent(
    model,
    vector_store,
    *,
    checkpointer=None,
    middleware=(),
    simulate_errors=False,
    fail_once=False,
):
    return create_agent(
        model=model,
        tools=[
            make_policy_tool(vector_store),
            make_order_tool(simulate_errors=simulate_errors, fail_once=fail_once),
        ],
        checkpointer=checkpointer,
        middleware=list(middleware),
        system_prompt=SYSTEM_PROMPT,
    )


@contextmanager
def sqlite_checkpointer(path: Path | None = None):
    """Python 检查点使用独立文件，不覆盖旧版数据库。"""
    path = path if path is not None else DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        yield checkpointer


def thread_question() -> tuple[str, str]:
    thread_id = sys.argv[1] if len(sys.argv) > 1 else "session-a"
    question = sys.argv[2] if len(sys.argv) > 2 else "我叫什么？我的订单编号是什么？"
    return thread_id, question


def ask(agent, thread_id: str, question: str, *, stream=False, debug=False):
    print(f"\n[{thread_id}] 用户：{question}")
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [{"role": "user", "content": question}]}
    if stream:
        started = False
        for chunk, metadata in agent.stream(inputs, config, stream_mode="messages"):
            if metadata.get("lc_source") == "summarization":
                continue
            # Python 的 AIMessageChunk 也是 AIMessage；不要把工具结果打印成回答。
            if not isinstance(chunk, AIMessage) or not chunk.text:
                continue
            if not started:
                print("\n助手：", end="", flush=True)
                started = True
            print(chunk.text, end="", flush=True)
        print()
        # 某些中间件直接注入结束消息；它们不一定产生模型 token 事件。
        state = agent.get_state(config)
        messages = state.values.get("messages", [])
        if not started and messages and isinstance(messages[-1], AIMessage):
            print("助手：", messages[-1].text)
    else:
        result = agent.invoke(inputs, config)
        print("助手：", result["messages"][-1].text)
        state = agent.get_state(config)
    if debug:
        print_state(state.values)
    return state


def print_state(values):
    if "thread_model_call_count" in values:
        print("[会话累计模型调用次数]", values["thread_model_call_count"])
    messages = values.get("messages", [])
    print("[当前状态消息数]", len(messages))
    for message in messages:
        if message.additional_kwargs.get("lc_source") == "summarization":
            print("[摘要内容]", message.text)
