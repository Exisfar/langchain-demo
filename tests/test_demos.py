"""离线验证示例迁移，所有模型和向量调用都替换为确定性的本地实现。"""

import json
import re
import runpy
import socket
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field, ValidationError

import agent_common
import demo_common

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(PROJECT.glob("[0-9][0-9]-*.py"))


class OfflineEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [
            1.0 if any(word in text for word in words) else 0.01
            for words in [("退", "到账"), ("配送", "发货", "物流"), ("保修", "凭证")]
        ]


class OfflineChatModel(BaseChatModel):
    calls: int = 0
    tool_names: list[str] = Field(default_factory=list)
    loop_forever: bool = False

    @property
    def _llm_type(self):
        return "offline-demo-test"

    def bind_tools(self, tools, **kwargs):
        self.tool_names = [item.name for item in tools]
        return self

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(
            lambda _: schema(name="小明", age=22, skills=["Python", "SQL"])
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        last = messages[-1]
        question = last.text
        if "将以下对话压缩为简短中文摘要" in question:
            reply = AIMessage(content="[对话摘要] 用户叫小明，偏好中文简短回答和列表。")
        elif any(
            "检索问题" in message.text
            for message in messages
            if message.type == "system"
        ):
            reply = AIMessage(content="申请退款需要什么材料？")
        elif isinstance(last, ToolMessage) and not self.loop_forever:
            tools = []
            for message in reversed(messages):
                if message.type == "human":
                    break
                if isinstance(message, ToolMessage):
                    tools.append(message)
            failed = any(message.status == "error" for message in tools)
            missing = any("订单不存在" in message.text for message in tools)
            content = (
                "订单服务暂时不可用。"
                if failed
                else ("订单不存在。" if missing else "订单已查询。")
            )
            reply = AIMessage(
                content=content
                + "退款请提供订单编号和商品照片。[knowledge/shop.md#片段1]"
            )
        elif self.tool_names and (
            self.loop_forever
            or any(word in question for word in ["查询", "退款", "发货", "保修"])
        ):
            calls = []
            if "get_order" in self.tool_names and (
                self.loop_forever or "查询" in question or "发货" in question
            ):
                identifiers = re.findall(
                    r"ERR001|A100[12]|A9999", " ".join(m.text for m in messages)
                )
                order_id = identifiers[-1] if identifiers else "A1002"
                calls.append(
                    {
                        "name": "get_order",
                        "args": {"order_id": order_id},
                        "id": str(uuid4()),
                        "type": "tool_call",
                    }
                )
            if "search_policies" in self.tool_names and any(
                word in question for word in ["退款", "保修"]
            ):
                calls.append(
                    {
                        "name": "search_policies",
                        "args": {"query": "退款需要什么材料"},
                        "id": str(uuid4()),
                        "type": "tool_call",
                    }
                )
            reply = AIMessage(content="" if calls else "你好。", tool_calls=calls)
        elif "我叫什么" in question:
            known = any("小明" in message.text for message in messages)
            reply = AIMessage(
                content="你叫小明，偏好中文简短回答和列表。"
                if known
                else "无法确定姓名和订单编号。"
            )
        else:
            reply = AIMessage(content="你好，小明。我会使用中文简短回答。")
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        reply = (
            self._generate(messages, stop, run_manager, **kwargs).generations[0].message
        )
        if reply.tool_calls:
            chunks = [
                {
                    "name": call["name"],
                    "args": json.dumps(call["args"]),
                    "id": call["id"],
                    "index": index,
                }
                for index, call in enumerate(reply.tool_calls)
            ]
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", tool_call_chunks=chunks)
            )
        else:
            for text in [reply.text[:4], reply.text[4:]]:
                if text:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=text))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("离线测试禁止访问网络")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge/shop.md").write_text(
        (PROJECT / "knowledge/shop.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setattr(demo_common, "ROOT", tmp_path)
    monkeypatch.setattr(demo_common, "INDEX_PATH", tmp_path / "storage/index.json")
    monkeypatch.setattr(
        agent_common, "DATABASE_PATH", tmp_path / "storage/conversations-python.sqlite"
    )
    monkeypatch.setattr(demo_common, "chat_model", OfflineChatModel)
    monkeypatch.setattr(
        demo_common,
        "embedding_model",
        lambda model=demo_common.EMBEDDING_MODEL: OfflineEmbeddings(),
    )
    demo_common.save_index(demo_common.shop_documents(), OfflineEmbeddings())
    return tmp_path


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_every_demo_runs_offline(script, sandbox, monkeypatch, capsys):
    if script.name.startswith(("07-", "08-query", "10-")):
        args = [script.name, "查询 A1002 并说明退款材料"]
    elif script.name.startswith(tuple(f"{index}-" for index in range(12, 18))):
        args = [script.name, "smoke", "查询 A1002 并说明退款材料"]
    else:
        args = [script.name]
    monkeypatch.setattr(sys, "argv", args)
    inputs = iter(["我想退款", "需要准备什么？", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out.strip()


def test_existing_index_restores_without_document_api(sandbox):
    path = demo_common.INDEX_PATH
    saved = json.loads(path.read_text(encoding="utf-8"))
    # 也验证之前保存的 JSON 字段可以直接读取。
    saved["embeddingModel"] = saved.pop("embedding_model")
    for doc in saved["documents"]:
        doc["pageContent"] = doc.pop("page_content")
    path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    class QueryOnlyEmbeddings(OfflineEmbeddings):
        def embed_documents(self, texts):
            pytest.fail("恢复索引不应重新调用文档向量 API")

    store = demo_common.load_vector_store(embeddings=QueryOnlyEmbeddings())
    documents = store.similarity_search("退款材料", k=1)
    assert "订单编号和商品照片" in documents[0].page_content


def test_invalid_index_reports_mismatched_vectors(sandbox):
    path = demo_common.INDEX_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    data["vectors"].pop()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="数量不匹配"):
        demo_common.load_vector_store(embeddings=OfflineEmbeddings())


@pytest.mark.parametrize(
    "order_id, expected_calls, error",
    [("A1001", 2, False), ("ERR001", 3, True), ("A9999", 1, False)],
)
def test_retries_and_error_messages(sandbox, capsys, order_id, expected_calls, error):
    agent = agent_common.create_shop_agent(
        OfflineChatModel(),
        demo_common.load_vector_store(),
        checkpointer=InMemorySaver(),
        middleware=[agent_common.handle_tool_errors, agent_common.order_retry()],
        simulate_errors=True,
        fail_once=True,
    )
    state = agent_common.ask(
        agent, "retry", f"查询 {order_id}，同时说明退款材料", stream=True
    )
    output = capsys.readouterr().out
    assert output.count(f"[查订单] {order_id}") == expected_calls
    assert '"source":' not in output
    assert '"error":' not in output
    tools = [
        message
        for message in state.values["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert len(tools) == 2
    assert any(message.status == "error" for message in tools) is error
    assert any("订单编号和商品照片" in message.text for message in tools)


def test_sqlite_reopen_and_thread_isolation(sandbox, capsys):
    with agent_common.sqlite_checkpointer() as saver:
        agent = agent_common.create_shop_agent(
            OfflineChatModel(), demo_common.load_vector_store(), checkpointer=saver
        )
        agent_common.ask(agent, "a", "我叫小明，我的订单编号是 A1001。")
    with agent_common.sqlite_checkpointer() as saver:
        agent = agent_common.create_shop_agent(
            OfflineChatModel(), demo_common.load_vector_store(), checkpointer=saver
        )
        recovered = agent_common.ask(agent, "a", "我叫什么？")
        isolated = agent_common.ask(agent, "b", "我叫什么？")
    assert "小明" in recovered.values["messages"][-1].text
    assert "无法确定" in isolated.values["messages"][-1].text


def test_summary_preserves_recent_question_and_hides_internal_stream(sandbox, capsys):
    model = OfflineChatModel()
    agent = agent_common.create_shop_agent(
        model,
        demo_common.load_vector_store(),
        checkpointer=InMemorySaver(),
        middleware=[agent_common.history_summary(model)],
    )
    for question in ["我叫小明", "请用中文简短回答", "我偏好列表", "我叫什么？"]:
        state = agent_common.ask(agent, "summary", question, stream=True)
    assert "[对话摘要]" not in capsys.readouterr().out
    messages = state.values["messages"]
    assert len(messages) < 8
    assert any(
        message.additional_kwargs.get("lc_source") == "summarization"
        for message in messages
    )
    assert any(message.text == "我叫什么？" for message in messages)
    assert "小明" in messages[-1].text


def test_call_limit_stops_loop_and_persists_across_reopen(sandbox):
    model = OfflineChatModel(loop_forever=True)
    with agent_common.sqlite_checkpointer() as saver:
        agent = agent_common.create_shop_agent(
            model,
            demo_common.load_vector_store(),
            checkpointer=saver,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=1, thread_limit=1, exit_behavior="end"
                )
            ],
        )
        state = agent_common.ask(agent, "limit", "查询 A1002", stream=True)
    assert model.calls == 1
    assert state.values["thread_model_call_count"] == 1
    assert "limit" in state.values["messages"][-1].text.lower()
    second_model = OfflineChatModel()
    with agent_common.sqlite_checkpointer() as saver:
        agent = agent_common.create_shop_agent(
            second_model,
            demo_common.load_vector_store(),
            checkpointer=saver,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=3, thread_limit=1, exit_behavior="end"
                )
            ],
        )
        agent_common.ask(agent, "limit", "你好", stream=True)
    assert second_model.calls == 0


def test_structured_schema_rejects_wrong_types():
    schema = runpy.run_path(str(PROJECT / "05-structured.py"))["UserInfo"]
    assert schema(name=None, age=None, skills=[]).age is None
    with pytest.raises(ValidationError):
        schema(name="小明", age="不是数字", skills=[])


def test_vector_search_keeps_source_metadata():
    documents = [
        Document(
            page_content="退款需要订单编号和商品照片", metadata={"source": "退款政策"}
        ),
        Document(page_content="保修需要凭证", metadata={"source": "保修政策"}),
    ]
    store = InMemoryVectorStore.from_documents(documents, OfflineEmbeddings())
    assert store.similarity_search("退款", k=1)[0].metadata["source"] == "退款政策"
