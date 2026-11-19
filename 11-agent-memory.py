"""11：按 thread_id 保存内存会话状态。"""

from langgraph.checkpoint.memory import InMemorySaver

from agent_common import ask, create_shop_agent
from demo_common import chat_model, load_vector_store


def main():
    agent = create_shop_agent(
        chat_model(), load_vector_store(), checkpointer=InMemorySaver()
    )
    ask(agent, "session-a", "我叫小明，我的订单编号是 A1001。")
    ask(agent, "session-a", "我叫什么？刚才的订单号是什么？")
    ask(agent, "session-a", "帮我查一下刚才的订单发货了吗？")
    ask(agent, "session-b", "我叫什么？我的订单编号是什么？")


if __name__ == "__main__":
    main()
