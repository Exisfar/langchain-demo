"""14：捕获 ERR001 模拟服务故障，向模型返回工具失败消息。"""

from agent_common import (
    ask,
    create_shop_agent,
    handle_tool_errors,
    sqlite_checkpointer,
    thread_question,
)
from demo_common import chat_model, load_vector_store


def main():
    thread_id, question = thread_question()
    with sqlite_checkpointer() as checkpointer:
        agent = create_shop_agent(
            chat_model(),
            load_vector_store(),
            checkpointer=checkpointer,
            middleware=[handle_tool_errors],
            simulate_errors=True,
        )
        ask(agent, thread_id, question, stream=True)


if __name__ == "__main__":
    main()
