"""15：工具故障自动重试，首次执行之外最多重试两次。"""

from agent_common import (
    ask,
    create_shop_agent,
    handle_tool_errors,
    order_retry,
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
            # 外层负责最终失败，内层先执行有限次数的重试。
            middleware=[handle_tool_errors, order_retry()],
            simulate_errors=True,
            fail_once=True,
        )
        ask(agent, thread_id, question, stream=True)


if __name__ == "__main__":
    main()
