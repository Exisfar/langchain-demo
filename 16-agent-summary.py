"""16：达到阈值后摘要历史，保留近期消息。"""

from agent_common import (
    ask,
    create_shop_agent,
    handle_tool_errors,
    history_summary,
    order_retry,
    sqlite_checkpointer,
    thread_question,
)
from demo_common import chat_model, load_vector_store


def main():
    thread_id, question = thread_question()
    model = chat_model()
    with sqlite_checkpointer() as checkpointer:
        agent = create_shop_agent(
            model,
            load_vector_store(),
            checkpointer=checkpointer,
            middleware=[history_summary(model), handle_tool_errors, order_retry()],
            simulate_errors=True,
            fail_once=True,
        )
        ask(agent, thread_id, question, stream=True, debug=True)


if __name__ == "__main__":
    main()
