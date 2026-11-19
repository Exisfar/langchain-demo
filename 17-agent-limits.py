"""17：限制单轮和会话累计模型调用次数。"""

from langchain.agents.middleware import ModelCallLimitMiddleware

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
    limits = ModelCallLimitMiddleware(
        run_limit=3,
        thread_limit=6,
        exit_behavior="end",
    )
    with sqlite_checkpointer() as checkpointer:
        agent = create_shop_agent(
            model,
            load_vector_store(),
            checkpointer=checkpointer,
            middleware=[
                limits,
                history_summary(model),
                handle_tool_errors,
                order_retry(),
            ],
            simulate_errors=True,
            fail_once=True,
        )
        ask(agent, thread_id, question, stream=True, debug=True)


if __name__ == "__main__":
    main()
