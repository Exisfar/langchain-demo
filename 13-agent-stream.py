"""13：流式显示 Agent 回答，过滤工具消息。"""

from agent_common import ask, create_shop_agent, sqlite_checkpointer, thread_question
from demo_common import chat_model, load_vector_store


def main():
    thread_id, question = thread_question()
    with sqlite_checkpointer() as checkpointer:
        agent = create_shop_agent(
            chat_model(), load_vector_store(), checkpointer=checkpointer
        )
        # ask 内部使用 agent.stream(..., stream_mode="messages")。
        ask(agent, thread_id, question, stream=True)


if __name__ == "__main__":
    main()
