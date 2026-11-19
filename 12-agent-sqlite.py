"""12：将会话状态写入 SQLite，支持跨进程恢复。"""

from agent_common import ask, create_shop_agent, sqlite_checkpointer, thread_question
from demo_common import chat_model, load_vector_store


def main():
    thread_id, question = thread_question()
    with sqlite_checkpointer() as checkpointer:
        agent = create_shop_agent(
            chat_model(), load_vector_store(), checkpointer=checkpointer
        )
        ask(agent, thread_id, question)


if __name__ == "__main__":
    main()
