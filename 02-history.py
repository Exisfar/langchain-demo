"""02：手动保存消息历史，实现多轮对话。"""

from demo_common import chat_model


def main():
    model = chat_model()
    messages = [{"role": "system", "content": "你是一位中文编程老师，回答保持简洁。"}]
    messages.append({"role": "user", "content": "我叫小明，正在学习 Python。"})
    first_reply = model.invoke(messages)
    print("第一轮：", first_reply.text)
    messages.append(first_reply)
    messages.append({"role": "user", "content": "我叫什么？正在学什么？"})
    second_reply = model.invoke(messages)
    print("第二轮：", second_reply.text)


if __name__ == "__main__":
    main()
