"""01：调用聊天模型。运行：uv run 01-chat.py"""

from demo_common import chat_model


def main():
    model = chat_model()
    response = model.invoke(
        [
            {
                "role": "system",
                "content": "你是一位 Python 老师，用中文简洁回答并给出例子。",
            },
            {"role": "user", "content": "解释一下 Python 的列表推导式。"},
        ]
    )
    print(response.text)


if __name__ == "__main__":
    main()
