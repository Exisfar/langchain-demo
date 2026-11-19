"""03：逐段输出模型回答。"""

from demo_common import chat_model


def main():
    model = chat_model()
    answer = ""
    for chunk in model.stream(
        [
            {"role": "system", "content": "你是一位中文编程老师，用简短例子解释概念。"},
            {
                "role": "user",
                "content": "解释 Python 中的 async 和 await，并给出示例。",
            },
        ]
    ):
        print(chunk.text, end="", flush=True)
        answer += chunk.text
    print("\n流式输出结束")


if __name__ == "__main__":
    main()
