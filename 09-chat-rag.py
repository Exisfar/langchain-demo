"""09：交互式 RAG，结合历史改写追问并流式回答。"""

from demo_common import chat_model, format_context, load_vector_store


def main():
    model = chat_model()
    store = load_vector_store()
    history = []
    print("知识库助手已启动，输入 /exit 退出。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question == "/exit":
            break
        if not question:
            continue
        try:
            search_question = question
            if history:
                rewritten = model.invoke(
                    [
                        {
                            "role": "system",
                            "content": "结合历史，将最后的问题改写成可独立理解的检索问题。只输出问题，不回答，不添加未知信息。切换话题时以最新话题为准。",
                        },
                        *history,
                        {"role": "user", "content": question},
                    ]
                )
                search_question = rewritten.text.strip() or question
            print("[检索问题]", search_question)
            documents = store.similarity_search(search_question, k=3)
            if not documents:
                print("助手：没有检索到参考资料，无法确认。")
                continue
            context = format_context(documents)
            print(f"[检索结果] {len(documents)} 个片段\n{context}")
            stream = model.stream(
                [
                    {
                        "role": "system",
                        "content": "历史用于理解意图，政策事实只以本轮资料为依据。回答标注来源，资料不足时说无法确认。",
                    },
                    *history,
                    {
                        "role": "user",
                        "content": f"参考资料：\n{context}\n\n问题：{question}",
                    },
                ]
            )
            print("助手：", end="", flush=True)
            answer = ""
            for chunk in stream:
                print(chunk.text, end="", flush=True)
                answer += chunk.text
            print()
            history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
        except Exception as error:  # noqa: BLE001 - 单轮失败后仍允许继续提问
            print(f"\n本轮失败：{error}")


if __name__ == "__main__":
    main()
