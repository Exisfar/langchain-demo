"""08：加载已有索引，避免重新向量化文档。"""

import sys

from demo_common import chat_model, format_context, load_vector_store


def main():
    store = load_vector_store()
    question = sys.argv[1] if len(sys.argv) > 1 else "申请保修需要准备什么？"
    context = format_context(store.similarity_search(question, k=3))
    print("检索结果：\n", context)
    response = chat_model().invoke(
        [
            {
                "role": "system",
                "content": "只根据参考资料回答并标注来源，资料不足时说无法确认。",
            },
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"},
        ]
    )
    print("回答：", response.text)


if __name__ == "__main__":
    main()
