"""07：读取并切分本地 Markdown，检索相关片段。"""

import sys

from langchain_core.vectorstores import InMemoryVectorStore

from demo_common import chat_model, embedding_model, format_context, shop_documents


def main():
    documents = shop_documents()
    print(f"文档共切成 {len(documents)} 个片段")
    store = InMemoryVectorStore.from_documents(documents, embedding_model())
    question = sys.argv[1] if len(sys.argv) > 1 else "退款审核通过后，多久能收到钱？"
    context = format_context(store.similarity_search(question, k=3))
    print("检索结果：\n", context)
    response = chat_model().invoke(
        [
            {
                "role": "system",
                "content": "只根据参考资料回答，标注文件与片段，资料不足时说无法确认。",
            },
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"},
        ]
    )
    print("回答：", response.text)


if __name__ == "__main__":
    main()
