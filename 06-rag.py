"""06：检索示例资料，再根据资料回答。"""

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from demo_common import chat_model, embedding_model, format_context


def main():
    documents = [
        Document(
            page_content="退款政策：商品签收后 7 天内，未使用且包装完整可以申请退款。",
            metadata={"source": "退款政策"},
        ),
        Document(
            page_content="配送政策：订单付款后 48 小时内发货，包邮地区不含偏远地区。",
            metadata={"source": "配送政策"},
        ),
        Document(
            page_content="保修政策：电子产品提供 12 个月保修，人为损坏不在保修范围内。",
            metadata={"source": "保修政策"},
        ),
    ]
    store = InMemoryVectorStore.from_documents(documents, embedding_model())
    question = "东西收到 3 天了，还没拆封，可以退吗？"
    context = format_context(store.similarity_search(question, k=2))
    print("检索到的资料：\n", context)
    response = chat_model().invoke(
        [
            {
                "role": "system",
                "content": "只根据参考资料回答并标注来源，资料不足时说无法确认。",
            },
            {
                "role": "user",
                "content": f"参考资料：\n{context}\n\n用户问题：{question}",
            },
        ]
    )
    print("客服：", response.text)


if __name__ == "__main__":
    main()
