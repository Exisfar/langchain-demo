"""08：建立索引并保存文档、向量和模型名称。"""

from demo_common import embedding_model, save_index, shop_documents


def main():
    documents = shop_documents()
    save_index(documents, embedding_model())
    print(f"已保存 {len(documents)} 个文档片段及其向量")


if __name__ == "__main__":
    main()
