"""05：使用 Pydantic 校验结构化输出。"""

from pydantic import BaseModel, Field

from demo_common import chat_model


class UserInfo(BaseModel):
    name: str | None = Field(description="姓名，未提及则返回 null")
    age: int | None = Field(description="年龄，未提及则返回 null")
    skills: list[str] = Field(description="技能列表，未提及则返回空列表")


def main():
    extractor = chat_model().with_structured_output(UserInfo, method="json_schema")
    result = extractor.invoke(
        [
            {"role": "system", "content": "只提取用户明确提供的信息，不要猜测。"},
            {"role": "user", "content": "我叫小明，今年 22 岁，会 Python 和 SQL。"},
        ]
    )
    print(result.model_dump_json(indent=2))
    print("姓名：", result.name)
    print("技能数量：", len(result.skills))


if __name__ == "__main__":
    main()
