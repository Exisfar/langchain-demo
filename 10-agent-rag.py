"""10：让 Agent 按需调用政策检索与订单查询工具。"""

import sys

from agent_common import create_shop_agent
from demo_common import chat_model, load_vector_store


def main():
    agent = create_shop_agent(chat_model(), load_vector_store())
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "帮我查一下 A1001 发货了吗？申请退款需要什么材料？"
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("助手：", result["messages"][-1].text)


if __name__ == "__main__":
    main()
