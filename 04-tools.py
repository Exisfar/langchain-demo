"""04：让 Agent 调用订单查询函数。"""

from langchain.agents import create_agent

from agent_common import make_order_tool
from demo_common import chat_model


def main():
    agent = create_agent(
        model=chat_model(),
        tools=[make_order_tool()],
        system_prompt="你是订单客服。查询订单时必须使用工具，根据结果回答。",
    )
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "帮我查一下 A1001 买了什么，发货了吗？"}
            ]
        }
    )
    print("客服：", result["messages"][-1].text)


if __name__ == "__main__":
    main()
