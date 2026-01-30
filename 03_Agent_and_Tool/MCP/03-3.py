# test_mcp.py
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv

_ = load_dotenv()

async def main():
    client = MultiServerMCPClient(
        {
            "japan_transfer": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "japan-transfer-mcp"],
            }
        }
    )

    async with client.session("japan_transfer") as session:  
        tools = await load_mcp_tools(session)
    
        llm = init_chat_model(model="gpt-4o-mini", temperature=0)

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="""
            You are a Japan transit assistant.
            - Station names MUST be in japanese.
            - Call tools at most 2 times.
            - Prefer the fastest route.
            - Do NOT guess station names.
            - Translate the answer into Korean.
            """
        )

        query = "도쿄에서 오사카까지 2026년 1월 30일 오전 9시에 가장 빠른 경로 알려줘"

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )

        print("\n=== Agent Output ===")
        print(result["messages"][-1].content)

asyncio.run(main())
