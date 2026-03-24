from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# ====== custom built tool
# from tavily import TavilyClient
#
# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#        Tool that searches over internet
#        Args:
#            query: The query to search for
#        Returns:
#            The search result
#    """
#     return tavily.search(query = query)
#  tools = [search]


llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    result = agent.invoke({"messages": HumanMessage(content="What is the weather in London")})
    print(result)

if __name__ == "__main__":
    main()
