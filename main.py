from typing import List
from pydantic import BaseModel, Field

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


# this is to format the response
class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="Source URL")

class AgentResponse(BaseModel):
    """Schema for a response from the agent"""
    answer:str = Field(description="Response from the agent")
    source:List[Source] = Field(default_factory=List, description="Source used by the agent")



llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")})
    print(result)

if __name__ == "__main__":
    main()
