from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """
    Schema for a source used by agent
    """

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """
    Schema for agent response
    """

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatOpenAI()
tools = [TavilySearch(max_results=3)]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for 3 job postings for an ai engineer in Bangalore on linkedIn and list their details"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
