from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools.save_to_file import save_tool
from tools.code_search import code_search_tool


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a expert software engineer that will help in developing and analyzing code base. Answer the questions using necessory tools.
         wrap the output in this format and provide no other text\n {format_instructions}
        """
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())


llm = ChatOllama(model="qwen3")

tools = [save_tool, code_search_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
raw_response = agent_executor.invoke({
    "query": "Please generate a sample react component code and put them in a file with .tsx extension.'"
})
