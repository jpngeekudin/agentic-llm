from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor


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
        You are a helpful research assistant that will help in generating a research paper. Answer the questions using necessory tools.
         wrap the output in this format and provide no other text\n {format_instructions}
        """
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}")
]).partial(format_instructions=parser.get_format_instructions())


def save_to_file(content: str, filename: str = 'content.txt'):
    """
    Save the given content to a text file.

    Args:
        content (str): The content to save.
        filename (str): The name of the file to save the content to.
    """

    with open(filename, 'w') as file:
        file.write(content)
    return f"Content saved to {filename}"


llm = ChatOllama(model="qwen3")

class SaveToolInput(BaseModel):
    content: str = Field(..., description="The content to save")
    filename: str = Field(..., description="The name of the file to save the content to")

save_tool = StructuredTool.from_function(
    func=save_to_file,
    name="save_to_file",
    description="Saves the provided content to a file, input should be the content",
    args_schema=SaveToolInput
)

tools = [save_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
raw_response = agent_executor.invoke({
    "query": "Find capital of france and save it to a filename 'capital.txt'"
})
