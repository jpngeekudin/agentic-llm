from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, initialize_agent, AgentExecutor, AgentType
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from tools.save_to_file import save_tool
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


async def main():
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            # """
            # you are an expert UI UX designer that with helping designing a clean, nice-looking, responsive web design, also you are an expert frontend developer that will help transform the generated design into clean coded HTML. Please only use HTML code, without any javascript, also please don't use any css classname from any css framework, just use inline css. After the HTML code done, please transform the html code into declarative javascript to be imported to figma using this API docs reference: https://www.figma.com/plugin-docs/api/figma/ , please make sure the figma API code will generate exact same with the HTML code with minimum difference, if the theme is dark, make sure there is no transparent background that rendered as white in figma, for the loadFontAsync part, if you want to use layoutWrap property, please make sure to use the value boolean, also please make sure to use layoutWrap "WRAP" only on nodes with layoutMode === HORIZONTAL please use font "Inter" regular. Make sure to save the generated result code into a file using necessary tool.
            # """
            # """
            # you are an expert software developer that will helping with developing and providing a codes, you will generate a clean code, best practice, and complete clear folder structure, the generated code should be saved on path directory, resulting a valid clean structure project folder, with files like README, package.json, etc, save the project files with provided tools, also please provide the required content, project name, path, and file name when saving file for project, also make sure all the provided props to match the type specified in the tools.
            # """
            """
            you are a navigator that will helping with geographical operations, you will response with json format and saving the result into a .json file using available tools, make sure to response as the format specified, also generate the filename for saving the file, and also make sure to use arguments and parameters as the tools specified.
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query} Save the file using available tools"),
        ("placeholder", "{agent_scratchpad}")
    ]).partial(format_instructions=parser.get_format_instructions())
    llm = ChatOpenAI(model="gpt-4o-mini",
                     openai_api_key=os.getenv('OPENAI_API_KEY'), verbose=False)

    client = MultiServerMCPClient({
        "mapbox": {
            "command": "node",
            "transport": "stdio",
            "args": ["/home/fe-mshalahuddin/dev/research/mapbox-mcp-server/dist/index.js"],
            "env": {
                "MAPBOX_ACCESS_TOKEN": os.getenv('MAPBOX_ACCESS_TOKEN')
            }
        }
    })
    mapbox_tools = await client.get_tools()

    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=[*mapbox_tools, save_tool],
    )

    # agent = initialize_agent(
    #     tools=[*mapbox_tools, save_tool], llm=llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS,)

    agent_executor = AgentExecutor(
        agent=agent, tools=[*mapbox_tools, save_tool], verbose=True)

    async for chunk in agent_executor.astream({
        "query": "Based on public transportation schedules, when I should go from 'Alfamart Rawa Belong 2, Palmerah, Indonesia' to 'PT. Indonesia Indicator, Tangerang Selatan, Indonesia' that takes the fastest time."
    }):
        print(chunk, end="\n")


asyncio.run(main())
