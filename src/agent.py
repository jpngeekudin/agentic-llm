import wikipedia
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType


def wiki_search(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Error: {str(e)}"


def calculataor_tool(expr: str) -> str:
    try:
        result = eval(expr)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# llm = OpenAI(temperature=0, openai_api_key="aaa")
llm = ChatOllama(model="qwen3")

agent = initialize_agent([
    Tool(name="Wikipedia", func=wiki_search,
         description="Use for answering general knowledge or 'who is / what is' questions"),
    Tool(name="Calculator", func=calculataor_tool,
         description="Use for solving math expression")
], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)


if __name__ == '__main__':
    print("\n--- Example 1: Math ---")
    result1 = agent.invoke("What is 25 * 4 + 3?")
    print("Answer:", result1)

    print("\n--- Example 2: Wiki ---")
    result2 = agent.invoke("Who is Nikola Tesla?")
    print("Answer:", result2)
