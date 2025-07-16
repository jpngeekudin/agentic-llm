
from langchain_ollama import OllamaEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma

loader = DirectoryLoader(
    path="/home/fe-mshalahuddin/dev/research/python-research/src",
    glob=["**/*.py"],
    loader_cls=TextLoader,
    show_progress=True,
)

documents = loader.load()
splitter = RecursiveCharacterTextSplitter()
chunks = splitter.split_documents(documents)
embedding = OllamaEmbeddings(model="mistral")
vector_store = Chroma.from_documents(chunks, embedding=embedding)
retriever = vector_store.as_retriever()

code_search_tool = create_retriever_tool(
    retriever=retriever,
    name="code_search_tool",
    description="Search relevant parts of the codebase to answer questions about project structure, logic, or specific files"
)
