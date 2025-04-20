from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_upstage import ChatUpstage
from langchain_ollama import ChatOllama

load_dotenv()

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = ChatUpstage(model="solar-pro", temperature=0)
llm = ChatOllama(model="llama3.2", temperature=0)

message = llm.invoke("Hello, world!").content

print(message)
