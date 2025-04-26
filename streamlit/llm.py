from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain import hub

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    embedding_function = embeddings,
    persist_directory = "./chroma",
    collection_name = "tax",
)

prompt = hub.pull("rlm/rag-prompt")

def get_ai_message(query):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = vector_store.as_retriever(),
        chain_type_kwargs = {"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain.invoke({"query": query})
