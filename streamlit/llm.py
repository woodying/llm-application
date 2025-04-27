from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain import hub

load_dotenv()

def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        embedding_function = embeddings,
        persist_directory = "./chroma",
        collection_name = "tax",
    )
    return vector_store.as_retriever(search_kwargs={'k': 4})


def get_llm(model='gpt-4o-mini'):
    return ChatOpenAI(model=model)

prompt = hub.pull("rlm/rag-prompt")

qa_chain = RetrievalQA.from_chain_type(
    llm = get_llm(),
    retriever = get_retriever(),
    chain_type_kwargs = {"prompt": prompt},
    return_source_documents=True
)


def get_ai_message(query):
    return qa_chain.invoke({"query": query})['result']
