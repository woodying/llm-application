# 1. 문서의 내용을 읽는다
# 2. 문서를 쪼갠다
#  - 토큰 수 초과로 답변을 생성하지 못할 수 있음
#  - 문서가 길면 (인풋이 길면) 답변 생성이 오래 걸림
# 3. 임베딩 -> 벡터 DB에 저장한다
# 4. 사용자의 질문에 대해서 벡터 DB에서 유사도가 높은 문서를 찾는다
# 5. 유사도가 높은 문서를 LLM 에 질문과 함께 전달

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains import RetrievalQA
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

# loader = Docx2txtLoader("./tax.docx")
# documents = loader.load_and_split(text_splitter)

# print(len(documents))

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# vector_store = Chroma.from_documents(
#     documents = documents, 
#     embedding = embeddings,
#     persist_directory = "./chroma",
#     collection_name = "tax",
# )

vector_store = Chroma(
    embedding_function = embeddings,
    persist_directory = "./chroma",
    collection_name = "tax",
)

###
query = "연봉 5천만원인 직장인의 소득세는 얼마인가요?"

retrieved_docs = vector_store.similarity_search(query)

print(retrieved_docs)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = f"""[Identity]
- 당신은 최고의 한국 소득세 전문가 입니다
- [Context]를 참고해서 사용자의 질문에 답변해주세요

[Context]
{retrieved_docs}

[Question]
{query}
"""

response = llm.invoke(prompt)

print(response.content)

###
prompt = hub.pull("rlm/rag-prompt")
print(prompt)

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = vector_store.as_retriever(),
    chain_type_kwargs = {"prompt": prompt}
)

message = qa_chain.invoke({"query": query})

print(message)
