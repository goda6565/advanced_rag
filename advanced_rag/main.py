from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import GitLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter

""" 手法のインポート """
from advanced_rag.method import multi_query_chain
from advanced_rag.method import reciprocal_rank_fusion

load_dotenv()

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".md")

loader = GitLoader(
    clone_url="https://github.com/Chainlit/chainlit", 
    repo_path="./chainlit",
    branch="main",
    file_filter=file_filter,
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
split_documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(split_documents, embeddings)
retriever = vectorstore.as_retriever()


""" 検索器（Hybrid Search） """

faiss_retriever = retriever.with_config(
    {"run_name": "faiss_retriever"}
)

bm25_retriever = BM25Retriever.from_documents(split_documents).with_config(
    {"run_name": "bm25_retriever"}
)

hybrid_retriever = (
    RunnableParallel({
        "faiss_documents": faiss_retriever,
        "bm25_documents": bm25_retriever,
    })
    | (lambda x: x["faiss_documents"] + x["bm25_documents"])
)

""""""

model = ChatOpenAI(model="gpt-4o-mini", streaming=True)

# 質問プロンプトテンプレート
system_prompt = (
"You are an assistant to assist in the work of the company. "
"Use the following pieces of retrieved context to answer "
"the question. If you don't know the answer, say that you "
"don't know. Use three sentences maximum and keep the "
"answer concise."
"\n\n"
"{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# 実行チェーンの構築
runnable = (
    {
        "context": multi_query_chain(model) | hybrid_retriever.map() | reciprocal_rank_fusion,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

runnable.invoke("chainlitって何？")