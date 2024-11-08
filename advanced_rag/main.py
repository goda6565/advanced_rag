from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import GitLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pydantic import BaseModel, Field
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

model = ChatOpenAI(model="gpt-4o-mini", streaming=True)

""" HyDE """

hypothetical_prompt = ChatPromptTemplate.from_template("""
Write one sentence answering the following question in English
\n\n
question: {question}
""")

# HyDE chain
hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

""""""

""" 複数の検索クリエ """

class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クリエのリスト")
    
query_generation_prompt = ChatPromptTemplate.from_template("""
Generate three different search queries to retrieve relevant documents from the vector database for a question in English. 
The goal is to provide multiple views of the user's question to overcome the limitations of distance-based similarity search.
\n\n
question: {question}
""")

query_generation_chain = query_generation_prompt | model.with_structured_output(QueryGenerationOutput) | (lambda x: x.queries)

""""""
""" After Retrieval """

""" RAG-FUSION """

def reciprocal_rank_fusion(retriever_outputs: list[list[Document]], k: int = 60) -> list[str]:
    # 各ドキュメントのコンテンツとそのスコアを保持する辞書
    content_score_mapping = {}
    
    # 検索クリエごとにループ
    for docs in retriever_outputs:
        # 各クリエのドキュメントごとにループ
        for rank, doc in enumerate(docs):
            content = doc.page_content
            
            # 初めて登場したコンテンツの場合は、0で初期化
            if content not in content_score_mapping:
                content_score_mapping[content] = 0
                
            content_score_mapping[content] += 1 / (rank + k)
            
    # スコアの大きい順にソート
    ranked = sorted(content_score_mapping.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in ranked]


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
        "context": query_generation_chain | retriever.map() | reciprocal_rank_fusion,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

runnable.invoke("chainlitって何？")