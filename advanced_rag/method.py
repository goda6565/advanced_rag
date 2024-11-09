from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


""" HyDE """
def hyde_chain(model):
    
    hypothetical_prompt = ChatPromptTemplate.from_template("""
    Write one sentence answering the following question in English
    \n\n
    question: {question}
    """)

    # HyDE chain
    return  hypothetical_prompt | model | StrOutputParser()

""""""


""" Multi-Query """

class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クリエのリスト")
    
def multi_query_chain(model):
    query_generation_prompt = ChatPromptTemplate.from_template("""
    Generate three different search queries to retrieve relevant documents from the vector database for a question in English. 
    The goal is to provide multiple views of the user's question to overcome the limitations of distance-based similarity search.
    \n\n
    question: {question}
    """)

    return query_generation_prompt | model.with_structured_output(QueryGenerationOutput) | (lambda x: x.queries)

""""""


""" RAG-Fusion """

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

""""""