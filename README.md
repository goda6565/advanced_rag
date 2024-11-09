# RAGの発展的な手法を実装
## 手法
### HyDE
```
""" HyDE """

hypothetical_prompt = ChatPromptTemplate.from_template("""
Write one sentence answering the following question in English
\n\n
question: {question}
""")

# HyDE chain
hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

""""""
```
### Multi Query
```
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
```
### RAG Fusion
```
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

""""""
```
### Hybrid Search
```
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
```