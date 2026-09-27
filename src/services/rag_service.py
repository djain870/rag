from langsmith import traceable
import asyncio

from src._4embedding import embed_query
from src._8reranker import rerank_documents
from src._6generator import build_context, build_prompt, call_llm
from src._9logger import logger



@traceable(name="Get Query Response")
async def get_query_response(query):
    query_embedding = embed_query(query)
    logger.info("Query embedding generated")
    results = build_context(query_embedding)
    logger.info("Results retrieved from db: %s", results)
    documents = [doc["text"] for doc in results]
    reranked_docs = rerank_documents(query, documents, top_k=3)
    prompt = build_prompt(query, reranked_docs)
    response = await call_llm(prompt)
    print(response)
    return {
        "answer": response,
        "vector_results": results,
        "reranked_docs": reranked_docs,
        "query": query
        }