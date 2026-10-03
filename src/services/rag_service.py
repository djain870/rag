
from langsmith import traceable

from src.embedding import embed_query
from src.generator import build_context, build_prompt, generate_response
from src.logger import logger
from src.reranker import rerank_documents


@traceable(name="Get Query Response")
async def get_query_response(query):
    query_embedding = embed_query(query)
    logger.info("Query embedding generated")
    results = build_context(query_embedding)
    # logger.info("Results retrieved from db: %s", results)
    documents = [doc["text"] for doc in results]
    reranked_docs = rerank_documents(query, documents, top_k=3)
    prompt = build_prompt(query, reranked_docs)
    response = await generate_response(prompt)
    # logger.info("LLM response generated: %s", response)
    return {
        "answer": response,
        "vector_results": results,
        "reranked_docs": reranked_docs,
        "query": query
        }