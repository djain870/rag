from src._1config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src._2parser import parse_pdf, clean_text, get_first_pdf
from src._3chunker import chunk_text
from src._4embedding import embed_chunks, embed_query
from src._5retriever import search_enterprise_knowledge
from src._6generator import call_llm, build_prompt
from src._7vectordb import upsert_embeddings_to_qdrant


def main():
    pdf_path = get_first_pdf(DATA_DIR)
    raw_text = parse_pdf(pdf_path)
    print("parse pdf done")
    cleaned_text = clean_text(raw_text)
    chunk_texts = chunk_text(
        cleaned_text,
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )
    chunks = [
        {
            "text": chunk,
            "source": pdf_path.name,
            "chunk_id": i
        }
        for i, chunk in enumerate(chunk_texts)
    ]
    print(f"Total chunks created: {len(chunks)}")
    embeddings = embed_chunks(chunk_texts)
    print("Embeddings generated for all chunks.")
    upsert_embeddings_to_qdrant(embeddings, chunks)
    query = "Who authored document titled 100G Networking technology overview"
    print(f"Query: {query}")
    query_embedding = embed_query(query)
    context = search_enterprise_knowledge(query_embedding)
    prompt = build_prompt(query, context)
    response = call_llm(prompt)
    print(f"Response: {response}")
    return response

if __name__ == "__main__":
    main()

