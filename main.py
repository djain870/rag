from src._1config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src._2parser import parse_pdf, clean_text, get_first_pdf
from src._3chunker import chunk_text
from src._4embedding import embed_chunks, embed_query
from src._5retriever import search_enterprise_knowledge
from src._6generator import call_llm, build_prompt, build_context
from src._7vectordb import upsert_embeddings_to_qdrant


def prepare_chunks(cleaned_pages, pdf_path):
    all_chunks = []
    chunk_id = 0
    for page in cleaned_pages:
        page_chunks = chunk_text(
            page["text"],
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP
        )
        for chunk in page_chunks:
            all_chunks.append({
                "text": chunk,
                "source": pdf_path.name,
                "page": page["page"],
                "chunk_id": chunk_id,
            })
            chunk_id +=1
    return all_chunks




def main():
    pdf_path = get_first_pdf(DATA_DIR)
    raw_pages = parse_pdf(pdf_path)
    print("parse pdf done")
    cleaned_pages = clean_text(raw_pages)
    all_chunks = prepare_chunks(cleaned_pages, pdf_path)
    print(f"Total chunks created: {len(all_chunks)}")
    embeddings = embed_chunks([chunk["text"] for chunk in all_chunks])
    print("Embeddings generated for all chunks.")
    upsert_embeddings_to_qdrant(embeddings, all_chunks)
    query = "Who authored document titled 100G Networking technology overview"
    print(f"Query: {query}")
    query_embedding = embed_query(query)
    context = build_context(query_embedding)
    prompt = build_prompt(query, context)
    response = call_llm(prompt)
    print(f"Response: {response}")
    return response

if __name__ == "__main__":
    main()

