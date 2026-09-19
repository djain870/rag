from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.parser import parse_pdf, clean_text, get_first_pdf
from src.chunker import chunk_text
from src.embedding import embed_chunks, embed_query
from src.retriever import top3_chunks_combined_context
from src.generator import call_llm, build_prompt


def main():
    pdf_path = get_first_pdf(DATA_DIR)
    raw_text = parse_pdf(pdf_path)
    print("parse pdf done")
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(
        cleaned_text,
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )
    print(f"Total chunks created: {len(chunks)}")
    embeddings = embed_chunks(chunks)
    print("Embeddings generated for all chunks.")
    query = "Who authored document titled 100G Networking technology overview"
    print(f"Query: {query}")
    query_embedding = embed_query(query)
    context = top3_chunks_combined_context(query_embedding, chunks, embeddings)
    prompt = build_prompt(query, context)
    response = call_llm(prompt)
    print(f"Response: {response}")
    return response

if __name__ == "__main__":
    main()

