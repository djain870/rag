from src._1config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src._2parser import parse_pdf, clean_text, get_first_pdf, get_all_pdfs, get_document_hash
from src._3chunker import chunk_text
from src._4embedding import embed_chunks, embed_query
from src._5retriever import search_enterprise_knowledge
from src._6generator import call_llm, build_prompt, build_context
from src._7vectordb import upsert_embeddings_to_qdrant, check_document_exists, retrieve_document_hash, delete_document


def prepare_chunks(cleaned_pages, pdf_path, doc_hash):
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
                "doc_hash": doc_hash,
                "chunk_id": chunk_id,
            })
            chunk_id +=1
    return all_chunks

def ingest_document(pdf_path, doc_hash):
    raw_pages = parse_pdf(pdf_path)
    cleaned_pages = clean_text(raw_pages)
    all_chunks = prepare_chunks(cleaned_pages, pdf_path, doc_hash)
    print(f"Total chunks created: {len(all_chunks)}")
    embeddings = embed_chunks([c["text"] for c in all_chunks])
    # print("Embeddings generated for all chunks.")
    upsert_embeddings_to_qdrant(embeddings, all_chunks)

def get_query_response(query):
    query_embedding = embed_query(query)
    # results = search_enterprise_knowledge(
    #     query_vector = query_embedding
    # )
    context = build_context(query_embedding)
    prompt = build_prompt(query, context)
    response = call_llm(prompt)
    return response


def main():
    pdf_paths = get_all_pdfs(DATA_DIR)
    #ingest only those documents which are not present or have changed in vectordb
    for pdf_path in pdf_paths:
        doc_hash = get_document_hash(pdf_path)
        stored_hash = retrieve_document_hash(pdf_path.name)
        #compare with stored hash
        if stored_hash is None:
            ingest_document(pdf_path, doc_hash)
        elif stored_hash == doc_hash:
            print(f"Skipping {pdf_path.name}")
            continue
        else:
            print(f"Document {pdf_path.name} has changed. Updating...")
            delete_document(pdf_path.name)
            ingest_document(pdf_path, doc_hash)

    query = "Who authored document titled 100G Networking technology overview"
    response = get_query_response(query)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()

