from langsmith import traceable

from src._1config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from src._2parser import (
    get_document_hash,
    parse_and_chunk,
)
from src._3chunker_unused import chunk_text
from src._4embedding import embed_chunks
from src._7vectordb import (
    delete_document,
    retrieve_document_hash,
    upsert_embeddings_to_qdrant,
)
from src._9logger import logger


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


@traceable(name="Ingest Document")
def ingest_document(doc_path, doc_hash):
    '''To be called only when document is not present in vectordb'''
    chunk_text_meta = parse_and_chunk(doc_path, doc_hash)
    # cleaned_pages = clean_text(raw_pages)
    # all_chunks = prepare_chunks(cleaned_pages, doc_path, doc_hash)
    logger.info("Total chunks created: %s", len(chunk_text_meta))
    embeddings = embed_chunks([c["text"] for c in chunk_text_meta])
    # print("Embeddings generated for all chunks.")
    upsert_embeddings_to_qdrant(embeddings, chunk_text_meta)


# @traceable(name="Process Documents")
# def process_documents():
#     '''To be called if need to ingest new documents or documents have changed'''
#     pdf_paths = get_all_pdfs(DATA_DIR)
#     new_pdfs = 0
#     changed_pdfs = 0
#     skipped_pdfs = 0
#     pdfs_processed = 0
#     #ingest only those documents which are not present or have changed in vectordb
#     for pdf_path in pdf_paths:
#         doc_hash = get_document_hash(pdf_path)
#         logger.info("Doc hash generated")
#         stored_hash = retrieve_document_hash(pdf_path.name)
#         #compare with stored hash
#         if stored_hash is None:
#             ingest_document(pdf_path, doc_hash)
#             new_pdfs += 1
#             pdfs_processed += 1
#         elif stored_hash == doc_hash:
#             logger.info("Skipping as pdf is already ingested and no change detected %s", pdf_path.name)
#             skipped_pdfs += 1
#             pdfs_processed += 1
#             continue
#         else:
#             logger.info("Document %s has changed. Updating...", pdf_path.name)
#             delete_document(pdf_path.name)
#             ingest_document(pdf_path, doc_hash)
#             changed_pdfs += 1
#             pdfs_processed += 1
#     logger.info("New PDFs: %s", new_pdfs)
#     logger.info("Changed PDFs: %s", changed_pdfs)
#     logger.info("Total PDFs processed: %s", pdfs_processed)
#     return{
#         "new_pdfs": new_pdfs,
#         "changed_pdfs": changed_pdfs,
#         "skipped_pdfs": skipped_pdfs,
#         "total_pdfs": pdfs_processed
#     }


@traceable(name="Process Documents")
def process_documents():
    '''To be called if need to ingest new documents or documents have changed'''
    docs_path = [f for f in DATA_DIR.glob("**/*.*") if f.suffix in [".pdf", ".pptx", ".docx", ".html", ".txt"]]
    new_docs = 0
    changed_docs = 0
    skipped_docs = 0
    docs_processed = 0
    #ingest only those documents which are not present or have changed in vectordb
    for doc_path in docs_path:
        # print("doc path: ", doc_path)
        doc_hash = get_document_hash(doc_path)
        # print("doc hash: ", doc_hash)
        logger.info("Doc hash generated")
        stored_hash = retrieve_document_hash(doc_path.name)
        print("stored hash: ", stored_hash)
        #compare with stored hash
        if stored_hash is None:
            ingest_document(doc_path, doc_hash)
            # print("ingested")
            new_docs += 1
            docs_processed += 1
        elif stored_hash == doc_hash:
            logger.info("Skipping as pdf is already ingested and no change detected %s", doc_path.name)
            skipped_docs += 1
            docs_processed += 1
            continue
        else:
            logger.info("Document %s has changed. Updating...", doc_path.name)
            delete_document(doc_path.name)
            # print("deleting doc in db")
            ingest_document(doc_path, doc_hash)
            changed_docs += 1
            docs_processed += 1
    logger.info("New Documents: %s", new_docs)
    logger.info("Changed Documents: %s", changed_docs)
    logger.info("Total Documents processed: %s", docs_processed)
    return{
        "new_docs": new_docs,
        "changed_docs": changed_docs,
        "skipped_docs": skipped_docs,
        "total_docs": docs_processed
    }