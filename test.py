from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.schema.document import Document
# from transformers import GPT2TokenizerFast
# from sentence_transformers import SentenceTransformer
import argparse
import os
import shutil
import sys
import uuid
from utils.unit_conversion import convert_units_dynamic
from parser import get_chunks

# Set default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

CHROMA_PATH = "db"
DATA_PATH = "data"


def get_embedding_function():
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    return embedding
    # model = SentenceTransformer("gbyuvd/ChemEmbed-v01")
    # return model.encode(chunks)

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # type_ = chunk.metadata.orig_elements
        # page = chunk.metadata.get("page")
        # current_page_id = f"{source}:{page}"
        chunk_texts = 
        metadaata = chunk.metadata.to_dict()

        # If the page ID is the same as the last one, increment the index.
        chunk_id = str(uuid.uuid4())

        # Calculate the chunk ID.
        # chunk_id = f"{current_page_id}:{current_chunk_index}"
        # last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id
        chunk.metadata = 

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def create_database(file_path = None):
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
        print("Cleared the database.")
        return
    
    if not file_path:
        DATA_PATH = input("Path for data directory: ")
    else:
        DATA_PATH = file_path

    # Create (or update) the data store.
    # documents = load_documents(DATA_PATH)
    # # documents = SemanticChunker(documents)
    # print(type(documents))
    # # print(documents[1].page_content)
    # # for doc in documents:
    # #     doc.page_content = convert_units_dynamic(doc)
    # chunks = split_documents(documents)
    chunks = get_chunks(DATA_PATH)
    add_to_chroma(chunks)
    return

# documents = load_documents()
# print(len(documents))
# chunks = split_documents(documents)
# print(chunks[1])

if __name__ == "__main__":
    create_database()