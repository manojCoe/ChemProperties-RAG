from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.schema.document import Document
from transformers import GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
import argparse
from unstructured.partition.auto import partition
import os
import shutil
import sys
from utils.unit_conversion import convert_units_dynamic

# Set default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

CHROMA_PATH = "db"
DATA_PATH = "data"


def load_documents(data_path: str):
    # C:\Users\nandi\OneDrive\Documents\battery_components_extractor\battery_components_extractor\data
    global CHROMA_PATH
    data_path = data_path.replace("\\", "/")
    CHROMA_PATH = CHROMA_PATH + "/" + data_path.split("/")[-1].split(".")[0]
    print(f"Chroma_Path: {CHROMA_PATH}")
    document_loader = PyPDFLoader(data_path)
    return document_loader.load()

def load_unstructured_pdf(data_path: str):
    elements = partition(filename=data_path, content_type="application/pdf")
    print("\n\n".join([str(el) for el in elements][:10]))


# def split_documents(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size = 1200,
#         chunk_overlap = 110,
#         length_function = len,
#         is_separator_regex=False
#     )
#     return text_splitter.split_documents(documents)

def split_documents(documents: list[Document]):
    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    # tokenizer, chunk_size=100, chunk_overlap=0
    # )
    # return text_splitter.split_documents(documents)
    splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=2)
    return splitter.split_documents(documents)

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
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

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
    documents = load_documents(DATA_PATH)
    # documents = SemanticChunker(documents)
    print(type(documents))
    # print(documents[1].page_content)
    # for doc in documents:
    #     doc.page_content = convert_units_dynamic(doc)
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    return

# documents = load_documents()
# print(len(documents))
# chunks = split_documents(documents)
# print(chunks[1])

if __name__ == "__main__":
    create_database()