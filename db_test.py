from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.schema.document import Document
import argparse
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
    data_path = data_path.replace("\\", "/")
    document_loader = PyPDFLoader(data_path)
    return document_loader.load()

DATA_PATH = input("Path for data directory: ")
docs = load_documents(DATA_PATH)
print(docs[0])