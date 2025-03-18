import os
import uuid
from unstructured.partition.pdf import partition_pdf
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.embeddings.ollama import OllamaEmbeddings

os.environ["PATH"] += r";C:\Program Files\Tesseract-OCR"

# ✅ Set ChromaDB storage path
CHROMA_PATH = "db/multimodal_rag/34"

# ✅ Use Ollama (Local LLM)
ollama_model = ChatOllama(model="llama3.1", num_ctx=26384)

# ✅ Set up embedding function for vector storage
embedding_function = OllamaEmbeddings(model="nomic-embed-text", num_ctx=2048)

# ✅ Ensure persistent ChromaDB storage
if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)

# ✅ File path
output_path = "./content/"
file_path = input("Enter file path: ")
file_path = file_path.replace("\\", "/")

# ✅ Extract PDF elements (ONLY TEXT & TABLES)
chunks = partition_pdf(
    filename=file_path,
    infer_table_structure=True,   # Extract tables
    strategy="hi_res",            # Required for table extraction
    chunking_strategy="by_title",
    max_characters=4000,
    combine_text_under_n_chars=1000,
    new_after_n_chars=2000,
)

# ✅ Separate extracted elements into text and tables
tables, texts = [], []

for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)
    elif "CompositeElement" in str(type(chunk)):
        texts.append(chunk)

# ✅ Store texts as they are (no summarization)
text_summaries = [chunk.text for chunk in texts]

# ✅ Convert tables to readable text format
def convert_table_to_text(table_chunk):
    """Converts table content to readable Markdown-like text."""
    table_html = table_chunk.metadata.text_as_html
    rows = table_html.split("<tr>")[1:]  # Extract table rows
    formatted_table = ["Table Data:\n"]

    for row in rows:
        cells = row.replace("</td>", "|").replace("<td>", "").strip()
        formatted_table.append(cells)

    return "\n".join(formatted_table)

# ✅ Convert all tables to text format
# table_texts = [convert_table_to_text(table) for table in tables]

# ✅ Set up persistent ChromaDB
vectorstore = Chroma(collection_name="multi_modal_rag", persist_directory=CHROMA_PATH, embedding_function=embedding_function)
store = InMemoryStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# ✅ Function to generate unique document IDs
def generate_chunk_id(source, page, index):
    return f"{source}:{page}:{index}"

# ✅ Check existing database and add only new documents
def add_to_chroma_text(elements: list[Document], category="text"):
    existing_items = vectorstore.get(include=[])  # IDs are always included
    existing_ids = set(existing_items["ids"])

    new_documents = []
    chunk_ids = []

    for idx, element in enumerate(elements):
        metadata = element.metadata
        element.metadata.pop("languages")
        element.metadata.pop("orig_elements")
        source = metadata.get('filename', 'unknown')
        page = metadata.get('page_number', 0)
        chunk_id = generate_chunk_id(source, page, idx)
        # type_ = metadata.get('filetype', "unknown")
        
        # # Create a Document object with metadata
        # doc = Document(
        #     page_content=element,
        #     metadata={
        #         "id": chunk_id,
        #         "source": source,
        #         "page_number": page,
        #         "type": type_
        #     }
        # )
        new_documents.append(element)
        chunk_ids.append(chunk_id)

    # Add documents to vectorstore
    if new_documents:
        vectorstore.add_documents(new_documents, ids=chunk_ids)
        vectorstore.persist()
        print(f"Added {len(new_documents)} new documents to ChromaDB.")
    else:
        print("No new documents to add.")



# ✅ Add raw texts to ChromaDB
add_to_chroma_text([Document(page_content=text.text, metadata=text.metadata.to_dict()) for text in texts], "text")

# ✅ Add converted tables to ChromaDB
if len(tables):
    add_to_chroma_text([Document(page_content=convert_table_to_text(table_text), metadata=table_text.metadata.to_dict()) for table_text in tables], "table")
