import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

# from create_db import get_embedding_function, create_database, clear_database
# embedding_function = OllamaEmbeddings(model="mxbai-embed-large")


CHROMA_PATH = "db/multimodal_rag"
FILE_PATH = ""
PROMPT_TEMPLATE = """
Answer the question based only on the following context and do not hallucinate any values or confuse one electrolyte with another and try to remember properties of each electrolyte separately:

{context}

---

Answer the question based on the above context: {question} and return the answer in a JSON format.
"""
def get_embedding_function():
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    return embedding

def main():
    global CHROMA_PATH, FILE_PATH
    # Create CLI.
    FILE_PATH = input("Enter file path: ")
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    # clear_database()
    # create_database(FILE_PATH)
    CHROMA_PATH = CHROMA_PATH + "/" + FILE_PATH.split("/")[-1].split(".")[0]
    print(f"Chroma Path: {CHROMA_PATH}")
    response = query_rag(query_text)
    print(response)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=30)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # model = Ollama(model="mistral", num_ctx=32768)
    model = Ollama(model="llama3.1", num_ctx=16384)
    # model = Ollama(model="deepseek-r1:8b", num_ctx=16384)
    # model = Ollama(model="granite3.1-dense", num_ctx=16384)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)
    return formatted_response

if __name__ == "__main__":
    main()