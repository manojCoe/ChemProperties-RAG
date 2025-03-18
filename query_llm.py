import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

# from createDataBase import get_embedding_function

CHROMA_PATH = "db"
FILE_PATH = ""
PROMPT_TEMPLATE = """
Answer the question based only on the following context and do not hallucinate any values, confuse one electrolyte with another and try to remember properties of each electrolyte separately:

{context}

---

Answer the question based on the above context: {question} and return the answer in a JSON format.
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    loader = PyPDFLoader("C:/Users/nandi/OneDrive/Documents/battery_components_extractor/battery_components_extractor/data/f5/29.pdf")
    documents = loader.load()  # Extract text from the entire PDF
    pdf_text = " ".join([doc.page_content for doc in documents])
    response = query_rag(query_text, pdf_text)
    print(response)


def query_rag(query_text: str, context_text: str):

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # model = Ollama(model="mistral")
    # model = Ollama(model="llama3.1")
    model = Ollama(model="deepseek-r1:8b", num_ctx=30000)
    # model = Ollama(model="granite3.1-dense", num_ctx=30000)
    response_text = model.invoke(prompt)
    # formatted_response = f"Response: {response_text}"
    # print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()