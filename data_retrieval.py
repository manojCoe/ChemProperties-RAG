import os
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ✅ Set ChromaDB storage path
CHROMA_PATH = "db/multimodal_rag/34"

# ✅ Ensure database exists
if not os.path.exists(CHROMA_PATH):
    raise FileNotFoundError("❌ Error: No database found! Run `create_db.py` first.")

# ✅ Load embedding function
embedding_function = OllamaEmbeddings(model="nomic-embed-text", num_ctx=2048)

# ✅ Load ChromaDB
vectorstore = Chroma(collection_name="multi_modal_rag", persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# ✅ Set up retriever
# retriever = MultiVectorRetriever(
#     vectorstore=vectorstore,
#     docstore=None,  # Not needed since we only query stored data
#     id_key="doc_id",
# )

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})


# ✅ Function to parse retrieved documents
def parse_docs(docs):
    """Only handle text-based retrieval."""
    return {"texts": docs}

# ✅ Function to format prompt for LLM
def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    # print(docs_by_type["texts"])

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.dict().get("page_content")


    # ✅ Only process text context
    # context_text = "\n".join([tfor t in docs_by_type["texts"]])
    # context_text = "\n".join([doc for doc in docs_by_type])
    print(f"length of context: {len(context_text)}")

    prompt_template = f"""
    Answer the question based only on the following context which can include text and tables and do not hallucinate any values or confuse one electrolyte with another and try to remember properties and their values to the best of your understanding of each electrolyte separately. The output should be in a JSON format with properties and respective values:
    Context: {context_text}
    Question: {user_question}
    """

    return ChatPromptTemplate.from_messages([HumanMessage(content=[{"type": "text", "text": prompt_template}])])

# ✅ Use Ollama for inference
ollama_model = ChatOllama(model="deepseek-r1:8b", temperature=0, num_ctx=26384)

chain = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | ollama_model
    | StrOutputParser()
)

# ✅ Interactive Querying
def query_rag():
    print("\n🔎 MultiModal RAG Chat with Mistral (Local Ollama)")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_query = input("💬 Ask a question: ")
        if user_query.lower() == "exit":
            print("👋 Exiting RAG Chat.")
            break
        
        response = chain.invoke(user_query)
        print("\n🧠 Answer:\n", response, "\n" + "-"*50 + "\n")

# ✅ Run interactive chat
if __name__ == "__main__":
    query_rag()
