from unstructured.partition.pdf import partition_pdf
from langchain.schema.document import Document
# from unstructured.staging.base import elements_from_base64_gzipped_json
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import os
from io import BytesIO
import re
from langchain_community.llms.ollama import Ollama

os.environ["PATH"] += r";C:\Program Files\Tesseract-OCR"

def clean_text(text):
    # Replace consecutive non-alphabetic characters with a space
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add spaces between joined words
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)  # Add spaces between numbers and letters
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)  # Add spaces between letters and numbers
    return text

def get_chunks(file_path: str):
    chunks = partition_pdf(
    filename=file_path,
    infer_table_structure=True,   # Extract tables
    strategy="hi_res",            # Required for table extraction
    chunking_strategy="by_title",
    max_characters=10000,
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000,
    )
    return chunks

file_path = "C:/Users/nandi/OneDrive/Documents/battery_components_extractor/battery_components_extractor/data/f5/27.pdf"
chunks = get_chunks(file_path=file_path)

tables, texts = [], []

for chunk in chunks:
    if "Table" in str(type(chunk)):
        tables.append(chunk)
    elif "CompositeElement" in str(type(chunk)):
        texts.append(chunk)

# ✅ Store texts as they are (no summarization)
text_summaries = [chunk.text for chunk in texts]

def convert_table_to_text(table_chunk):
    """Converts table content to readable Markdown-like text."""
    table_html = table_chunk.metadata.text_as_html
    rows = table_html.split("<tr>")[1:]  # Extract table rows
    formatted_table = ["Table Data:\n"]

    for row in rows:
        cells = row.replace("</td>", "|").replace("<td>", "").strip()
        formatted_table.append(cells)

    return "\n".join(formatted_table)

def generate_chunk_id(source, page, index):
    return f"{source}:{page}:{index}"

table_texts = [convert_table_to_text(table) for table in tables]

print(texts[0].metadata.to_dict().get("filename"))
print(texts[0].text)

d1 = Document(page_content=texts[0].text, metadata=texts[0].metadata.to_dict())
print(d1)

# metadata = texts[0].metadata.to_dict()
# print(f"metadata: {metadata}")
# source = metadata.get('filename', 'unknown')
# page = metadata.get('page_number', 0)
# chunk_id = generate_chunk_id(source, page, 0)

# print(chunks[0])
# print("----------------------------")
# print(chunks[1].metadata)
# elements = chunks[0].metadata.orig_elements
# chunk_texts = [el for el in elements]
# print(chunk_texts[0].to_dict())
# print("----------------------------")
# # print(chunks[0])
# # print(texts[0])
# metadata = chunks[11].metadata.to_dict()
# print(metadata)
# print("----------------------------")
# # orig_elements = elements_from_base64_gzipped_json(metadata["orig_elements"])
# # print(metadata)
# # for orig_element in orig_elements:
# #     print(f"    {orig_element.category}: {orig_element.text}")

# print(clean_text(chunks[11].text))
# # print(chunk_texts[1].to_dict())

# # print(len(chunks))

# tables = []
# texts = []

# for chunk in chunks:
#     if "Table" in str(type(chunk)):
#         tables.append(chunk)

#     if "CompositeElement" in str(type((chunk))):
#         texts.append(chunk)

# # Get the images from the CompositeElement objects
# def get_images_base64(chunks):
#     images_b64 = []
#     for chunk in chunks:
#         if "CompositeElement" in str(type(chunk)):
#             chunk_els = chunk.metadata.orig_elements
#             for el in chunk_els:
#                 if "Image" in str(type(el)):
#                     images_b64.append(el.metadata.image_base64)
#     return images_b64

# images = get_images_base64(chunks)

# import base64
# # from IPython.display import Image, display
# from PIL import Image

# def display_base64_image(base64_code):
#     # Decode the base64 string to binary
#     image_data = base64.b64decode(base64_code)
#     # Display the image
#     # display(Image(data=image_data))
#     image = Image.open(BytesIO(image_data))
#     image.show() 

# display_base64_image(images[1])

# # Prompt
# prompt_text = """
# You are an assistant tasked with summarizing tables and text.
# Give a concise summary of the table or text.

# Respond only with the summary, no additionnal comment.
# Do not start your message by saying "Here is a summary" or anything like that.
# Just give the summary as it is.

# Table or text chunk: {element}

# """
# prompt = ChatPromptTemplate.from_template(prompt_text)

# # Summary chain
# model = Ollama(temperature=0.5, model="deepseek-r1:8b")
# summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# # Summarize text
# text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

# # Summarize tables
# tables_html = [table.metadata.text_as_html for table in tables]
# table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

# print(text_summaries[0])

# prompt_template = """Describe the image in detail. For context,
#                   the image is part of a research paper explaining the transformers
#                   architecture. Be specific about graphs, such as bar plots."""
# messages = [
#     (
#         "user",
#         [
#             {"type": "text", "text": prompt_template},
#             {
#                 "type": "image_url",
#                 "image_url": {"url": "data:image/jpeg;base64,{image}"},
#             },
#         ],
#     )
# ]

# prompt = ChatPromptTemplate.from_messages(messages)

# chain = prompt | Ollama(model="deepseek-r1:8b") | StrOutputParser()


# image_summaries = chain.batch(images)
# print(image_summaries[1])

