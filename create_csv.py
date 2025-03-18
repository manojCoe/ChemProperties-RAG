import os
from main import main
DIR_PATH = "C:/Users/nandi/OneDrive/Documents/battery_components_extractor/battery_components_extractor/data"

def list_pdf_files(directory):
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def create_csv():
    pdf_files = list_pdf_files(DIR_PATH)
    ...




# Example usage
pdf_files = list_pdf_files(DIR_PATH)
for pdf in pdf_files:
    print(pdf)
