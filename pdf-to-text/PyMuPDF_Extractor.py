import fitz  # PyMuPDF
import os

def convert_pdf_to_text(pdf_path, text_path):
    """
    Convert a PDF file to a text file.
    
    Args:
        pdf_path (str): Path to the PDF file.
        text_path (str): Path where the text file will be saved.
    """
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    text_content = ""
    
    # Iterate through each page in the PDF
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text_content += page.get_text() + "\n"  # Append the text of each page
    
    # Close the PDF document
    pdf_document.close()
    
    # Write the extracted text to the text file
    with open(text_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text_content)

def convert_all_pdfs_in_directory(pdf_directory_path):
    """
    Convert all PDF files in a directory to text files, saving each text file in the same directory as the PDF file.
    
    Args:
        pdf_directory_path (str): Path to the root directory containing PDF files in nested directories.
    """
    # List all directories and files in the root PDF directory
    for root, _, files in os.walk(pdf_directory_path):
        for file_name in files:
            if file_name.lower().endswith(".pdf"):
                pdf_file_path = os.path.join(root, file_name)
                text_file_name = file_name.replace(".pdf", ".txt")
                text_file_path = os.path.join(root, text_file_name)
                
                # Convert the PDF file to a text file
                convert_pdf_to_text(pdf_file_path, text_file_path)
                print(f"Converted {pdf_file_path} to {text_file_path}")

# Path to the root directory containing PDF files in nested directories
pdf_directory_path = "/data/etdrepo/"

# Convert all PDFs in the specified directory to text files
convert_all_pdfs_in_directory(pdf_directory_path)

print("Completed converting all PDF files to text files.")
