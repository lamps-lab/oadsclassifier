import os
import re
import pandas as pd
import string
import fitz  # PyMuPDF
import PyPDF2
from nltk.tokenize import sent_tokenize

# Download NLTK tokenizer data if not already present
import nltk
nltk.download('punkt')

# Define the directory containing your PDF and text files
directory_path = "/content/s2orc"
output_csv_file = "Extracted_URLs.csv"

# Initialize an empty DataFrame to store results
df = pd.DataFrame(columns=["File", "Source", "Sentence", "URL"])

# Function to extract URLs and the corresponding sentences from text content
def extract_urls_and_sentences_from_text(text_content):
    url_pattern = re.compile(r'''(?xi)
    (?:(?:https?|ftp|sftp|file|data|javascript|mailto|tel|git|ssh|magnet):\/\/  # Protocols
      |www\d{0,3}[.]  # www. without protocol
      |[a-z0-9.\-]+[.][a-z]{2,4}\/)  # domain without protocol
    (?:\S+(?::\S*)?@)?  # user:pass authentication
    (?:
        (?!(?:10|127)(?:\.\d{1,3}){3})  # exclude private & local networks
        (?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})
        (?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})
        (?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])  # IP address
        (?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}
        (?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))
    |
        (?:  # host name
            (?:
                [a-z0-9\u00a1-\uffff]
                [a-z0-9\u00a1-\uffff_-]{0,62}
            )?
            [a-z0-9\u00a1-\uffff]\.
        )*
        (?:[a-z\u00a1-\uffff]{2,}\.?)  # domain name
    )
    (?::\d{2,5})?  # port number
    (?:[/?#][^\s]*)?  # resource path, including special characters
    (?:[^\s]*)  # continue matching non-whitespace characters
    ''', re.DOTALL)

    sentences = sent_tokenize(text_content)
    results = []

    for sentence in sentences:
        urls = url_pattern.findall(sentence)
        for url in urls:
            url = url.rstrip(string.punctuation.replace('/', ''))
            results.append((sentence, url))

    return results

# Function to extract URLs from annotations in a PDF
def extract_urls_from_annotations(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        urls = []

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            if "/Annots" in page:
                annotations = page["/Annots"]
                for annotation in annotations:
                    annotation_obj = annotation.get_object()
                    if "/A" in annotation_obj and "/URI" in annotation_obj["/A"]:
                        uri = annotation_obj["/A"]["/URI"]
                        urls.append(uri)

        return urls

# Function to extract URLs from hyperlinked images in a PDF
def extract_hyperlinked_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    hyperlinked_urls = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        links = page.get_links()

        for link in links:
            if 'uri' in link and link['kind'] == fitz.LINK_URI:
                hyperlinked_urls.append(link['uri'])

    return hyperlinked_urls

# Process each file in the directory
for root, dirs, files in os.walk(directory_path):
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith(".pdf"):
            pdf_document = fitz.open(file_path)

            # Extract URLs and sentences from text content
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text_content = page.get_text("text")
                url_sentence_pairs = extract_urls_and_sentences_from_text(text_content)

                for sentence, url in url_sentence_pairs:
                    new_row = pd.DataFrame({
                        "File": [os.path.basename(file_path)],
                        "Source": ["PlainText"],
                        "Sentence": [sentence],
                        "URL": [url]
                    })
                    df = pd.concat([df, new_row], ignore_index=True)

            # Extract URLs from annotations
            annotation_urls = extract_urls_from_annotations(file_path)
            for url in annotation_urls:
                new_row = pd.DataFrame({
                    "File": [os.path.basename(file_path)],
                    "Source": ["Annotation"],
                    "Sentence": [""],  # No sentence context for annotations
                    "URL": [url]
                })
                df = pd.concat([df, new_row], ignore_index=True)

            # Extract URLs from hyperlinked images
            hyperlinked_image_urls = extract_hyperlinked_images_from_pdf(file_path)
            for url in hyperlinked_image_urls:
                new_row = pd.DataFrame({
                    "File": [os.path.basename(file_path)],
                    "Source": ["Hyperlinked Image"],
                    "Sentence": [""],  # No sentence context for hyperlinked images
                    "URL": [url]
                })
                df = pd.concat([df, new_row], ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv(output_csv_file, index=False)
print(f"Extracted URLs, sentences, and hyperlinked image URLs from files in {directory_path} and saved to {output_csv_file}")
