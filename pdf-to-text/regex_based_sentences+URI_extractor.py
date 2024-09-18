import os
import re
import time
import pandas as pd
from wtpsplit import SaT

# Directory containing the text files
text_files_dir = "/data/PMC-Data/TEXT"
output_csv_file = "/data/PMC-Data/regex-urls.csv"

# Initialize an empty DataFrame
df = pd.DataFrame(columns=["File", "Sentence"])

def extract_sentences_with_urls(text_content):
    sentences_with_urls = []
    url_pattern = r'''(?xi)
    (?:(?:https?|ftp|file|data|javascript|mailto|tel|git|ssh|magnet):\/\/  # Protocols
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
    (?:%20|\s)?  # allow for spaces encoded as %20 or as single spaces
    (?:[^\s]*)  # continue matching non-whitespace characters
    '''
    sat_6l_sm = SaT("sat-6l-sm")

    sentences = sat_6l_sm.split(text_content)
    for sentence in sentences:
        if url_pattern.search(sentence):
            sentences_with_urls.append(sentence)

    return sentences_with_urls

# Read each text file in the directory and extract sentences with URLs
for filename in os.listdir(text_files_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(text_files_dir, filename)
        with open(file_path, 'r') as file:
            text_content = file.read()
            sentences_with_urls = extract_sentences_with_urls(text_content)
            for sentence in sentences_with_urls:
                print(f"File: {filename}, Sentence: {sentence}")
                new_row = pd.DataFrame({"File": [filename], "Sentence": [sentence]})
                df = pd.concat([df, new_row], ignore_index=True)
                # Save the DataFrame to a CSV file immediately
                df.to_csv(output_csv_file, index=False)
            

print(f"Extracted sentences with URLs from {len(df)} files and saved to {output_csv_file}")
