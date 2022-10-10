import PyPDF2
import numpy as np
import re
import spacy
import textract
import pandas as pd
from nltk.tokenize import sent_tokenize 
 

sentence_with_url_list = []

# regular expression to detect URLs in a sentence.
regex = r'(http|https|ftp|ftps)\:\/\/[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(\/\S*)?'

# PyPDF2 to extract text from pdf
pdfFileObject = open(r"reproducibility.pdf", 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObject)

# get number of pages of a pdf file
pages_number = pdfReader.numPages

# iterate through all the pdf pages to extract sentences having URLs
for page in range (0, pdfReader.numPages):
  pageObject = pdfReader.getPage(page)
  text = pageObject.extractText()
  
  # tokenize sentences
  sentences = sent_tokenize(text)

  # remove '/n' from sentences
  sentences_after_remove_n = [x.replace('\n',' ') for x in  sentences]

  # create a list of sentences having URLs
  for sentence in range(0,len(sentences_after_remove_n)):
    url = re.findall(regex,sentences_after_remove_n[sentence])   
    if url:
      sentence_with_url_list.append(sentences_after_remove_n[sentence])
  
 
pdfFileObject.close()
print("List of sentences with URLs: ", sentence_with_url_list)