import nltk
import re
import pandas as pd

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
fileObj = open('filename', 'r')
text = fileObj.read()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


text = sent_tokenize(text)

sentence_list = []
regex = r'(https?://[^\s]+)'
for i in range(len(text)):
  url = re.findall(regex,text[i])   
  if url:
    sentence_list.append(text[i])


df = pd.DataFrame(sentence_list)
df.to_csv('test.csv',index=False)
