import numpy as np
import language_model
import pub_url_cleaner
import re
import spacy
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pickle

def url_extract(filename):
  fileObj = open(filename, 'r')
  input_text = fileObj.read()
  sentence_tokenizer = spacy.load("en_core_web_sm")


  text = sentence_tokenizer(input_text)
  text = list(text.sents)

  sentence_list = []
  regex = r'(https?://[^\s]+)'
  for i in range(0,len(text)):
    url = re.findall(regex,text[i])   
    if url:
      sentence_list.append(text[i])
  return sentence_list

#------------------Function for data preprocessing---------------
def data_clean(*feature_data):
  feature_data = feature_data[0].apply(lambda x: x.strip())
  feature_data = feature_data.apply(lambda x: x.lower())
  return feature_data




#An array of publisher name
publishers_list = ["springer","umi","ebscohost","sciencedirect","emeraldinsight","sagepub","scopus"
            "nlai","intechopen","dart-europe","digitool","highwire"
            "doaj",
            "pnas",
            "eprints.nottingham",
            "digital.library.upenn.edu/books/",
            "etd.ohiolink",
            "escholarship",
            "lib",
            "ieeexplore",
              "acm",
              "wiley",
              "sciencedirect",
              "acs",
              "aiaa",
              "aip",
              "ajpe",
              "aps",
              "ascelibrary",
              "asm",
              "asme",
              "bioone",
              "birpublications",
              "bmj",
              "emeraldinsight",
              "geoscienceworld",
              "icevirtuallibrary",
              "informs",
              "ingentaconnect",
              "iop",
              "jamanetwork",
              "joponline",
              "jstor",
              "mitpressjournals",
              "nrcresearchpress",
              "oxfordjournals",
              "royalsociety",
              "rsc",
              "rubberchemtechnol",
              "sagepub",
              "scientific",
              "spiedigitallibrary",
              "tandfonline",
              "theiet"]


sentence_list = url_extract(filename)
dataset = pd.array(sentence_list)


#Process Test Data and Label to convert into numpy array
test_features = np.array(dataset)
new_test_features=[]



#Eliminate all publisher url from test data
new_test_features = pub_url_cleaner.eliminiate_publisher_url(test_features, publishers_list)


# Store test features and test labels in pandas series
test_features_without_publisher_url = pd.Series(np.array(new_test_features))


#Use transformer model on test data
#Load transformer model
tokenizer, model = language_model.load_model()
features_test = language_model.tokenize_mask_data(test_features_without_publisher_url,tokenizer,model)

# load the model from disk
loaded_model = pickle.load(open("model_weight.sav", 'rb'))

#Predict and classification result
y_pred = loaded_model.predict(features_test)
prediction_label = []

for idx in range(0,len(y_pred)):
  if(y_pred[idx]==0):
    prediction_label.append("Not-OADS")
  else:
    prediction_label.append("OADS")

df1 = pd.DataFrame(sentence_list,columns = ["Sentence"])
df2 = pd.DataFrame(prediction_label,columns = ["class"])
dataframe = pd.concat([df1, df2],axis=1)
dataframe.to_csv('Output.csv',index=False)
print(dataframe)




