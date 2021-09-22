import numpy as np
import re
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

#------------------Function for data preprocessing---------------
def data_clean(*feature_data):
  feature_data = feature_data[0].apply(lambda x: x.strip())
  feature_data = feature_data.apply(lambda x: x.lower())
  return feature_data

#----------------Function for splitting dataset into train and test dataset----------------
def split_data(data, label):
  train_features, test_features,train_labels,test_labels = train_test_split(data,label,test_size=0.2, random_state = 0)
  return train_features, test_features, train_labels, test_labels

#-------------------Function for loading transformer model----------------
def load_model():
  #------------------ To load SPECTER model uncomment following three lines-------------------
  # from transformers import AutoTokenizer, AutoModel
  # tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
  # model = AutoModel.from_pretrained('allenai/specter')

  #------------------ To load Distibert model uncomment following line-------------------
  model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
  #------------------ To load Roberta model uncomment following line-------------------
  # model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.RobertaTokenizer, 'roberta-base')
  #------------------ To load BERT model uncomment following line-------------------
  # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
  tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
  model = model_class.from_pretrained(pretrained_weights)
  return tokenizer, model


#-------------------Function for tokenization and masking train dataset------------------------------
def tokenize_mask_train(train_features,tokenizer,model):
  tokenized_train = train_features.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
  max_len = 0
  for i in tokenized_train.values:
      if len(i) > max_len:
          max_len = len(i)

  padded_train = np.array([i + [0]*(max_len-len(i)) for i in tokenized_train.values])
  attention_mask_train = np.where(padded_train != 0, 1, 0)
  input_ids = torch.tensor(padded_train)  
  attention_mask_train = torch.tensor(attention_mask_train)
  with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask_train)
  features_train = last_hidden_states[0][:,0,:].numpy()
  return features_train
#--------------------------Function for eliminating publisher url from train dataset----------------------------
def eliminiate_publisher_url_train(train_features, train_labels):
  for i in range(0,400):

    if train_features[i] == "\n":
      print("Empty")
    else:
      # CHECK publisher URL
            urls = re.findall(r'(https?://\S+)',train_features[i])
            
            temp = 0
            for j in range(0,len(urls)):
              for k in range(0,len(publishers_list)):
                if (urls[j].find(publishers_list[k]) != -1):
                  temp+=1
          
            if temp==len(urls) and len(urls)!=0:
              print("Publisher URL Found in training data")
              #  test_features.pop(i)
              #  test_labels.pop(i)
              #  test_features=np.delete(test_features,i)
              #  test_labels=np.delete(test_labels,i)
            else:
              new_train_features.append(str(train_features[i]))
              new_train_labels.append(train_labels[i])
  return new_train_features, new_train_labels


#--------------------------Function for eliminating publisher url from test dataset----------------------------
def eliminiate_publisher_url(test_features, test_labels):
  for i in range(0,100):

    if test_features[i] == "\n":
      print("Empty")
    else:
      # CHECK publisher URL
            urls = re.findall(r'(https?://\S+)',test_features[i])
            
            temp = 0
            for j in range(0,len(urls)):
              for k in range(0,len(publishers_list)):
                if (urls[j].find(publishers_list[k]) != -1):
                  temp+=1
          
            if temp==len(urls) and len(urls)!=0:
              print("Publisher URL Found in test data")
              #  test_features.pop(i)
              #  test_labels.pop(i)
              #  test_features=np.delete(test_features,i)
              #  test_labels=np.delete(test_labels,i)
            else:
              new_test_features.append(str(test_features[i]))
              new_test_labels.append(test_labels[i])
  return new_test_features, new_test_labels


#----------------------Function for tokenization and masking test dataset-------------------------------
def tokenize_mask_test(test_features,tokenizer,model):
  tokenized_test = test_features.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
  max_len = 0
  for i in tokenized_test.values:
      if len(i) > max_len:
          max_len = len(i)

  padded_test = np.array([i + [0]*(max_len-len(i)) for i in tokenized_test.values])
  attention_mask_test = np.where(padded_test != 0, 1, 0)
  attention_mask_test.shape 
  input_ids_test = torch.tensor(padded_test)  
  attention_mask_test = torch.tensor(attention_mask_test)

  with torch.no_grad():
      last_hidden_states_test = model(input_ids_test, attention_mask=attention_mask_test)
  features_test = last_hidden_states_test[0][:,0,:].numpy()
  return features_test


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

#Read dataset from CSV file              
dataset = pd.read_csv('Merged-dataset.csv', header = None)
feature_data = pd.DataFrame(dataset)
feature_data=feature_data[0:]
feature_data1= feature_data
print("Feature Data Loaded: ")
print(feature_data)


#Data Cleaning and split
feature_data[0] = data_clean(feature_data[0])
train_features, test_features,train_labels,test_labels = split_data(feature_data[0], feature_data[1])

#Load transformer model
tokenizer, model = load_model()

new_array1 = train_features.tolist()
new_label1=train_labels.tolist()
train_features = np.array(new_array1)
train_labels = np.array(new_label1)
new_train_features=[]
new_train_labels=[]

new_train_features, new_train_labels = eliminiate_publisher_url_train(train_features, train_labels)
# Store train features and train labels in pandas series
new_train_features1 = np.array(new_train_features)
new_train_features2 = pd.Series(new_train_features1)

new_train_labels1=np.array(new_train_labels)
new_train_labels2=pd.Series(new_train_labels1)

#Use transformer model on train_data
features_train = tokenize_mask_train(new_train_features2,tokenizer,model)
print("Encoded Train Data: ")
print(features_train)

#Process Test Data and Label to convert into numpy array
new_array = test_features.tolist()
new_label=test_labels.tolist()
test_features = np.array(new_array)
test_labels = np.array(new_label)
new_test_features=[]
new_test_labels=[]

print(test_features.size)

#Eliminate all publisher url from test data
new_test_features, new_test_labels = eliminiate_publisher_url(test_features, test_labels)
print(len(new_test_features))

# Store test features and test labels in pandas series
new_test_features1 = np.array(new_test_features)
new_test_features2 = pd.Series(new_test_features1)

new_test_labels1=np.array(new_test_labels)
new_test_labels2=pd.Series(new_test_labels1)

#Use transformer model on test data
features_test = tokenize_mask_test(new_test_features2,tokenizer,model)

#Train Model
lr_clf = LogisticRegression()
lr_clf.fit(features_train, new_train_labels2)

#Predict and classification result
y_pred = lr_clf.predict(features_test)


print("Classification Report:")
print('accuracy %s' % lr_clf.score(features_test,new_test_labels2 ))
print(classification_report(new_test_labels2, y_pred))

