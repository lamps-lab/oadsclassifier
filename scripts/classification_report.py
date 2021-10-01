import numpy as np
import classifier_model
import pub_url_cleaner
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
dataset = pd.read_csv('dataset.csv', header = None)
feature_data = pd.DataFrame(dataset)
feature_data=feature_data[0:]
feature_data1= feature_data
print("Feature Data Loaded: ")
print(feature_data)


#Data Cleaning and split
feature_data[0] = data_clean(feature_data[0])
train_features, test_features,train_labels,test_labels = split_data(feature_data[0], feature_data[1])

#Load transformer model
tokenizer, model = classifier_model.load_model()

#Process Train Data and Train Label to convert into numpy array
train_features = np.array(train_features.tolist())
train_labels = np.array(train_labels.tolist())
new_train_features=[]
new_train_labels=[]

new_train_features, new_train_labels = pub_url_cleaner.eliminiate_publisher_url(train_features, train_labels,publishers_list)
# Store train features and train labels in pandas series
train_features_without_publisher_url = pd.Series(np.array(new_train_features))

train_labels_without_publisher_url=pd.Series(np.array(new_train_labels))

#Use transformer model on train_data
features_train = classifier_model.tokenize_mask_data(train_features_without_publisher_url,tokenizer,model)


#Process Test Data and Label to convert into numpy array
test_features = np.array(test_features.tolist())
test_labels = np.array(test_labels.tolist())
new_test_features=[]
new_test_labels=[]



#Eliminate all publisher url from test data
new_test_features, new_test_labels = pub_url_cleaner.eliminiate_publisher_url(test_features, test_labels,publishers_list)


# Store test features and test labels in pandas series
test_features_without_publisher_url = pd.Series(np.array(new_test_features))
test_labels_without_publisher_url=pd.Series(np.array(new_test_labels))

#Use transformer model on test data
features_test = classifier_model.tokenize_mask_data(test_features_without_publisher_url,tokenizer,model)

#Train Model
lr_clf = LogisticRegression()
lr_clf.fit(features_train, train_labels_without_publisher_url)

#Predict and classification result
y_pred = lr_clf.predict(features_test)


print("Classification Report:")
print('accuracy %s' % lr_clf.score(features_test,test_labels_without_publisher_url ))
print(classification_report(test_labels_without_publisher_url, y_pred))

