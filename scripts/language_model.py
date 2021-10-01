import transformers as ppb
import numpy as np
import torch

#-------------------Function for loading transformer model----------------
def load_model():
#------------------ To load SPECTER model uncomment following three lines-------------------
#   from transformers import AutoTokenizer, AutoModel
#   tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
#   model = AutoModel.from_pretrained('allenai/specter')

#   ------------------ To load Distibert model uncomment following line-------------------
  model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
  #------------------ To load Roberta model uncomment following line-------------------
  # model_class, tokenizer_class, pretrained_weights = (ppb.RobertaModel, ppb.RobertaTokenizer, 'roberta-base')
  #------------------ To load BERT model uncomment following line-------------------
  # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
  tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
  model = model_class.from_pretrained(pretrained_weights)
  return tokenizer, model

  #-------------------Function for tokenization and masking train dataset------------------------------
def tokenize_mask_data(features,tokenizer,model):
  tokenized_train = features.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
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
  encoded_output = last_hidden_states[0][:,0,:].numpy()
  return encoded_output