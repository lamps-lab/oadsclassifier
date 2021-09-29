import transformers as ppb

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
