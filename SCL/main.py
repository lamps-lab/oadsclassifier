import torch
import pandas as pd
from tqdm import tqdm
from model import Transformer
from config import get_config
from loss_func import CELoss, SupConLoss, DualLoss
from data_utils import load_data
from transformers import logging, AutoTokenizer, AutoModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics import classification_report
import pickle

class Instructor:

    def __init__(self, args, logger):
        self.args = args
        
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            base_model = AutoModel.from_pretrained('roberta-base')
        elif args.model_name == 'DistilBERT':
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            base_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        elif args.model_name == 'GPT-2':
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            base_model = GPT2Model.from_pretrained("gpt2")
        else:
            raise ValueError('unknown model')
        self.model = Transformer(base_model, args.num_classes, args.method)
        best_model_state_dict = None  # Variable to store the best model's state_dict
        best_test_prediction = None  # Variable to store the best test prediction
        best_model_file_path = 'best_model_bert.pkl'
        with open(best_model_file_path, 'rb') as model_file:
            best_model_state_dict = pickle.load(model_file)
        self.model.load_state_dict(best_model_state_dict)


        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    
    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        prediction = []
        label = []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
                n_test += targets.size(0)
            
                prediction += torch.argmax(outputs['predicts'], dim=1).tolist()
              
                label += targets.tolist()
                print("Target:", n_test)
  
        return test_loss / n_test, n_correct / n_test, label, prediction

    def run(self):
        
        best_model_state_dict = None  # Variable to store the best model's state_dict
        test_dataloader = load_data(dataset=self.args.dataset,
                                          data_dir=self.args.data_dir,  # Specify the path to your other test data
                                          tokenizer=self.tokenizer,
                                          test_batch_size=self.args.test_batch_size,
                                          model_name=self.args.model_name,
                                          method=self.args.method,
                                          workers=0)
       
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.method == 'ce':
            criterion = CELoss()
        elif self.args.method == 'scl':
            criterion = SupConLoss(self.args.alpha, self.args.temp)
        elif self.args.method == 'dualcl':
            criterion = DualLoss(self.args.alpha, self.args.temp)
        else:
            raise ValueError('unknown method')
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay)
        test_loss, test_acc, label, prediction = self._test(test_dataloader, criterion)
        self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc*100))
        print(prediction)
        self.logger.info('Classification report:'.format(print(classification_report(label, prediction))))
        
        


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    ins = Instructor(args, logger)
    ins.run()
