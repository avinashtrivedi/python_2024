import time
import random
import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable 
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
import numpy as np
from check_models import ensure_boe_model_is_correct, ensure_vec_model_is_correct

torch.set_num_threads(2)
torch.set_num_interop_threads(2)

def set_seed():
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

class BoEModel(nn.Module):
    """ Bag of Embeddings represenataion of text data model.
    """
    def __init__(self, vocab_size, embed_dim, num_class):
        """ Constructor
        :param vocab_size: size of the dictionary of embeddings
        :param embed_dim: size of each embedding vector
        :param num_class: size of each output sample
        """
        super(BoEModel, self).__init__()
        #your code goes here; check the init_weights function below to see the names of the layers in your model
        self.init_weights()
        self.double()

    def init_weights(self):
        initrange = 0.5
        self.embedding_bag.weight.data.uniform_(-initrange, initrange)
        self.fc_layer.weight.data.uniform_(-initrange, initrange)
        self.fc_layer.bias.data.zero_()

    def forward(self, boe, offsets):
        """ forward function defines the computation performed at every call
        
        Parameters:
        :param boe: tensor containing a batch of text embeddings
        :param offsets: offsets: tensor containing offsets for each document in the batch
        
        Returns:
        tensor containing the output of the last layer
        """
        #your code goes here
        pass

def get_boe_model_parameters(\
        train_dataloader,\
        val_dataloader,\
        vocab
        ):
    """ Get the average precision score of the validation dataset of the BoE model
    
    Parameters:
    :param train_dataloader: Torch utils data loader for the training dataset
    :param val_dataloader: Torch utils data loader for the validation dataset
    :param vocab: Torch vocab object which maps tokens to indices
    
    Returns:
    A dictionary which maps the hyperparameter embedding dimension to the average
    precision score of the validation dataset.
    
    """
    NUM_CLASS = 1
    EPOCHS = 200 # epoch
    LR = 1 # learning rate

    # Remove the following line after you have added code to BoEModel
    return {}

    # the next line checks your model against the model used by the autograder
    ensure_boe_model_is_correct(BoEModel(len(vocab), 5, NUM_CLASS))    
    
    vocab_size = len(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed()
    to_return = {}
    
    for d in [10,25,50,100]:

        model = BoEModel(vocab_size, d,  NUM_CLASS).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor (6.0))
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            
            for idx, (label, text_embeddings, offsets) in enumerate(train_dataloader):
                
                optimizer.zero_grad()
                predicted_label = model(text_embeddings, offsets)
                loss = criterion(predicted_label, label)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
        
        (my_label, text, offsets) = list(val_dataloader)[0] # only 1 tuple in val_dataloader (batch_size=100)
        predicted_label = model(text, offsets)

        output_prob = torch.sigmoid(predicted_label).reshape(-1).detach().cpu().numpy()
        
        gt = (my_label).reshape(-1).detach().cpu().numpy()
        avg_prec_score = average_precision_score(gt, output_prob)
        
        to_return['{}'.format(d)]=avg_prec_score
        #print("embed dim {} average precision score {}".format(d,avg_prec_score))
    return to_return


class VecModel(nn.Module):
    """ Pre-trained vector embeddings of the text data model.
    """
    def __init__(self, pre_embed_size, layer, num_class):
        """ Constructor
        :param pre_embed_size: size of the pre-trained vector embedding
        :param layer: size of each output sample from the first fully connected layer
        :param num_class: size of each output sample
        """        
        super(VecModel, self).__init__()
        #your code goes here


    def forward(self, pre_embeddings):
        """ forward function defines the computation performed at every call
        
        Parameters:
        :param pre_embeddings: tensor containing a batch of pre-trained vector embeddings
        
        Returns:
        tensor containing the output of the last layer
        """        
        #your code goes here
        pass

def get_vec_model_parameters(\
        train_dataloader,\
        val_dataloader
        ):
    """ Get the average precision score of the validation dataset of the Vec model
    
    Parameters:
    :param train_dataloader: Torch utils data loader for the training dataset
    :param val_dataloader: Torch utils data loader for the validation dataset
    
    Returns:
    A dictionary which maps the hyperparameter embedding dimension to the average
    precision score of the validation dataset.
    
    """    
    NUM_CLASS = 1
    VEC_SIZE = 200
    EPOCHS = 200 # epoch
    LR = 1 # learning rate
    
    # Remove the following line after you have added code to VecModel
    return {}

    # the next line checks your model against the model used by the autograder
    ensure_vec_model_is_correct(VecModel(VEC_SIZE, 5, NUM_CLASS))    
    
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    to_return = {}
    for d in [10,25,50,100]:
 
            model = VecModel(VEC_SIZE, d, NUM_CLASS).to(device)
            
            criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor (6.0))
            optimizer = torch.optim.SGD(model.parameters(), lr=LR)
            
            for epoch in range(1, EPOCHS + 1):
                model.train()
                for idx, (label, embeddings) in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    
                    predicted_label = model(embeddings)
                    loss = criterion(predicted_label, label)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

            (my_label, embeddings) = list(val_dataloader)[0]
            predicted_label = model(embeddings)

            output_prob = torch.sigmoid(predicted_label).reshape(-1).detach().cpu().numpy()
            
            gt = (my_label).reshape(-1).detach().cpu().numpy()
            avg_prec_score = average_precision_score(gt, output_prob)
            
            to_return['{}'.format(d)] = avg_prec_score
            # print("d {} average precision score {}".format(d,avg_prec_score))
    return to_return


class MixedModel(nn.Module):
    """ Combination of Bag of Embeddings represenataion of text data and 
    pre-trained Vector embeddings of the text data model.
    """    
    def __init__(self, vocab_size, embed_dim, pre_embed_size, layer, num_class):
        """ Constructor
        :param vocab_size: size of the dictionary of embeddings
        :param embed_dim: size of each embedding vector
        :param pre_embed_size: size of the pre-trained vector embedding
        :param layer: size of each output sample from the fully connected layer in the vec model
        :param num_class: size of each output sample
        """        
        super(MixedModel, self).__init__()
        #your code goes here
        self.init_weights()
        self.double()

    def init_weights(self):
        initrange = 0.5
        self.embedding_bag.weight.data.uniform_(-initrange, initrange)
        self.fc_layer.weight.data.uniform_(-initrange, initrange)
        self.fc_layer.bias.data.zero_()

    def forward(self, text_embeddings, offsets, pre_embeddings):
        """ forward function defines the computation performed at every call
        
        Parameters:
        :param boe: tensor containing a batch of text embeddings
        :param offsets: offsets: tensor containing offsets for each document in the batch
        :param pre_embeddings: tensor containing a batch of pre-trained vector embeddings        
        
        Returns:
        tensor containing the output of the last layer
        """        
        #your code goes here
        pass

def get_mixed_model_parameters(\
        train_dataloader,\
        val_dataloader,\
        vocab\
        ):
    """ Get the average precision score of the validation dataset of the Mixed model
    
    Parameters:
    :param train_dataloader: Torch utils data loader for the training dataset
    :param val_dataloader: Torch utils data loader for the validation dataset
    :param vocab: Torch vocab object which maps tokens to indices    
    
    Returns:
    A dictionary which maps the hyperparameter embedding dimension to the average
    precision score of the validation dataset.
    
    """       
    NUM_CLASS = 1
    EPOCHS = 200 # epoch
    LR = 1 # learning rate

    VEC_SIZE = 200
    
    # Remove the following line after you have added code to MixedModel and get_mixed_data
    return {}, None

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = len(vocab)
    to_return = {}
    best_model = None
    best_avg_prec_score = -1.

    # these are sample hyper parameters; you can choose a different set of hyperparameters
    # for example, 10,25,50,100 like above
    for d in [5,10]:
        for c in [5,10]:
            model = MixedModel(vocab_size, d, VEC_SIZE, c, NUM_CLASS).to(device)
            
            criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor (6.0))
            optimizer = torch.optim.SGD(model.parameters(), lr=LR)
            
            for epoch in range(1, EPOCHS + 1):
                model.train()
                
                for idx, (label, text_embeddings, pre_embeddings,offsets) in enumerate(train_dataloader):
                    
                    optimizer.zero_grad()
                    predicted_label = model(text_embeddings, offsets, pre_embeddings)
                    loss = criterion(predicted_label, label)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    
            (my_label, text_embeddings, pre_embeddings, offsets) = list(val_dataloader)[0]
            predicted_label = model(text_embeddings, offsets, pre_embeddings)
            
            output_prob = torch.sigmoid(predicted_label).reshape(-1).detach().cpu().numpy()
            
            gt = (my_label).reshape(-1).detach().cpu().numpy()
            avg_prec_score = average_precision_score(gt, output_prob)
            if avg_prec_score > best_avg_prec_score:
                best_avg_prec_score = avg_prec_score
                best_model = model

            to_return['{}-{}'.format(d,c)]=avg_prec_score
            # print("d {} c {} average precision score{}".format(d,c,avg_prec_score))
    return to_return, best_model
