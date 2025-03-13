import pandas as pd
import pickle
import torch
from torchtext.data.utils import get_tokenizer
import torchtext
from torch.utils.data import DataLoader, Dataset
        
class TextEmbeddingsTypeDataset(Dataset):
    """ Use PyTorch Dataset to generate Text Embeddings dataset
    """
    def __init__(self,data,pipeline):
        """ Constructor
        :param data: list containing tuples of length 3 - label, input features, embedding vectors
        :param pipeline: lambda function to tokenize the text using the PyTorch vocab object
        """
        self.samples = []
        self._init_dataset(data)
        self.text_pipeline = pipeline
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, text, _ = self.samples[idx]
        return label, text
    
    def _init_dataset(self,data):
        self.samples=data

    def collate_batch(self,batch):
        """ Collate function to achieve custom batching: find the offsets for text documents which are of
        varying lengths.
        
        Returns:
        label_list: tensor containing the labels in this batch
        text_embeddings_list: tensor containing the list of text embeddings
        offsets: tensor containing the offsets for text documents in the batch
        """
        label_list, text_embeddings_list, offsets  = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(_label)
        
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_embeddings_list.append(processed_text)
            offsets.append(processed_text.size(0))
            
        label_list = torch.tensor(label_list, dtype=torch.float64).unsqueeze(1)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_embeddings_list = torch.cat(text_embeddings_list)
        return label_list.to(self.device), text_embeddings_list.to(self.device), offsets.to(self.device)    
        
       
class VecTypeDataset(Dataset):
    def __init__(self,data):
        self.samples = []
        self._init_dataset(data)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, _, embedding = self.samples[idx]

        return label,embedding
    
    def _init_dataset(self,data):
        self.samples=data
        
    def collate_batch(self,batch):
        """ Collate function to achieve custom batching
        
        Returns:
        label_list: tensor containing the labels in this batch
        embed_list: tensor containing the list of vector embeddings (pre-trained) of text in this batch
        """        
        label_list,  embed_list = [], []
        for (_label, _embedding) in batch:
            e = torch.tensor(_embedding, dtype=torch.float64)
            label_list.append(_label)
            embed_list.append(e)
            
        label_list = torch.tensor(label_list, dtype=torch.float64).unsqueeze(1)
        embed_list = torch.stack(embed_list)
        return label_list.to(self.device), embed_list.to(self.device)

class MixedTypeDataset(Dataset):
    def __init__(self,data,pipeline):
        #your code goes here
        pass
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        #your code goes here
        pass
    
    def _init_dataset(self,data):
        self.samples=data
        
    def collate_batch(self,batch):
        #your code goes here
        pass


def get_boe_data():
    """ Get Bag of Embdeddings data
    Load the train and validation data which are rows of tuples of length 3 - label, input features, embedding vectors from pickle
    files. Load the PyTorch vocab object from vocab.pkl. Initialize the train and validation data loaders.
    
    Returns:
    train_dataloader: Torch utils data loader for the training dataset
    val_dataloader: Torch utils data loader for the validation dataset
    vocab: Torch vocab object which maps tokens to indices
    """
    # train & valid.pkl contain the train and validation data (rows of tuples of length 3 - label, input features, embedding vectors)
    with open('../../assets/assignment3/train.pkl', 'rb') as f:
        train_iter = pickle.load(f)
    with open('../../assets/assignment3/valid.pkl', 'rb') as f:
        val_iter = pickle.load(f)
    with open('../../assets/assignment3/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)    
    
    tokenizer = get_tokenizer('basic_english')
    text_pipeline = lambda x: vocab(tokenizer(x))

    trainset = TextEmbeddingsTypeDataset(train_iter,text_pipeline)
    valset = TextEmbeddingsTypeDataset(val_iter,text_pipeline)

    train_dataloader = DataLoader(trainset, batch_size=len(trainset),shuffle=False, collate_fn=trainset.collate_batch)
    val_dataloader = DataLoader(valset, batch_size=len(valset),shuffle=False, collate_fn=valset.collate_batch)
    
    return train_dataloader, val_dataloader, vocab
    
def get_vec_data():
    """ Get Vector (pre-trained) data
    Load the train and validation data which are rows of tuples of length 3 - label, input features, embedding vectors from pickle
    files. Initialize the train and validation data loaders.
    
    Returns:
    train_dataloader: Torch utils data loader for the training dataset
    val_dataloader: Torch utils data loader for the validation dataset
    """    
    with open('../../assets/assignment3/train.pkl', 'rb') as f:
        train_iter = pickle.load(f)
    with open('../../assets/assignment3/valid.pkl', 'rb') as f:
        val_iter = pickle.load(f)
    
    trainset = VecTypeDataset(train_iter)
    valset = VecTypeDataset(val_iter)
    
    train_dataloader = DataLoader(trainset, batch_size=len(trainset),shuffle=False, collate_fn=trainset.collate_batch)
    val_dataloader = DataLoader(valset, batch_size=len(valset),shuffle=False, collate_fn=valset.collate_batch)
    
    return train_dataloader, val_dataloader

def get_mixed_data():
    """ Get Vector (pre-trained) data
    Load the train and validation data which are rows of tuples of length 3 - label, input features, embedding vectors from pickle
    files. Initialize the train and validation data loaders.
    
    Returns:
    train_dataloader: Torch utils data loader for the training dataset
    val_dataloader: Torch utils data loader for the validation dataset
    vocab: Torch vocab object which maps tokens to indices
    """       
    with open('../../assets/assignment3/train.pkl', 'rb') as f:
        train_iter = pickle.load(f)
    with open('../../assets/assignment3/valid.pkl', 'rb') as f:
        val_iter = pickle.load(f)
    with open('../../assets/assignment3/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # add your code here
    return None, None, None

def get_raw_data():
    """ Get raw data from the Pandas dataframe
    
    Returns:
    X: features (words) found in the transcription for each patient in the dataframe
    y: label for each patient in the dataframe
    """
    df = pd.read_csv("../../assets/assignment3/mt_dia_labelled.csv")

    y = df.dropna(subset=["Diabetes","transcription"])["Diabetes"]
    X = df.dropna(subset=["Diabetes","transcription"])["transcription"]
    return X,y
