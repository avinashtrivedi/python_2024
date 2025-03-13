#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch # Importing PyTorch for building neural network models
from transformers import AutoTokenizer,AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from torch import nn, optim
from sklearn.metrics import precision_recall_fscore_support


# In[ ]:


get_ipython().system('huggingface-cli login --token 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')


# In[ ]:


data = pd.read_csv('all-data.csv',
                   encoding='unicode_escape',
                   names=['Sentiment', 'Text'])
data.head()


# In[ ]:


data['Sentiment'].value_counts()


# In[ ]:


data.shape


# In[ ]:


Bert_checkpoint = "bert-base-uncased"
Roberta_checkpoint = "soleimanian/financial-roberta-large-sentiment"


# # Run the below code twice
# - First for Bert_checkpoint
# - next time for Roberta_checkpoint

# In[ ]:


# Convert sentiment labels from textual to numerical format for easier processing
label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}  # Mapping labels to numerical values
data['Sentiment'] = data['Sentiment'].replace(label_dict)  # Replacing text labels with corresponding numerical values

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)  # 80% for training, 20% for testing
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
test_data.to_csv('test.csv',index=False)
tokenizer = AutoTokenizer.from_pretrained(Roberta_checkpoint)
# Using the 'bert-base-uncased' pre-trained tokenizer

# Defining a preprocessing function for tokenizing and encoding sequences
def preprocess_for_bert(data):
    # Tokenizing and encoding the text data with padding and truncation to handle variable lengths
    # 'max_length=512' sets the maximum length of the sequences
    # 'return_tensors="pt"' returns PyTorch tensors
    return tokenizer(data['Text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Applying the preprocessing function to the training and validation data
train_encoded = preprocess_for_bert(train_data)
test_encoded = preprocess_for_bert(test_data)
val_encoded = preprocess_for_bert(val_data)

# Function to compute metrics for evaluation
def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels = pred.label_ids  # Actual labels
    preds = pred.predictions.argmax(-1)  # Predictions from the model
    # Calculating precision, recall, F1-score, and accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Defining a custom dataset class for handling the BERT-processed data
class SentimentDataset(torch.utils.data.Dataset):
    # Initialization with encodings and labels
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # Method to get an item by index
    def __getitem__(self, idx):
        # Preparing each item by retrieving encoded data and corresponding label
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    # Method to get the total number of items in the dataset
    def __len__(self):
        return len(self.labels)


# In[ ]:


# Creating dataset objects for the training and validation datasets
train_dataset = SentimentDataset(train_encoded, train_data['Sentiment'].values)
val_dataset = SentimentDataset(val_encoded, val_data['Sentiment'].values)
test_dataset = SentimentDataset(test_encoded, test_data['Sentiment'].values)

# Loading a pre-trained BERT model specifically for sequence classification
# 'bert-base-uncased' is the model type, and 'num_labels=3' indicates three output labels (negative, neutral, positive)
model = AutoModelForSequenceClassification.from_pretrained(Roberta_checkpoint, num_labels=3)



# Defining various training parameters
training_args = TrainingArguments(
    output_dir='RoBERTa_FPB_finetuned_v2',               # Directory where the model predictions and checkpoints will be written
    num_train_epochs=4,                   # Total number of training epochs
    per_device_train_batch_size=16,       # Batch size per device during training
    per_device_eval_batch_size=64,        # Batch size for evaluation
    warmup_steps=500,                     # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                    # Weight decay if we apply some form of weight regularization
    logging_dir='./logs',                 # Directory for storing logs
    logging_steps=10,                     # How often to print logs
    evaluation_strategy="epoch",          # Evaluation is done at the end of each epoch
    report_to="none"                      # Disables the integration with any external reporting system
)


# Initializing the Trainer with the model, training arguments, and datasets
trainer = Trainer(
    model=model,                          # The pre-trained BERT model
    args=training_args,                   # Training arguments defined above
    train_dataset=train_dataset,          # Training dataset
    eval_dataset=val_dataset,             # Validation dataset
    compute_metrics=compute_metrics       # Function for computing evaluation metrics
)

# Starting the training process
trainer.train()

# Evaluating the trained model on the validation dataset
evaluation_results = trainer.evaluate()


# In[ ]:


trainer.push_to_hub()

