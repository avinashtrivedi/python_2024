#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.special import softmax
from sklearn.metrics import confusion_matrix,classification_report,f1_score,accuracy_score,ConfusionMatrixDisplay
from transformers import AutoTokenizer,AutoModelForSequenceClassification, Trainer, TrainingArguments


# In[3]:


test_data = pd.read_csv('test.csv')
test_data.head()


# In[4]:


test_data.shape


# In[5]:


Bert_finetuned_checkpoint = "Elanthamiljeeva/BERT_FPB_finetuned_v2"
Roberta_finetuned_checkpoint = "Elanthamiljeeva/RoBERTa_FPB_finetuned_v2"


# In[6]:


# Convert sentiment labels from textual to numerical format for easier processing
label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}  # Mapping labels to numerical values
test_data['Sentiment'] = test_data['Sentiment'].replace(label_dict)  # Replacing text labels with corresponding numerical values

tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer_Roberta = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')


# In[7]:


# Loading a fine-tuned BERT model specifically for sequence classification
model_bert = AutoModelForSequenceClassification.from_pretrained(Bert_finetuned_checkpoint, num_labels=3)
model_Roberta = AutoModelForSequenceClassification.from_pretrained(Roberta_finetuned_checkpoint, num_labels=3)


# In[8]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_prediction(model, tokenizer, text):
    model.to(device)
    encoded_input = tokenizer([text], padding=True, truncation=True,
                              max_length=512, return_tensors="pt").to(device)
    output = model(**encoded_input)

    # Move the output to CPU and convert to numpy array
    scores = output[0][0].detach().cpu().numpy()
    scores = softmax(scores)
    # Get ranking
    ranking = np.argsort(scores)[::-1]
    return ranking[0]


# # Prediction on finetuned BERT

# In[9]:


pred_bert = []
for text in tqdm(test_data['Text']):
    pred_bert.append(get_prediction(model_bert,tokenizer_bert,text))


# In[10]:


print(classification_report(test_data['Sentiment'],pred_bert))


# In[11]:


acc = accuracy_score(test_data['Sentiment'],pred_bert)
f1 = f1_score(test_data['Sentiment'],pred_bert,average='weighted')

acc,f1


# In[14]:


cm = confusion_matrix(test_data['Sentiment'],pred_bert)
labels = ["negative","neutral", "positive"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.show()


# # Prediction on finetuned RoBERTa

# In[15]:


pred_Roberta = []
for text in tqdm(test_data['Text']):
    pred_Roberta.append(get_prediction(model_Roberta,tokenizer_Roberta,text))


# In[16]:


print(classification_report(test_data['Sentiment'],pred_Roberta))
acc = accuracy_score(test_data['Sentiment'],pred_Roberta)
f1 = f1_score(test_data['Sentiment'],pred_Roberta,average='weighted')

acc,f1


# In[17]:


cm = confusion_matrix(test_data['Sentiment'],pred_Roberta)
labels = ["negative","neutral", "positive"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.show()


# In[ ]:




