#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


pip install gradio


# In[3]:


# PyTorch
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# Etc
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import glob
import json


# Use GPU if available
if (torch.cuda.is_available()):
    device = torch.device('cuda')
    print("Running on GPU")
else: 
    device = torch.device('cpu')
    print("Running on CPU")


# In[4]:


# Get absolute paths of files
dialogues_regex_folder_path = "data/dialogues/*.txt"

# Get the absolute paths for each file 
list_of_files = glob.glob(dialogues_regex_folder_path)
print(list_of_files[:3]) # Visualize the first 3
print(len(list_of_files)) # 47


# In[5]:


# Parsing
list_of_dicts = [] # Init

# Loop for each file
for filename in list_of_files:
  with open(filename) as f:
      for line in f: # Loop for each line (inside each file)
          list_of_dicts.append(json.loads(line)) # insert in a dictionary


# In[7]:


# Create a new dict containing only useful data
new_list_of_dicts = [] 

for old_dict in list_of_dicts:
  foodict = {k: v for k, v in old_dict.items() if (k == 'turns')} 
  new_list_of_dicts.append(foodict)

print(len(new_list_of_dicts))

# Just to be sure we don't make bad use of the old variable,
# we will make the old dict equal to the new one.
# In the end, they are all the same.
list_of_dicts = []
list_of_dicts = new_list_of_dicts 

print(list_of_dicts[:2])


# In[8]:


# Init matrices
questions = []
answers = []

matrix_greetings = ["Hey", "Hi"]

matrix_byes = ["Ok", "Okie", "Bye"]

# For each dictionary in the list
for dictionary in list_of_dicts:
  matrix_QA = dictionary['turns']
  
  # Append a first random greeting, as explained above
  questions.append(random.choice(matrix_greetings))

  bot_flag = True # Init

  # For each Q/A in the matrix
  for sentence in matrix_QA:

    if bot_flag == True:
      answers.append(sentence) # Used for bot's answers
      bot_flag = False # Switch
      continue
    else:
      questions.append(sentence) # Used for user's questions
      bot_flag = True # Switch
      continue
  if bot_flag == True: 
    answers.append(random.choice(matrix_byes))


# In[9]:


assert len(questions) == len(answers), "ERROR: The length of the questions and answer matrices are different."
# If it does not return any warning/error, then everything is good.

print(len(questions)) # We have 238051 QAs (if we load all 47 texts)


# In[10]:


"""
    Write to tsv file so we just load this each time
"""
import csv

filepath_to_save = '/tmp/output.tsv' # Change accordingly
with open(filepath_to_save, 'wt') as out_file:
    # Instantiate object
    tsv_writer = csv.writer(out_file, delimiter='\t')

    # Loop QAs & write to file
    for i in range(len(questions)):
        tsv_writer.writerow([questions[i], answers[i]])


# In[11]:


#### HELPERS

### Helper class for word indexing
SOS_TOKEN = 0 # Start of sentence
EOS_TOKEN = 1 # End of sentence

# Let's define a QA (Questions/Answers) class
# since each class has its own 'language'.

class QA_Lang:
    """ 
    # The constructor should be specified by its:
    # - word2index, a dictionary that maps each word to each index
    # - index2word, a dictionary that maps each index to each word
    # - n_words, the number of words in the dictionary
    """
    def __init__(self):
        self.word2index = {}
        self.index2word = {0: 'SOS', 1: 'EOS'} # Reserved for start and end token
        self.n_words = 2 # Initialize with start and end token

    # Use each sentence and instantiate the class properties
    def add_sentence(self, sentence):
        for word in sentence.split(' '): # For each word in the sentence
            if word not in self.word2index: # If word is not seen
                # Add new word
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1
            


# ## Text Preprocessing
# Let's remove non-alphabet/punctuation characters and make them all ASCII encoded.

# In[12]:


# Preprocessing helper function
def preprocess_text(sentence):
    """
    Preprocesses text to lowercase ASCII alphabet-only characters
    without punctuation
    """

    # Conver sentence to lowercase, after removing whitespaces
    sentence = sentence.lower().strip()

    # Convert Unicode string to plain ASCII characters
    normalized_sentence = [c for c in unicodedata.normalize('NFD', sentence) if
                           unicodedata.category(c) != 'Mn']

    # Append the normalized sentence
    sentence = ''
    sentence = ''.join(normalized_sentence)
    
    # Remove punctuation and non-alphabet characters
    sentence = re.sub(r"([.!?])", r" \1", sentence)
    sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)

    return sentence


# In[13]:


# Visualize the path once again
print(os.getcwd())


# In[14]:


# Reading helper function
def readQA():
    """
    Reads the tab-separated data from the storage and cleans it
    """

    print('Reading lines from file...')

    # Read text from file and split into lines
    # Remember that .tsv file separates pairs with the tab character and
    # each pair is separated with a newline character

    data_path = os.getcwd() + "/data/dataset.tsv" # Change to your own
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')

    # Split lines into pairs, normalize
    TAB_CHARACTER = '\t'

    pairs = [[preprocess_text(sentence) \
              for sentence in line.split(TAB_CHARACTER)] \
              for line in lines]
    
    ''' 
    # Find maximum length of pairs
    count1 = count2 = 0
    max_words = 0
    for i in range(len(pairs)):
        count1 = len(pairs[i][0].split())
        count2 = len(pairs[i][1].split())
        result = count1 + count2
        if result > max_words:
            max_words = result

    print(max_words) # 304
    '''
    
    questions = QA_Lang()
    answers = QA_Lang()

    return questions, answers, pairs


# In[15]:


MAX_LENGTH = 35 # Arbitrary, try different values!

# Filtering helper function
def filter(pairs):
    """
    Filters sentences based on the max length defined above.
    """
    new_pairs = []

    for pair in pairs:
        question_length = len(pair[0].split(' '))
        answer_length = len(pair[1].split(' '))

        if question_length < MAX_LENGTH and answer_length < MAX_LENGTH:
            new_pairs.append(pair)

    return new_pairs


# ## Preparing the dataset
# Let's combine all the above little methods in one.

# In[16]:


def prepare_data():
    """
    Prepares the data, combining all of the above methods and returns:
    questions, answers objects and the pairs of sentences
    """
    # Read sentence pairs
    questions, answers, pairs = readQA()
    print("Read " + str(len(pairs)) + " sentence pairs")

    # Filter pairs
    pairs = filter(pairs)
    print("Filtered down to " + str(len(pairs)) + " sentence pairs")

    # Count words and instantiate the 'language' objects 
    for pair in pairs:
        questions.add_sentence(pair[0])
        answers.add_sentence(pair[1])

    print("The questions object is defined by " +
                        str(questions.n_words) + " words")
    
    print("The answers object is defined by " +
                        str(answers.n_words) + " words")

    return questions, answers, pairs


# Finally, let's call the method.

# In[17]:


# Load and prepare the dataset, printing some characteristics
questions, answers, pairs = prepare_data()


# In[18]:


len(pairs)


# In[ ]:





# In[19]:


# Visualize 3 random pairs of Q&A
for _ in range(3):
    print(random.choice(pairs))


# In[20]:


##### SEQ2SEQ MODEL

class EncoderRNN(nn.Module):
    """
    The encoder is a GRU in our case.
    It takes the questions matrix as input. For each word in the 
    sentence, it produces a vector and a hidden state; The last one
    will be passed to the decoder in order to initialize it.
    """
    # Initialize encoder
    def __init__(self, input_size, hidden_size): 
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # Embedding layers convert the padded sentences into appropriate vectors
        # The input size is equal to the questions vocabulary
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # We use a GRU because it's simpler and more efficient (training-wise)
        # than an LSTM
        self.gru = nn.GRU(hidden_size, hidden_size)

    # Forward passes
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded

        # Pass the hidden state and the encoder output to the next word input
        output, hidden = self.gru(output, hidden) 

        return output, hidden

    # PyTorch Forward Passes
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

##### ATTENTION-BASED DECODER
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        # Initialize the constructor
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # Combine Fully Connected Layer
        self.attention = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attention_combine = nn.Linear(self.hidden_size * 2,
                                           self.hidden_size)
        # Use dropout
        self.dropout = nn.Dropout(self.dropout_p)

        # Follow with a GRU and a FC layer
        # We use a GRU because it's simpler and more efficient (training-wise)
        # than an LSTM
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # Forward passes as from the repo
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attention_weights = F.softmax(self.attention(torch.cat((embedded[0],
                                                                hidden[0]), 1)),
                                                                 dim=1)
        
        attention_applied = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attention_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)

        # Follow with a ReLU activation function after dropout
        output = F.relu(output)

        # Then, use the GRU
        output, hidden = self.gru(output, hidden)

        # And use softmax as the activation function
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attention_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[21]:


##### NETWORK PREPROCESSING HELPERS

def tensor_from_sentence(lang, sentence):
    """
    Given an input sentence and a 'language' object, 
    it creates an appropriate tensor with the EOS_TOKEN in the end.
    """

    # For each sentence, get a list of the word indices
    indices = [lang.word2index[word] for word in sentence.split(' ')]
    indices.append(EOS_TOKEN) # That will help the decoder know when to stop

    # Convert to a PyTorch tensor
    sentence_tensor = torch.tensor(indices, dtype=torch.long, device=device).view(-1, 1)

    return sentence_tensor

def tensors_from_pair(pair):
    """
    Given our 2D dataset as a list, it calls the 'tensor_from_sentence' method
    and returns the appropriate input/target tensors
    """
    
    input_tensor = tensor_from_sentence(questions, pair[0])
    target_tensor = tensor_from_sentence(answers, pair[1])

    return (input_tensor, target_tensor)


# Some display helpers will be used in the training.

# In[22]:


##### DISPLAY HELPERS
"""
Helper functions for printing time elapsed and estimated remaining time for
training.
"""
import time
import math

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s

    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


# In[23]:


# Specify path name
encoder_name = 'encoder_serialized2.pt'
decoder_name = 'decoder_serialized2.pt'

## Load previously trained models
encoder = torch.load(encoder_name)
attention_decoder = torch.load(decoder_name)


# In[24]:


# Inference helper method
def inference(encoder, decoder, sentence, max_length=MAX_LENGTH):
    """
    Returns the decoded string after doing a forward pass in the seq2seq model.
    """
      
    with torch.no_grad(): # Stop autograd from tracking history on Tensors

        sentence = preprocess_text(sentence) # Preprocess sentence

        input_tensor = tensor_from_sentence(questions, sentence) # One-hot tensor
        input_length = input_tensor.size()[0]

        # Init encoder hidden state
        encoder_hidden = encoder.init_hidden()

        # Init encoder outputs
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # Forward pass in the encoder
        for encoder_input in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[encoder_input],
                                                     encoder_hidden)
            encoder_outputs[encoder_input] += encoder_output[0, 0]

        # Start of sentence token
        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)

        # Decoder's initial hidden state is encoder's last hidden state
        decoder_hidden = encoder_hidden

        # Init the results array
        decoded_words = []

        # Forward pass in the decoder
        for d_i in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            
            _, top_i = decoder_output.data.topk(1) 

            if top_i.item() == EOS_TOKEN: # If EOS is predicted
                break # Break and return the sentence to the user
            else:
                # Append prediction by using index2word
                decoded_words.append(answers.index2word[top_i.item()])

            # Use prediction as input
            decoder_input = top_i.squeeze().detach()

        return ' '.join(decoded_words) # Return the predicted sentence string 


# # Domains include bus schedules, apartment search, alarm setting, banking, and event reservation. Each dialog was grounded in a scenario with roles, pairing a person acting as the bot and a person acting as the user

# In[25]:


import random
def random_response(message, history):
    user_input = str(message)
    return str(inference(encoder, attention_decoder, user_input))


# In[26]:


import gradio as gr
gr.ChatInterface(random_response).launch()


# In[ ]:




