#!/usr/bin/env python
# coding: utf-8

# # Domains include bus schedules, apartment search, alarm setting, banking, and event reservation

# In[1]:


import os
print(os.getcwd())


# In[2]:


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


# In[3]:


get_ipython().system('ls')


# In[4]:


# Get absolute paths of files
# https://www.microsoft.com/en-us/research/project/metalwoz/
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


# In[6]:


# Visualize the dictionaries
print(list_of_dicts[0])
print(list_of_dicts[1].keys)
print(list_of_dicts[332])
print(list_of_dicts[:3])


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


# ## Data Augmentation & Preparation

# In[8]:


# Init matrices
questions = []
answers = []

# We assume that the first answer by the bot (aka "Hello, how may I help you?") 
# is returned after a user greeting.
# This is used in order to ensure that the dataset will be even 
# and each question is paired with an answer.
# That's why we create a mini random catalog 
# of artificial 'ghost' user greetings.
matrix_greetings = ["Hey", "Hi"]

# A similar situation happens in the corner case 
# when the last sentence is from the user.
# As said, each sentence from the user should be paired
# with a sentence from the bot.
# That's why we will in this case add an artificial one.
matrix_byes = ["Ok", "Okie", "Bye"]

# For each dictionary in the list
for dictionary in list_of_dicts:
  matrix_QA = dictionary['turns']
  
  # Append a first random greeting, as explained above
  questions.append(random.choice(matrix_greetings))
    
  # In order to split the QAs to 2 matrices (questions & answers),
  # we will use a flag to indicate if the sentence 
  # is given from the bot or from the user
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

  # The last loop (ideally) ends with a bot's answer,
  # thus making bot_flag equal to False.
  # Although, with data visualization and exploring,
  # we can see that this does not happen all the time.

  # Corner case: If the last answers was from the user, 
  # then we need to add one artificial 'ghost' response 
  # from the bot to make the dataset even.
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


# ## Load file 
# 
# Read the already-prepared tsv file from the local storage and clean it using the above-defined method.
# 
# The *preprocess_text() method must be compiled.*

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


# Visualize 3 random pairs of Q&A
for _ in range(3):
    print(random.choice(pairs))


# ## NN Design: Attention-based seq2seq Model 

# In[19]:


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
"""
(Description taken from PyTorch Tutorial, as referenced)

Calculate a set of attention weights.

Multiply attention weights by the encoder output vectors to create a weighted
combination. The result would contain information about that specific part of
the input sequence, and thus help the decoder choose the right output words.

To calculate the attention weights, we'll use a feed-forward layer that uses
the decoder's input and hidden state as inputs.

We will have to choose a max sentence length (input length, for encoder outputs),
wherein sentences of the max length will use all attention weights, while shorter
sentences would only use the first few.
"""
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


# ## NN Preprocessing
# Neural Networks require fixed-size integer vectors in order to operate.
# 
# That's why we will one-hot encode our sentences using the appropriate vocabulary (the encoder's or the decoder's one) each time.

# In[20]:


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

# In[21]:


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


# ## NN Training
# We will exploit the teacher forcing policy for training.
# 
# Also, we need to specify the encoder-decoder pipeline, along with any initialization needed.

# In[22]:


# Training helper method
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
            decoder_optimizer, criterion, max_length = MAX_LENGTH):
    """
    This method is responsible for the NN training. Specifically:

    - Runs input sentence through encoder
    - Keeps track of every output and the last hidden state
    - Then, the decoder is given the start of sentence token (SOS) 
            as its first input, and the last hidden state of the encoder
            as its first hidden state. We also utilize teacher forcing;
            The decoder uses the real target outputs as each next input.
    - Returns the current loss
    """

    # Train one iteration
    encoder_hidden = encoder.init_hidden()

    # Set gradients to zero 
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Get input and target length
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Init outputs to a zeros array equal to MAX_LENGTH 
    # and the encoder's latent dimensionality
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # Initialize the loss
    loss = 0 

    # Encode input
    for encoder_input in range(input_length):
        # Include hidden state from the last input when encoding current input
        encoder_output, encoder_hidden = encoder(input_tensor[encoder_input], encoder_hidden)
        encoder_outputs[encoder_input] = encoder_output[0, 0]

    # Decoder uses SOS token as first input
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)

    # Decoder uses last hidden state of encoder as first hidden state
    decoder_hidden = encoder_hidden

    # Teacher forcing: Feed the actual target as the next input instead of the predicted one
    for d_i in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                    decoder_hidden,
                                                                    encoder_outputs)

        loss += criterion(decoder_output, target_tensor[d_i])

        decoder_input = target_tensor[d_i] # Teacher forcing

    # Compute costs for each trainable parameter (dloss/dx)
    loss.backward()

    # Backpropagate & update parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# For a predefined number of iterations, we will train our Neural Network, using the above helper train() method.

# In[23]:


def train_iters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
    """
    Calls the train() method for a number of iterations.
    It tracks the time progress while initializing optimizers and cost function.
    In the same time, it creates the sets of the training pairs.
    """

    start = time.time() # Get start time
    print_loss_total = 0 # Reset after each print_every
    
    # Set optimizers
    #encoder_optimizer = optim.Adam(encoder.parameters(), amsgrad = True, lr=learning_rate)
    #decoder_optimizer = optim.Adam(encoder.parameters(), amsgrad = True, lr=learning_rate)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # Shuffle the training pairs
    training_pairs = [tensors_from_pair(random.choice(pairs)) for i in range(n_iters)]

    # Set the cost function
    criterion = nn.NLLLoss() # Also known as the multiclass cross-entropy 
    
    # For each iteration
    for i in range(1, n_iters + 1):
        training_pair = training_pairs[i - 1] # Create a training pair

        # Extract input and target tensor from the pair
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        # Train for each pair
        loss = train(input_tensor, target_tensor, encoder, decoder,
                encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss

        # Print progress
        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0 # Reset
            print('%s (%d %d%%) %.4f' % (time_since(start, i / n_iters),
                             i, i / n_iters * 100, print_loss_avg))


# Let's train our Neural Net.

# In[24]:


##### TRAIN 
hidden_size = 512 # Change arbitrarily depending on the results

# Instantiate Encoder and Attention Decoder
encoder = EncoderRNN(questions.n_words, hidden_size).to(device)
attention_decoder = AttnDecoderRNN(hidden_size, answers.n_words, dropout_p=0.2).to(device)

# Train for n_iters random samples
# The dataset holds 238051 dialogs while we filter some of them
# Obviously, deep learning computations need really high-performance hardware.
# Let's experiment with a number of iterations; I guess we just need a proof of concept.
n_iters = 500000#70000


# In[25]:


train_iters(encoder, attention_decoder, n_iters, print_every=(n_iters//15))


# ## Inference

# In[26]:


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


# ## Save

# In[28]:


encoder_name = 'encoder_serialized2.pt'
decoder_name = 'decoder_serialized2.pt'

# Serialize the encoder/decoder objects in your local directory
print('Saving model...')
torch.save(encoder, encoder_name)
torch.save(attention_decoder, decoder_name)

