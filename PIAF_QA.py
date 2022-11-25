# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:56:40 2022

@author: alber
"""
from modules import *


import os
import json
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import transformers

from transformers import BertTokenizer, BertConfig,AutoTokenizer, AutoModel,CamembertTokenizer,CamembertTokenizerFast
from transformers import BertForTokenClassification, AdamW, BertForSequenceClassification,RobertaForQuestionAnswering
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nltk import pos_tag
from nltk.tree import Tree
from nltk.chunk import conlltags2tree
#https://colab.research.google.com/drive/1WxGxCFE_1cESJ02baaBY-HBHmGjSlxJx?usp=sharing
#%% Hyperparameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
device_name=torch.cuda.get_device_name(0)

n_epochs =5
max_grad_norm = 1.0
MAX_LEN = 75
bs = 4

tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
#camembert = CamembertModel.from_pretrained("camembert-base")
model = RobertaForQuestionAnswering.from_pretrained("camembert-base") #?BertForQuestionAnswering??
"""
    num_labels=12,
    output_attentions = False,
    output_hidden_states = False)
"""

optim = AdamW(model.parameters(), lr=5e-5)
pth_file_name="piaf"
#%% data Preprocessing
data_path="C:/Users/alber/Bureau/Development/Data/NLP_data/QA/PIAF/piaf_v12.json"
"""
with open(data_path, 'rb') as f:
  squad = json.load(f)
  
print(squad['data'][0].keys())
print(squad['data'][186]['paragraphs'][0]['context'])
"""


#%%% contexts, questions, answers
train_contexts, train_questions, train_answers = read_data(data_path)

print(train_questions[0],train_answers[0])

#%%% add end idx

add_end_idx(train_answers, train_contexts)
print(train_questions[-10000])
print(train_answers[-10000])


#%%% tokens
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
no_of_encodings = len(train_encodings['input_ids'])

print(train_encodings.keys())

print(f'We have {no_of_encodings} context-question pairs')
#!! pas =1?? pas 0?
print(train_encodings['input_ids'][0])
print(tokenizer.decode(train_encodings['input_ids'][0]))



add_token_positions(train_encodings, train_answers,tokenizer)

print(train_encodings['start_positions'][:10])

#%%% print pre processing
"""
print(train_encodings)
"""


#%%% Prepare data for training
dataset = PIAF_SQuAD_Dataset(train_encodings)
split_dat,split_dat2=train_test_split(dataset,test_size=0.3)
train_dataset, val_dataset  = train_test_split(split_dat2,test_size=0.1)

train_loader = DataLoader(train_dataset, batch_size=bs,pin_memory=True, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=bs,pin_memory=True, shuffle=True)


 



#%% Training
#%%% Fine tune 


#%%% training loop
print("")
model.to(device)
model.train()
loss_values, validation_loss_values = [], []

for epoch in range(n_epochs):
    
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.
    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
      optim.zero_grad()
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      start_positions = batch['start_positions'].to(device)
      end_positions = batch['end_positions'].to(device)
      outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
      loss = outputs[0]
      loss.backward()
      optim.step()
  
      loop.set_description(f'Epoch {epoch+1}/{n_epochs}')
      loop.set_postfix(loss=loss.item())
      #save_transformer(model,pth_file_name)
      
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    
    model.eval()

    acc = []
    
    for batch in tqdm(valid_loader):
      with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_true = batch['start_positions'].to(device)
        end_true = batch['end_positions'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
    
        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
    
        acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
        acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
    
    acc = sum(acc)/len(acc)
    
    print("\n\nT/P\tanswer_start\tanswer_end\n")
    for i in range(len(start_true)):
      print(f"true\t{start_true[i]}\t{end_true[i]}\n"
            f"pred\t{start_pred[i]}\t{end_pred[i]}\n")





    

#%% Visu 


"""
# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(loss_values, 'b-o', label="training loss")
plt.plot(validation_loss_values, 'r-o', label="validation loss")

# Label the plot.
plt.title("Learning curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()
"""

