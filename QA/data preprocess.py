# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:16:42 2022

@author: alber
"""

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

data_path="C:/Users/alber/Bureau/Development/NLP_data/QA/PIAF/piaf_v12.json"

with open(data_path, 'rb') as f:
  squad = json.load(f)
  
print(squad['data'][0].keys())
print(squad['data'][186]['paragraphs'][0]['context'])



train_contexts, train_questions, train_answers = read_data(data_path)

print(train_questions[0],train_answers[0])



add_end_idx(train_answers, train_contexts)

print(train_questions[-10000])
print(train_answers[-10000])

tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
print(train_encodings.keys())
no_of_encodings = len(train_encodings['input_ids'])
print(f'We have {no_of_encodings} context-question pairs')
#!! pas =1?? pas 0?
print(train_encodings['input_ids'][0])
print(tokenizer.decode(train_encodings['input_ids'][0]))



add_token_positions(train_encodings, train_answers)

print(train_encodings['start_positions'][:10])



#%% train
#camembert = CamembertModel.from_pretrained("camembert-base")
model = RobertaForQuestionAnswering.from_pretrained("camembert-base") #?BertForQuestionAnswering??
train_dataset = PIAF_SQuAD_Dataset(train_encodings)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Working on {device}')

N_EPOCHS = 5
optim = AdamW(model.parameters(), lr=5e-5)

model.to(device)
model.train()

for epoch in range(N_EPOCHS):
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

    loop.set_description(f'Epoch {epoch+1}')
    loop.set_postfix(loss=loss.item())