# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:42:21 2022

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
from transformers import BertTokenizer, BertConfig
import transformers
from transformers import BertForTokenClassification, BertTokenizer, BertConfig,AutoTokenizer, AutoModel

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
from transformers import pipeline

# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
#%% test data
test_file_path="C:/Users/alber/Bureau/Development/NLP_data/NER/JNLPBA/test.tsv"
pth_file_path="C:/Users/alber/Bureau/Development/training_results/JNLPBA_BERT.pth"

data=pd.read_csv(test_file_path, sep='\t',names=["word","tag"])
data.dropna(axis=0, inplace=True)
data.drop(data.index[data['word'] == "-DOCSTART-"], inplace = True)

sent=[]
a=1
for i in range(len(data)):
    if  not data.word.iloc[i].endswith("."):
        sent.append(a)
    elif data.word.iloc[i].endswith("."):
        sent.append(a)
        a+=1
        #print(i)
data['sentence #']=sent

agg_func = lambda s: (" ".join(w for w in s["word"]))

sentences=data.groupby("sentence #").apply(agg_func)
#print(sentences.iloc[0])

#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


modelss = BertForTokenClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT",
    num_labels=12,
    output_attentions = False,
    output_hidden_states = False
).to(device)

model=load_transformer(modelss,pth_file_path,device)
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


tag_values=['B-cell_type', 'B-cell_line', 'I-cell_line', 'I-DNA', 'B-RNA', 'B-DNA', 'O', 'I-cell_type', 'I-protein', 'B-protein', 'I-RNA']
#%% Test 
test_sentence = sentences.iloc[2]
tokenized_sentence = tokenizer.encode(test_sentence)
input_ids = torch.tensor([tokenized_sentence]).cuda()

with torch.no_grad():
    output = model(input_ids)
label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
# join bpe split tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])

new_tokens, new_labels = [], []
for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(tag_values[label_idx])
        new_tokens.append(token)
for token, label in zip(new_tokens[1:-1], new_labels[1:-1]):
    print("{}\t{}".format(label, token))
 
    
