# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:42:21 2022

@author: alber
"""

from helpers import *

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
from transformers import BertForTokenClassification, AdamW
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

#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


modelss = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=12,
    output_attentions = False,
    output_hidden_states = False
)

model=load_transformer(modelss,"tensor.pt",device)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False) 


#%% Test 
ner = pipeline('ner', model=model)
test_sentence = "The study demonstrated a decreased level of glucocorticoid	receptor"
# #medical notes
ner(test_sentence)
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
        #new_labels.append(tag_values[label_idx])
        new_tokens.append(token)
        
for token, label in zip(new_tokens, new_labels):
    print("{}\t{}".format(label, token))
