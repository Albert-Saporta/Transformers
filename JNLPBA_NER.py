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

from transformers import BertTokenizer, BertConfig,AutoTokenizer, AutoModel
from transformers import BertForTokenClassification, AdamW, BertForSequenceClassification
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
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
#https://www.pragnakalp.com/named-entity-recognition-ner-using-biobert-demo/
#%% Hyperparameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
device_name=torch.cuda.get_device_name(0)

n_epochs = 15
max_grad_norm = 1.0
MAX_LEN = 75
bs = 24
pth_file_name="JNLPBA_BERT2"
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = BertForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",   
    num_labels=12,
    output_attentions = False,
    output_hidden_states = False)

#%%data
train_file_path="C:/Users/alber/Bureau/Development/NLP_data/NER/JNLPBA/train.tsv"
data=pd.read_csv(train_file_path, sep='\t',names=["word","tag"])
data.dropna(axis=0, inplace=True)
data.drop(data.index[data['word'] == "-DOCSTART-"], inplace = True)
#%%Preprocessing

#%%% add sentence number 
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

#%%% word tags pairs
getter = SentenceGetter(data)



sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
labels = [[s[1] for s in sentence] for sentence in getter.sentences]

tag_values = list(set(data["tag"].values))
#tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}
tag2idx["PAD"]=0
#Padding is addded end of each sentence,!! 0 is for PAD??


#%%% tokens
tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs,tokenizer)
    for sent, labs in zip(sentences, labels)
]



tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

#cut and pad to the desied length 75 
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")

#attenation mask to ignore PAD token
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
#%%% print pre processing
"""
print(getter.sentences[0])
print(data.head())
print(tag_values)
print("tag2idx:",tag2idx)
print(tokenized_texts)
print("attention_masks:",attention_masks)

"""
print(tag_values)

print("tag2idx[PAD]:",tags[0],input_ids[0])
print("sentence/lab:",labels[0],tokenized_texts[0])


#%%% Prepare data for training

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)  

# convert to torch tenors
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

#training time shuffling of the data and testing time we pass them sequentially
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

#%% Training
#%%% Fine tune 



model.cuda();

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

#schduler to reduce learning rate linearly throughout the epochs



# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * n_epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

#%%% training loop


## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []
#print(torch.cuda.current_device())

for epoch in range(n_epochs):

    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0
    loop = tqdm(train_dataloader, leave=True)

    # Training loop
    for batch in loop: 
        # add batch to gpu
        #print(batch)
        batch = tuple(t.type(torch.LongTensor).to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Always clear any previously calculated gradients before performing a backward pass.
        #print("devices:",b_input_ids.get_device())
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        #scheduler.step()
        loop.set_description(f'Epoch {epoch+1}/{n_epochs}')
        loop.set_postfix(loss=loss.item())
    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.type(torch.LongTensor).to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        eval_accuracy += flat_accuracy(logits, label_ids)
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    #print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print()
    save_transformer(model)

    

#%% Visu 



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

#%% eval

test_sentence = "The study demonstrated a decreased level of glucocorticoid	receptor ."

tokenized_sentence = tokenizer.encode(test_sentence)
input_ids = torch.tensor([tokenized_sentence]).cuda()

with torch.no_grad():
    output = model(input_ids)
label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
# join bpe split tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
print(label_indices)

new_tokens, new_labels = [], []
for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(tag_values[label_idx])
        new_tokens.append(token)
        
for token, label in zip(new_tokens, new_labels):
    print("{}\t{}".format(label, token))
