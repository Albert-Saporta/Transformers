# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:00:14 2022

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
#%% Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pth_file_path="C:/Users/alber/Bureau/Development/DeepLearning/training_results/piaf.pth"
tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")
transfo = RobertaForQuestionAnswering.from_pretrained("camembert-base") #?BertForQuestionAnswering?
model=load_transformer(transfo,pth_file_path,device)
#%% evaluation functions


def get_prediction(context, question):
  inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
  outputs = model(**inputs)
  
  answer_start = torch.argmax(outputs[0])  
  answer_end = torch.argmax(outputs[1]) + 1 
  
  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
  
  return answer

def normalize_text(s):
  """Removing articles and punctuation, and standardizing whitespace."""
  import string, re
  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)
  def white_space_fix(text):
    return " ".join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, truth):
    return bool(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()
  
  # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)
  
  common_tokens = set(pred_tokens) & set(truth_tokens)
  
  # if there are no common tokens then f1 = 0
  if len(common_tokens) == 0:
    return 0
  
  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)
  
  return round(2 * (prec * rec) / (prec + rec), 2)
  
def question_answer(context, question,answer):
  prediction = get_prediction(context,question)
  em_score = exact_match(prediction, answer)
  f1_score = compute_f1(prediction, answer)

  print(f'Question: {question}')
  print(f'Prediction: {prediction}')
  print(f'True Answer: {answer}')
  print(f'Exact match: {em_score}')
  print(f'F1 score: {f1_score}\n')
  
#%% evaluation


context_salari??="Le Minist??re de la Jeunesse et des Sports estime ?? 100 000 (58 % d'hommes pour 42 % de femmes) le nombre de salari??s travaillant pour le secteur sportif en France pour quelque 20 000 employeurs."
questions_salari?? = ["Quel minist??re?","Combien de personnes travaillent au minist??re des sports?","Combien d'employeurs?"]
answers_salari?? = ["Minist??re de la Jeunesse et des Sports","100 000","20 000"]

context_d??penses="Les d??penses des m??nages repr??sentent plus de 50 % de ces montants (14,2 milliards d'euros en 2003 et 12 milliards d'euros en 2019), contre 7,9 milliards d'euros pour les collectivit??s locales, 3,2 pour l'??tat, et 2,2 pour les entreprises. Parmi les d??penses sportives des m??nages en 2003, 3,7 milliards sont consacr??s aux v??tements de sport et chaussures, 2 aux biens durables, 2,7 aux autres biens et 5,8 aux services."
questions_d??penses = ["Quel montant en 2003?","Quel montant en 2019?"]
answers_d??penses = ["14,2 milliards d'euros","12 milliards d'euros"]

context3="Jakob Bohme est n?? le 8 mars 1575. Ses parents jouissent d'une relative aisance gr??ce ?? la carri??re du grand-p??re, Ambroise Bohme,  et d'un certain niveau de culture. Aussi envoient-ils leur fils ?? l'??cole, o?? il apprend ?? lire, ??crire et compter, ainsi que des rudiments de latin; comme sa constitution fragile l'emp??che de travailler la terre."
questions3= ["Qui est de constitution fragile ?", "Qui est le grand-p??re de Jakob?","Les parents de Jakob sont ils riches?"]
answers3=["Jakob Bohme","Ambroise B??hme","relative aisance"]


#%%% context
context,questions,answers=context_salari??,questions_salari??,answers_salari??
for question, answer in zip(questions, answers):
  question_answer(context, question, answer)

context,questions,answers=context_d??penses,questions_d??penses,answers_d??penses
for question, answer in zip(questions, answers):
  question_answer(context, question, answer)

# does not work with ??
context,questions,answers=context3,questions3,answers3
for question, answer in zip(questions, answers):
  question_answer(context, question, answer)