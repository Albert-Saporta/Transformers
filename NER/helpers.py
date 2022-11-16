# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:28:22 2022

@author: alber
"""

import torch
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor


import datetime
import time

import pickle

#%% Save and load

def save_transformer(model):#ajouter num label, tokeniser,tag_values
    torch.save(model.state_dict(), 'JNLPBA_BERT.pth')
    """
    torch.save(
    dict(
        #optimizer=gen_opt.state_dict(),
        z_dim=z_dim,
        gen_hidden_layers =gen_hidden_layers,
        g_hidden_dim=g_hidden_dim,
        state_dict=gen.state_dict(),
    ),
    output)
    """
    


def load_transformer(model,pth_file,device):
    transformer_pth_file =torch.load(pth_file)
    transfo = model.to(device)
    transfo.load_state_dict(transformer_pth_file)
    return transfo.eval()