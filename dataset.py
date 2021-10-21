import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.autograd.profiler as profiler
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as T

import numpy as np
import pandas as pd

import glob
import re

class DatasetPhenosim(Dataset):
    def __init__(self,samples,SNP,root_path):

        self.samples = samples
        self.SNP = SNP
        self.root_path = root_path

        self.cache = dict()

        # self.missing_data = False 

    def __len__(self):


        s = glob.glob("{path}*0.emma_geno".format(path=self.root_path))
        return len(s) 

    def __getitem__(self,idx):

        # if torch.is_tensor(idx):
           # idx = idx.tolist()
        if idx in self.cache.keys():
            return self.cache[idx] 
        f = idx

        genotype_path  = "{base}{f}0.emma_geno".format(f=f,base=self.root_path)
        NrSNP_path = "{base}{f}0.causal".format(f=f,base=self.root_path)
        Ysim_path = "{base}{f}0.emma_pheno".format(f=f,base=self.root_path)
        
        data_G = pd.read_csv(genotype_path,index_col=None,header=None,sep='\t').fillna(0)
        # data_G = pd.read_csv(genotype_path,index_col=None,header=None,sep='\t').fillna(-1)
        data_SNP = pd.read_csv(NrSNP_path,index_col=None,header=None,sep='\t')
        data_Ysim = pd.read_csv(Ysim_path,index_col=None,header=None,sep='\t')
        
        data_Ysim_sorted = data_Ysim.T.sort_values(by=[0])
        data_SNP = data_SNP.sort_values(by=[0])

        causal_SNP = data_SNP[1].to_numpy(dtype=int) 
        causal_SNP_eff = data_SNP[3].to_numpy(dtype=float) 

        min_eff = min(causal_SNP_eff)
        max_eff = max(causal_SNP_eff)

        if min_eff != max_eff:
            leftSpan = max_eff - min_eff 
            rightSpan = 1 
            causal_SNP_eff = ((causal_SNP_eff - min_eff) / leftSpan)
        else:
            causal_SNP_eff = 1 

        
        sorted_axes = data_Ysim_sorted.index.values
        data_input = data_G.T.reindex(sorted_axes).T.to_numpy()

        data_output = np.empty(data_input.shape[0])
        
        data_output[:] = -1 
        data_output[causal_SNP] = 1 
        # data_output[causal_SNP] = 1 

        # data_output[causal_SNP] = causal_SNP_eff 
        # population = pd.DataFrame(self.init_pop)
        # population = population.reindex(sorted_axes).to_numpy().flatten()
        
        n = self.SNP - len(causal_SNP)
        
        indexes = np.random.choice(np.setdiff1d(range(data_input.shape[0]),causal_SNP), n, replace=False)  
        indexes = np.concatenate((indexes,causal_SNP))
        np.random.shuffle(indexes)

        data_input = data_input[indexes,:]
        data_output = data_output[indexes]

        # output = torch.from_numpy(data_output).to(int)
        # output = output.unsqueeze(1)
        # output = torch.zeros(output.shape[0],2).scatter_(1, output, 1)

        
        final_output = { 'input':torch.from_numpy(data_input) , 
                         'output': torch.from_numpy(data_output)}

        self.cache[idx] = final_output

        return final_output
