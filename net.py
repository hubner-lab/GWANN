import torch
import torch.nn as nn
import torch.nn.functional as F 

import functools
import operator

import math

def shape_of_output(inputs, layers):
    sequential = nn.Sequential(*layers)
    return sequential(inputs).shape

def size_of_output(shape_of_input, list_of_layers):
    return functools.reduce(operator.mul, list(shape_of_output(shape_of_input, list_of_layers)))


class Net(nn.Module):
    def __init__(self,num_SNP,num_Samples,batch,width):
        super(Net,self).__init__()
            
        self.samples = num_Samples
        self.width = width 
        self.height = self.samples//self.width
        
        inputs = torch.rand((1,1,self.width,self.height))

        c1 = 2
        c2 = 2 
        c3 = 2 

        p = 2
        
        self.maxpool = nn.MaxPool2d(p)

        self.conv1 = nn.Conv2d(1,3,(c1,c1))
        self.conv2 = nn.Conv2d(3,5,(c2,c2))
        self.conv3 = nn.Conv2d(5,10,(c3,c3))

        self.final = size_of_output(inputs,[
            self.conv1,
            self.maxpool,
            self.conv2,
            self.maxpool,
            self.conv3
        ]) 

        self.lin1 = nn.Linear(self.final,10)
        # self.lin2 = nn.Linear(100,10)
        # self.lin2 = nn.Linear(10,2)
        self.lin2 = nn.Linear(10,1)

    def forward(self,x):
        # x shape: (batch,SNP,samples) 

        BATCH,SNP,SAMPLES = x.shape

        x = x.view(BATCH,SNP,self.width,-1)
        x = x.view(BATCH*SNP,self.width,-1)
        x = torch.unsqueeze(x,1)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)

        x = torch.flatten(x, 1)

        # x = torch.sigmoid(x)

        x = self.lin1(x)
        x = F.dropout(x,p = 0.2,training=self.training) 
        x = self.lin2(x)

        # x = torch.nn.functional.softmax(x,dim=1)
        # x = x.view(BATCH,2,SNP)

        x = x.view(BATCH,SNP)

        x_clone = x.detach().clone()
        # x = torch.sigmoid(x)

        return x
 
