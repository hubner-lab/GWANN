import torch
import torch.nn as nn
import torch.nn.functional as F 

import functools
import operator

def shape_of_output(inputs, layers):
    sequential = nn.Sequential(*layers)
    return sequential(inputs).shape

def size_of_output(shape_of_input, list_of_layers):
    return functools.reduce(operator.mul, list(shape_of_output(shape_of_input, list_of_layers)))


class Net(nn.Module):
    def __init__(self,num_SNP,num_Samples,batch,width):
        super(Net,self).__init__()
            
        self.SNP = num_SNP
        self.samples = num_Samples
        self.batch = batch 
        self.width = width 
        
        inputs = torch.rand((1,1,self.width,self.samples//self.width))
        
        self.c1 = 5
        self.c2 = 4 
        self.c3 = 3 
        
        self.conv1 = nn.Conv2d(1,3,(self.c1,self.c1))
        self.conv2 = nn.Conv2d(3,5,(self.c2,self.c2))
        self.conv3 = nn.Conv2d(5,10,(self.c3,self.c3))
        
        self.final = size_of_output(inputs,[self.conv1,self.conv2,self.conv3]) 
        
        self.lin1 = nn.Linear(self.final,100)
        self.lin2 = nn.Linear(100,10)
        self.lin3 = nn.Linear(10,1)

    def forward(self,x):
        # x shape: (batch,SNP,samples) 

        x = x.view(self.batch,self.SNP,self.width,-1)
        x = x.view(self.SNP*self.batch,self.width,-1)
        x = torch.unsqueeze(x,1)
        

        x = self.conv1(x)
        # x = F.dropout(x,p = 0.2,training=self.training) 
        x = self.conv2(x)
        # x = F.dropout(x,p = 0.2,training=self.training) 
        x = self.conv3(x)

        x = torch.flatten(x, 1)

        # x = F.dropout(x,p = 0.2,training=self.training) 
        x = self.lin1(x)
        # x = torch.sigmoid(x)
        # x = F.dropout(x,p = 0.2,training=self.training) 
        x = self.lin2(x)
        # x = torch.sigmoid(x)
        # x = F.dropout(x,p = 0.2,training=self.training) 
        x = self.lin3(x)
        
        x = x.view(self.batch,self.SNP)

        return x
