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
            
        # self.SNP = num_SNP
        # self.batch = batch 
        self.samples = num_Samples
        self.width = width 
        
        inputs = torch.rand((1,1,self.width,self.samples//self.width))
        
        c1 = 3
        c2 = 2 
        c3 = 2 

        p1 = 2
        p2 = 2
        
        self.conv1 = nn.Conv2d(1,3,(c1,c1))
        self.maxpool1 = nn.MaxPool2d(p1)
        self.conv2 = nn.Conv2d(3,5,(c2,c2))
        self.maxpool2 = nn.MaxPool2d(p1)
        self.conv3 = nn.Conv2d(5,10,(c3,c3))
        
        self.final = size_of_output(inputs,[
            self.conv1,
            self.maxpool1,
            self.conv2,
            self.maxpool2,
            self.conv3
        ]) 
        
        self.lin1 = nn.Linear(self.final,10)
        # self.lin2 = nn.Linear(100,10)
        self.lin2 = nn.Linear(10,1)

    def forward(self,x):
        # x shape: (batch,SNP,samples) 

        BATCH,SNP,_ = x.shape

        x = x.view(BATCH,SNP,self.width,-1)
        x = x.view(BATCH*SNP,self.width,-1)
        x = torch.unsqueeze(x,1)


        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)

        x = torch.flatten(x, 1)

        # x = torch.sigmoid(x)

        x = self.lin1(x)
        x = F.dropout(x,p = 0.2,training=self.training) 
        x = self.lin2(x)

        # x = torch.nn.functional.softmax(x,dim=1)
        # x = F.dropout(x,p = 0.2,training=self.training) 
        # x = self.lin3(x)
        
        x = x.view(BATCH,SNP)

        return x
 
