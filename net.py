import torch
import torch.nn as nn
import torch.nn.functional as F 

import functools
import operator

import math

# def shape_of_output(inputs, layers):
    # sequential = nn.Sequential(*layers)
    # return sequential(inputs).shape

# def size_of_output(shape_of_input, list_of_layers):
    # return functools.reduce(operator.mul, list(shape_of_output(shape_of_input, list_of_layers)))

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
        
        self.sequential = nn.Sequential(
            nn.Conv2d(1,3,(c1,c1)),
            nn.MaxPool2d(p),
            nn.Conv2d(3,5,(c2,c2)),
            nn.MaxPool2d(p),
            nn.Conv2d(5,10,(c3,c3))
        )


        self.sequential_pop = nn.Sequential(
            nn.Conv2d(1,3,(c1,c1)),
            nn.MaxPool2d(p),
            nn.Conv2d(3,5,(c2,c2)),
            nn.MaxPool2d(p),
            nn.Conv2d(5,10,(c3,c3))
        )


        self.final = torch.numel(self.sequential(inputs))

        self.lin1 = nn.Linear(2*self.final,10)
        self.lin2 = nn.Linear(10,1)

    def forward(self,x,pop):
        # x shape: (batch,SNP,samples) 


        BATCH,SNP,SAMPLES = x.shape

        x = x.view(BATCH,SNP,self.width,-1)
        x = x.view(BATCH*SNP,self.width,-1)
        x = torch.unsqueeze(x,1)

        x = self.sequential(x)

        x = torch.flatten(x, 1)


        pop = pop.view(BATCH,SAMPLES)
        pop = pop.view(BATCH,self.width,-1)
        pop = torch.unsqueeze(pop,1)

        pop = self.sequential_pop(pop)
        pop = torch.flatten(pop, 1)

        pop = pop.repeat(SNP,1)


        output = torch.cat((x, pop), 1) 

        output = self.lin1(output)
        output = F.dropout(output,p = 0.2,training=self.training) 
        output = self.lin2(output)

        # x = torch.nn.functional.softmax(x,dim=1)
        # x = x.view(BATCH,2,SNP)

        output = output.view(BATCH,SNP)

        # x_clone = x.detach().clone()
        # x = torch.sigmoid(x)

        return output
 
