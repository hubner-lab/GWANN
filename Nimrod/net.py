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

        self.final = torch.numel(self.sequential(inputs))

        self.lin_seq = nn.Sequential(
            nn.Linear(self.final, 10),
            nn.Dropout(p=0.2),
            # nn.ReLU(),
            nn.Linear(10,1),
            # nn.ReLU()
        )


        self.pop_seq = nn.Sequential(
                nn.Linear(self.samples,self.samples // 2),
                nn.Dropout(p=0.2),
                # nn.ReLU(),
                nn.Linear(self.samples // 2,self.final),
                nn.Sigmoid(),
        )

    def forward(self,x,pop):
        BATCH,SNP,SAMPLES = x.shape
        #SAMPLES - number of individual simulated genomes, available for sampling
        #SNP - number of unique SNP-sites (sampled randomly)
        #BATCH - number of resamplings of the SNP-sites
        #SNP * BATCH = total number of unique SNP-sites
   
        x = x.view(BATCH,SNP,self.width,-1)
        x = x.view(BATCH*SNP,self.width,-1)
        x = torch.unsqueeze(x,1)

        x = self.sequential(x)
        x = torch.flatten(x, 1)

        pop = pop.view(BATCH,SAMPLES)
        pop = self.pop_seq(pop)
        pop = pop.repeat(SNP,1)

        output = x * pop 
        # output = x 

        output = self.lin_seq(output)
        output = output.view(BATCH,SNP)

        return output
