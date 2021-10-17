import click
import sys

import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.autograd.profiler as profiler
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as T

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import random
import time
import functools
import operator
from functools import partial
import scipy.stats as stats
import seaborn as sns

import allel
from cyvcf2 import VCF

import sklearn



@click.group()
def cli1():
    pass

@cli1.command()
@click.option('-v', '--vcf','vcf',required=True)
@click.option('-p', '--pheno','pheno',required=True)

@click.option('-s', '--samples','n_samples',default=250,type=int)
@click.option('-S', '--SNPs','n_snps',default=250,type=int)
@click.option('-w', '--width','width',default=10,type=int)
@click.option('--seed','seed',default=random.randrange(sys.maxsize),type=int)

def run(vcf,pheno,n_samples,n_snps,width,seed):
    """Run"""

    from net import Net

    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir("vcf_data"):
        os.mkdir("vcf_data")

    if not os.path.exists("vcf_data/test.npz"):
        allel.vcf_to_npz(vcf, 'vcf_data/test.npz', fields='*', overwrite=True)

    callset = np.load("vcf_data/test.npz",allow_pickle=True)

    vcf = callset['calldata/GT']

    chrom = callset['variants/CHROM']

    vcf = (vcf[:,:,0] + vcf[:,:,1]) / 2

    final_vcf = torch.from_numpy(vcf).float().to(device)
    pheno = pd.read_csv(pheno,index_col=None,header=None,sep=' ')
    pheno_sorted = pheno.sort_values(by=[2])

    sorted_axes = np.array(pheno_sorted.index.values)
    sorted_vcf = final_vcf[:,sorted_axes]

    df_chrom = pd.DataFrame(chrom)
    color_labels = df_chrom[0].unique()

    # n_snps = sorted_vcf.shape[0] 
    # n_snps = 200

    net = Net(n_snps,n_samples,1,width).to(device)
    net.load_state_dict(torch.load("models/net.pt")['model_state_dict'])

    net.eval()
    with torch.no_grad():
        input_s = torch.split(sorted_vcf,n_snps)
        output = torch.zeros((sorted_vcf.shape[0])).float().to(device)
        
        for j in range(len(input_s)):
            input_tmp = input_s[j]
            if n_snps - input_s[j].shape[0] > 0:
                input_tmp = sorted_vcf[-n_snps:]
            pad_samples = n_samples - input_s[j].shape[1]
            pad_2 = torch.zeros((n_snps,pad_samples)).float().to(device) 
            input = torch.cat((pad_2,input_tmp),1)

            outputs = net(input)

            output[j*n_snps:j*n_snps + input_s[j].shape[0]] = outputs[:,-input_s[j].shape[0]:]

        plt.clf()
        current = 0
        for i in range(15):
            color = "black"
            if i % 2 == 0:
                color = "blue"

            indexes = np.where(df_chrom == "Ha{0}".format(i+1))[0]

            plt.scatter(range(current,current + len(indexes)),output[indexes].cpu(),s=1,c=color)

            current = current + len(indexes)


        plt.savefig('GWAS.png')




@click.group()
def cli2():
    pass

@cli2.command()
@click.option('-p', '--population-size','pop',required=True,type=int)
def simulate(pop):
    """Simulate"""

@click.group()
def cli3():
    pass

@cli3.command()
@click.option('-e', '--epochs','epochs',default=100,type=int)
@click.option('-s', '--samples','n_samples',default=250,type=int)
@click.option('-S', '--SNPs','n_snps',default=200,type=int)
@click.option('-b', '--batch','batch',default=1,type=int)
@click.option('-r', '--ratio','ratio',default=0.8,type=float)
@click.option('-w', '--width','width',default=10,type=int)
@click.option('--path','path',required=True,type=str)
@click.option('--seed','seed',default=random.randrange(sys.maxsize),type=int)
@click.option('--verbose/;','debug',default=False)

def train(epochs,n_samples,n_snps,batch,ratio,width,path,seed,debug):
    """Train"""

    from net import Net
    from dataset  import DatasetPhenosim
    
    torch.manual_seed(seed)

    full_dataset = DatasetPhenosim(n_samples,n_snps,path)

    train_size = int(ratio * len(full_dataset))
    test_size =  len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],torch.Generator().manual_seed(seed))

    dataloader_train = DataLoader(train_dataset, batch_size=batch,shuffle=True, num_workers=0)
    dataloader_test =  DataLoader(test_dataset,  batch_size=batch,shuffle=True, num_workers=0)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(n_snps,n_samples,batch,width).to(device)

    if debug:
        print(seed)
        print(net)
        print(device)
        print("SNPs: {0} , samples: {1}, batch: {2} , width : {3}".format(n_snps,n_samples,batch,width))


    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=1e-2,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.7)

    min_loss = np.Infinity
    max_accuracy = 0
    
    if not os.path.isdir("models"):
        os.mkdir("models")

    clip_grad_norm = 5
    e = 0

    plot_average = True

    if plot_average:
        fig,ax = plt.subplots(2,2)

        ax[0,0].set_xlabel("False Positives")
        ax[0,1].set_xlabel("False Negatives")
        ax[1,0].set_xlabel("True Positives")
        ax[1,1].set_xlabel("True Negatives")


    while e < epochs:
        total_loss = 0
        times = 0

        n_FN = 0
        n_FP = 0
        n_TP = 0
        n_TN = 0
    
        avr_FN = np.zeros((width,n_samples // width))
        avr_FP = np.zeros((width,n_samples // width))
        avr_TN = np.zeros((width,n_samples // width))
        avr_TP = np.zeros((width,n_samples // width))
    

        net.eval()
        with torch.no_grad():
            for i,data in enumerate(dataloader_test):

                inputs = data['input'].float().to(device)
                pred = data['output'].float().to(device)
                
                outputs = net(inputs)

                loss = F.mse_loss(outputs,pred)
                total_loss += loss.item()
                times += 1

                if plot_average:
                    x = inputs.cpu().detach().clone() 
                    x = x.view(batch,n_snps,width,-1)
                    x = x.view(batch*n_snps,width,-1)

                    x = torch.unsqueeze(x,1)
            
                    pred_copy = np.array(pred.detach().clone().cpu().flatten()) 
                    outputs_copy = np.array(outputs.detach().clone().cpu().flatten()) 
                    pred_copy = pred_copy.astype(int)
                    

                    false_ind = np.where(np.sign(outputs_copy) != pred_copy)
                    false = pred_copy[false_ind]
                    false_ind = false_ind[0]
                    
                    true_ind = np.where(np.sign(outputs_copy) == pred_copy)
                    true = pred_copy[true_ind]
                    true_ind = true_ind[0]
                    
                    false_negatives = false_ind[np.where(false == 1)] 
                    false_positives = false_ind[np.where(false == -1)]  
                    true_positives = true_ind[np.where(true == 1)]
                    true_negatives = true_ind[np.where(true == -1)]



                    if false_positives.size != 0 :
                        avr_FP += np.average(x[false_positives,0,:,:],axis=0)
                        n_FP += false_positives.size
                    if false_negatives.size != 0:
                        avr_FN += np.average(x[false_negatives,0,:,:],axis=0)
                        n_FN += false_negatives.size
                    if true_positives.size != 0 :
                        avr_TP += np.average(x[true_positives,0,:,:],axis=0)
                        n_TP += true_positives.size
                    if true_negatives.size != 0 :
                        avr_TN += np.average(x[true_negatives,0,:,:],axis=0)
                        n_TN += true_negatives.size

        if debug and plot_average:
            ax[0,0].matshow(avr_FP)
            ax[0,1].matshow(avr_FN)
            ax[1,0].matshow(avr_TP)
            ax[1,1].matshow(avr_TN)

            print("FP: {0}, FN : {1}, TP: {2}, TN: {3}".format(n_FP,n_FN,n_TP,n_TN))
            print(e,total_loss/times) 
        
            if e % 20 == 0:
                fig.canvas.draw()
                plt.savefig('matrix.png')
     
        if total_loss/times < min_loss:
            min_loss  = total_loss/times
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "models/net.pt")
    
        net.train()
        for i,data in enumerate(dataloader_train):
            optimizer.zero_grad()

            inputs = data['input'].float().to(device)
            pred = data['output'].float().to(device)

            outputs = net(inputs)
            loss = criterion(outputs, pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad_norm)
            optimizer.step()    

        e += 1 

        if e % 100 == 0:
            scheduler.step()


cli = click.CommandCollection(sources=[cli1, cli2,cli3])

if __name__ == '__main__':
    cli()
