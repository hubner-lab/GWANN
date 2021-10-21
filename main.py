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

import sklearn
import cv2

from pathlib import Path
import re

import subprocess
import shlex
import multiprocessing

import csv 

def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

@click.group()
def cli1():
    pass

@cli1.command()
@click.option('-v', '--vcf','vcf',required=True)
@click.option('-p', '--pheno','pheno_path',required=True)

@click.option('-s', '--samples','n_samples',default=250,type=int)
@click.option('-w', '--width','width',default=10,type=int)
@click.option('--seed','seed',default=random.randrange(sys.maxsize),type=int)
@click.option('--model','model',default="models/net.pt")
@click.option('--output','output_path',default="results/GWAS.png")

def run(vcf,pheno_path,n_samples,width,seed,model,output_path):
    """Run"""

    from net import Net

    if not Path("vcf_data").is_dir():
        Path.mkdir("vcf_data")

    npz_loc = "vcf_data/{0}.npz".format(Path(vcf).stem)

    if not Path(npz_loc).is_file():
        allel.vcf_to_npz(vcf, npz_loc, fields='*', overwrite=True)


    callset = np.load(npz_loc,allow_pickle=True)
    vcf = callset['calldata/GT']
    chrom = callset['variants/CHROM']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # tmp_vcf = np.empty(vcf[:,:,0].shape)
    tmp_vcf = (vcf[:,:,0] + vcf[:,:,1]) / 2
    # tmp_vcf[np.where(tmp_vcf == 0.5)] = 0 

    final_vcf = torch.from_numpy(tmp_vcf).float().to(device)

    n_snps = final_vcf.shape[0] // 4

    if not Path(pheno_path).is_file():
        print("Invalid file pheno")
        exit(1)

    pheno = pd.read_csv(pheno_path,index_col=None,header=None,sep=' ')
    pheno_sorted = pheno.sort_values(by=[2])

    sorted_axes = np.array(pheno_sorted.index.values)
    sorted_vcf = final_vcf[:,sorted_axes]

    df_chrom = pd.DataFrame(chrom)
    chrom_labels = df_chrom[0].unique().tolist()

    input_s = torch.split(sorted_vcf,n_snps)
    output = torch.zeros((sorted_vcf.shape[0])).float().to(device)
        
    net = Net(n_snps,n_samples,1,width).to(device)
    net.load_state_dict(torch.load(model)['model_state_dict'])

    net.eval()
    with torch.no_grad():
        for j in range(len(input_s)):
            input_tmp = input_s[j]
            if n_snps - input_s[j].shape[0] > 0:
                input_tmp = sorted_vcf[-n_snps:]
            pad_samples = n_samples - input_s[j].shape[1]
            pad_2 = torch.zeros((n_snps,pad_samples)).float().to(device) 
            input = torch.cat((pad_2,input_tmp),1)
            input = torch.unsqueeze(input,0)

            outputs = net(input)

            output[j*n_snps:j*n_snps + input_s[j].shape[0]] = outputs[:,-input_s[j].shape[0]:]


    output = output.cpu()

    plt.clf()
    fig,ax = plt.subplots(1)

    current = 0

    chrom_labels.sort(key=num_sort)
    chr_loc = []


    min = 0
    print(100 * (torch.count_nonzero(output > min)/output.shape[0]).item())

    color = ""
    output[np.where(output <= min)] = min

    for chr in chrom_labels:
        if color == "blue":
            color = "black"
        else: 
            color = "blue"

        indexes = np.where(df_chrom == chr)[0]
        ax.scatter(range(current,current + len(indexes)),output[indexes],s=1,color=color)
        chr_loc.append((2*current + len(indexes)) /2 )
        current = current + len(indexes)

    ax.set_xticks(chr_loc)
    ax.set_xticklabels(chrom_labels)
    plt.setp(ax.get_xticklabels(), rotation=70, horizontalalignment='right')
    fig.tight_layout()
    fig.savefig(output_path)




def simulate_helper(genome_command,phenosim_command,seed,i):
    out_file = open('simulation/data/genome{0}.txt'.format(i),'w')
    subprocess.call(genome_command + ["{0}".format(seed[i])],stdout=out_file)
    out_file.close()

    phenosim_command = shlex.split(phenosim_command.format(i))
    subprocess.call(phenosim_command,stdout=subprocess.DEVNULL)


@click.group()
def cli2():
    pass

@cli2.command()
@click.option('-p', '--population-size','pop',required=True,type=int)
@click.option('-s', '--samples','n_samples',required=True,type=int)
@click.option('-n', '--n_simulation','n_sim',required=True,type=int)
@click.option('-S', '--causal_snps','n_snps',default=1,type=int)
@click.option('-m', '--maf','maf',default=0.05,type=float)
@click.option('--miss','miss',default=0.03,type=float)
@click.option('--equal_variance/;','equal',default=False)

def simulate(pop,n_samples,n_sim,n_snps,maf,miss,equal):
    """Simulate"""

    seed_arr = np.array(list(range(pop))) + np.random.randint(1,1000000)
    np.random.shuffle(seed_arr)

    seed = 0

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpus)

    genome_command = shlex.split("simulation/genome/genome -s {pop} -pop 1 {samples} -seed".format(pop=pop,samples=n_samples))
    phenosim_command = "python2 simulation/phenosim/phenosim.py -i G -f simulation/data/genome{{0}}.txt --outfile simulation/data/{{0}} --maf_r {maf},1.0 --maf_c {maf} --miss {miss}".format(maf=maf,miss=miss)

    if n_snps > 1:

        variance = np.ones(n_snps)
        if equal:
            variance = variance / (2*n_snps)
        else:
            variance = np.random.dirichlet(variance,size=1)

        var_str = np.array2string(variance,precision=5,separator=',')
        var_str = re.sub("\[|\s*|\]","",var_str)
        phenosim_command += " -n {snps} -v {var}".format(snps=n_snps,var=var_str)

    ss = partial(simulate_helper,genome_command,phenosim_command,seed_arr)
    pool.map(ss,range(n_sim))


@click.group()
def cli3():
    pass

@cli3.command()
@click.option('-e', '--epochs','epochs',default=100,type=int)
@click.option('-s', '--samples','n_samples',required=True,type=int)
@click.option('-S', '--SNPs','n_snps',required=True,type=int)
@click.option('-b', '--batch','batch',default=20,type=int)
@click.option('-r', '--ratio','ratio',default=0.8,type=float)
@click.option('-w', '--width','width',default=15,type=int)
@click.option('--path','path',required=True,type=str)
@click.option('--verbose/;','debug',default=False)
@click.option('--deterministic/;','deterministic',default=False)

def train(epochs,n_samples,n_snps,batch,ratio,width,path,deterministic,debug):
    """Train"""

    from net import Net
    from dataset  import DatasetPhenosim,DatasetPhenosim
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generator = torch.Generator()

    if deterministic:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        generator.manual_seed(0)

        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
        # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        # if not torch.backends.cudnn.deterministic: 
            # exit(1)


    full_dataset = DatasetPhenosim(n_samples,n_snps,path)


    train_size = int(ratio * len(full_dataset))
    test_size =  len(full_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size],generator)

    dataloader_train = DataLoader(train_dataset, batch_size=batch,shuffle=True, num_workers=0)
    dataloader_test =  DataLoader(test_dataset,  batch_size=batch,shuffle=True, num_workers=0)


    net = Net(n_snps,n_samples,batch,width).to(device)

    if debug:
        print(torch.version.cuda)
        print(torch.backends.cudnn.version())
        print(net)
        print(device)
        print("SNPs: {0} , samples: {1}, batch: {2} , width : {3}".format(n_snps,n_samples,batch,width))


    # criterion = nn.SoftMarginLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    criterion_test = F.mse_loss

    # criterion = nn.HingeEmbeddingLoss()

    # m = nn.Sigmoid()

    optimizer = optim.SGD(net.parameters(), lr=1e-1,momentum=0.9,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)

    min_loss = np.Infinity

    if debug:
        max_accuracy = 0
    
    if not Path("models").is_dir():
        Path.mkdir("models")

    # clip_grad_norm = np.inf 
    clip_grad_norm = 5 
    e = 0

    plot_average = True 

    if plot_average:
        fig,ax = plt.subplots(2,2)

        ax[0,0].set_xlabel("False Positives")
        ax[0,1].set_xlabel("False Negatives")
        ax[1,0].set_xlabel("True Positives")
        ax[1,1].set_xlabel("True Negatives")

    loss_arr = torch.zeros(epochs)

    FP_arr = torch.zeros(epochs,dtype=int)
    FN_arr = torch.zeros(epochs,dtype=int)
    TP_arr = torch.zeros(epochs,dtype=int)
    TN_arr = torch.zeros(epochs,dtype=int)


    for e in range(epochs):
        total_loss = 0
        times = 0

        n_FN = 0
        n_FP = 0
        n_TP = 0
        n_TN = 0
    
        avr_FN = torch.zeros((width,n_samples // width)).to(device)
        avr_FP = torch.zeros((width,n_samples // width)).to(device)
        avr_TN = torch.zeros((width,n_samples // width)).to(device)
        avr_TP = torch.zeros((width,n_samples // width)).to(device)
    

        net.eval()
        if debug:
            full_dataset.eval_()

        with torch.no_grad():
            for i,data in enumerate(dataloader_test):

                inputs = data['input'].float().to(device)
                pred = data['output'].float().to(device)

                outputs = net(inputs)

                loss = criterion_test(outputs,pred)
                total_loss += loss.item()
                times += 1

                if debug:
                    x = inputs.detach().clone() 
                    tmp_n_snps = x.shape[1]
                    x = x.view(batch,tmp_n_snps,width,-1)
                    x = x.view(batch*tmp_n_snps,width,-1)
                    x = torch.unsqueeze(x,1)


                    pred_copy = pred.detach().clone().flatten()
                    outputs_copy = outputs.detach().clone().flatten()

                    # outputs_copy = torch.sign(outputs_copy)
                    ind_tmp = (outputs_copy >= 0).nonzero()
                    ind_tmp_2 = (outputs_copy < 0).nonzero()

                    if deterministic:
                        outputs_copy[ind_tmp] = torch.ones(outputs_copy[ind_tmp].shape).to(device)  # cuda problem when CUBLAS_WORKSPACE_CONFIG=":16:8"
                        outputs_copy[ind_tmp_2] = - torch.ones(outputs_copy[ind_tmp_2].shape).to(device) 
                    else:
                        outputs_copy[ind_tmp] = 1.
                        outputs_copy[ind_tmp_2] = -1.


                    false_ind = torch.where(outputs_copy != pred_copy)
                    false = pred_copy[false_ind]
                    false_ind = false_ind[0]
                    
                    true_ind = torch.where(outputs_copy == pred_copy)
                    true = pred_copy[true_ind]
                    true_ind = true_ind[0]

                    false_positives = false_ind[torch.where(false == 1)]  
                    false_negatives = false_ind[torch.where(false == -1)] 
                    true_positives = true_ind[torch.where(true == 1)]
                    true_negatives = true_ind[torch.where(true == -1)]

                    if len(false_positives) != 0 :
                        avr_FP += torch.mean(x[false_positives,0,:,:],axis=0)
                        n_FP += len(false_positives)
                    if len(false_negatives) != 0:
                        avr_FN += torch.mean(x[false_negatives,0,:,:],axis=0)
                        n_FN += len(false_negatives)
                    if len(true_positives) != 0 :
                        avr_TP += torch.mean(x[true_positives,0,:,:],axis=0)
                        n_TP += len(true_positives)
                    if len(true_negatives) != 0 :
                        avr_TN += torch.mean(x[true_negatives,0,:,:],axis=0)
                        n_TN += len(true_negatives)

        if debug:

            loss_arr[e] = total_loss / times

            TP_arr[e] = n_TP
            FP_arr[e] = n_FP
            TN_arr[e] = n_TN
            FN_arr[e] = n_FN

            print("FP: {0}, FN : {1}, TP: {2}, TN: {3}".format(n_FP,n_FN,n_TP,n_TN))
            accuracy = 100.0*(n_TP+n_TN)/(n_FP+n_FN+n_TN+n_TP)
            recall = 100.0*(n_TP/(n_TP+n_FN)) if n_TP + n_FN >0 else 0
            precision = 100.0*(n_TP/(n_TP+n_FP)) if n_TP + n_FP >0 else 0
            F1 = 100.0*(2.0*n_TP/(2.0*n_TP+n_FP+n_FN)) if n_TP + n_FP + n_FN > 0 else 0 
            print("Epoch : {0}, Accuracy: {1:.2f}, Recall: {2:.2f}, Precision: {3:.2f} F1: {4:.2f}, loss : {5:.5f}".format(
                e,
                accuracy,
                recall,
                precision,
                F1,
                loss_arr[e] 
            ))


            if max_accuracy < accuracy:
                accuracy =max_accuracy
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, "models/net-accuracy.pt")

            if plot_average and e % 20 == 0:
                ax[0,0].clear()
                ax[0,1].clear()
                ax[1,0].clear()
                ax[1,1].clear()

                if n_FP > 0:
                    ax[0,0].matshow(avr_FP.cpu())
                if n_FN > 0:
                    ax[0,1].matshow(avr_FN.cpu())
                if n_TP > 0:
                    ax[1,0].matshow(avr_TP.cpu())
                if n_TN > 0:
                    ax[1,1].matshow(avr_TN.cpu())

                fig.canvas.draw()
                fig.savefig('results/matrix.png'.format(e=e))

     
        else:
            print("Epoch: {0}".format(e))

        if total_loss/times < min_loss:
            min_loss  = total_loss/times
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, "models/net.pt")
    
        full_dataset.train()
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


        if e % 100 == 0:
            scheduler.step()


    if debug:
        data = {'TP':TP_arr,'TN':TN_arr,'FP':FP_arr,'FN':FN_arr,'loss':loss_arr}
        df_stats = pd.DataFrame(data)
        df_stats.to_csv('results/stats-r{ratio}.csv'.format(ratio=n_snps))

cli = click.CommandCollection(sources=[cli1, cli2,cli3])

if __name__ == '__main__':
    cli()

