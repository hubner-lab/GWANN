import click # command line 
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
# torch.backends.cudnn.benchmark = False
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import random
import time
import functools
import operator
from functools import partial
from typing import List, Optional, Tuple
from sklearn.manifold import MDS
# import sgkit as sg
# from sgkit.io.vcf import partition_into_regions, vcf_to_zarr
import allel
from pathlib import Path
import re
from glob import glob
import subprocess
import shlex
import multiprocessing
import csv 
import json
import resource
from const import * 

def json_update(key: str ,param:int) -> None:
    """
    Update the JSON file with a new value for the specified key.

    Parameters:
    key (str): The key in the JSON file to update.
    param (int): The new value to set for the specified key.
    """
    tmp_dict = json.loads(JSON_FILE.read_text())
    tmp_dict[key] = param 
    JSON_FILE.write_text(json.dumps(tmp_dict))


def seed(pop: int) -> np.ndarray:
    """
    Generates a seeded population array with random integers added to each element.

    Parameters:
    pop (int): The size of the population array to generate.

    Returns:
    np.ndarray: An array of integers where each element is a unique integer from 0 to pop-1 
                with a random integer between LOWER_BOUND=1 and UPPER_BOUND=1000000 added to it.
    """

    return np.array(list(range(pop))) + np.random.randint(LOWER_BOUND, UPPER_BOUND)



def generate_commands(pop: int, subpop: int, n_samples: int, maf: float, miss: float) -> Tuple[List[str], str]:
    """
    Generate genome and phenosim commands based on given parameters.

    Parameters:
        pop (int): Population size.
        subpop (int): Number of subpopulations.
        n_samples (int): Total number of samples.
        maf (float): Minor allele frequency.
        miss (float): Missing data rate.

    Returns:
        tuple: A tuple containing the genome command (list) and the phenosim command (str).
    """
    # Calculate sample distribution
    tmp = (n_samples // subpop)
    tmp2 = n_samples - tmp * subpop
    samples_str = " ".join([str(tmp)] * (subpop - 1) + [str(tmp + tmp2)])

    # Construct genome command
    genome_command = shlex.split(f"{GENOME_EXE} -s {pop} -pop {subpop} {samples_str} -seed")

    # Construct phenosim command
    phenosim_command = (
        f"python2 simulation/phenosim/phenosim.py -i G "
        f"-f {SIM_PATH}/genome{{0}}.txt --outfile {SIM_PATH}/{{0}} "
        f"--maf_r {maf},1.0 --maf_c {maf} --miss {miss}"
    )

    return genome_command, phenosim_command, samples_str


def json_get(key:str):
    """get the value of the key in the data.json file"""
    if current_dict is None:
        tmp_dict = json.loads(JSON_FILE.read_text())
        return tmp_dict[key] 
    return current_dict[key]



def configure_deterministic_behavior(deterministic: bool, seed: Optional[int] = 0) -> torch.Generator:
    """
    Configures deterministic behavior for PyTorch, NumPy, and Python's random module.

    Parameters:
        deterministic (bool): Whether to enable deterministic behavior.
        seed (Optional[int]): The seed value to use for random number generators. Defaults to 0.

    Returns:
        torch.Generator: A PyTorch random number generator with the specified seed (if deterministic).
    """
    generator = torch.Generator()

    if deterministic:
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        generator.manual_seed(seed)

        # Configure PyTorch for deterministic algorithms
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    return generator


def split_and_create_dataloaders(
    full_dataset: Dataset, 
    ratio: float, 
    batch_size: int, 
    generator: torch.Generator
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits a dataset into training and testing datasets and creates corresponding DataLoaders.

    Parameters:
        full_dataset (Dataset): The complete dataset to split.
        ratio (float): The ratio of the dataset to use for training (0 < ratio < 1).
        batch_size (int): The batch size for the DataLoaders.
        generator (torch.Generator): A PyTorch generator for reproducibility.

    Returns:
        Tuple[DataLoader, DataLoader]: The training and testing DataLoaders.
    """
    # Calculate the sizes of the train and test datasets
    train_size = int(ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Split the dataset into train and test datasets
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size], generator
    )

    # Create DataLoaders for the train and test datasets
    dataloader_train = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    dataloader_test = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return dataloader_train, dataloader_test


def debug_info(
    debug: bool, 
    net: torch.nn.Module, 
    device: torch.device, 
    n_snps: int, 
    n_samples: int, 
    batch: int, 
    width: int
) -> None:
    """
    Print debugging information if debugging is enabled.

    Parameters:
        debug (bool): Whether to enable debug printing.
        torch (torch): The PyTorch module for accessing version info.
        net (torch.nn.Module): The neural network model.
        device (torch.device): The device being used (CPU/GPU).
        n_snps (int): Number of simulated SNP sites.
        n_samples (int): Number of simulated sample genomes.
        batch (int): Batch size.
        width (int): Matrix width.
    """
    if debug:
        print("CUDA version : {0}".format(torch.version.cuda))
        print("CUDNN version : {0}".format(torch.backends.cudnn.version()))
        print(net)
        print(device)
        print("Simulated SNP sites: {0}, Simulated sample genomes: {1}".format(n_snps, n_samples))
        print("Batch size: {0}, Matrix width: {1}".format(batch, width))


def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

@click.group()
def cli1():
    pass

@cli1.command()
@click.option('-v', '--vcf','vcf',required=True,help='path to the VCF file')
@click.option('-p', '--pheno','pheno_path',required=True,help='path to the phenotype file (comma seperated csv file)')
@click.option('-t', '--trait','trait',required=True,help='name of the trait (header in the phenotype file)')

@click.option('--model','model',default="models/net.pt",help="path to the network model generated in the training step")
@click.option('--output','output_path',default="results/GWAS",help="prefix of output plot and causative SNPs indexes in the VCF")
@click.option('--cpu/;','cpu',default=False,required=False,help="force on cpu")

# @click.option('-s', '--samples','n_samples',default=250,type=int)
# @click.option('-w', '--width','width',default=10,type=int)

def run(vcf,pheno_path,trait,model,output_path,cpu):
    """Run on real data"""

    from net import Net

    width = json_get('width')
    n_samples = json_get('samples')

    if not Path("vcf_data").is_dir():
        Path("vcf_data").mkdir(parents=True,exist_ok=True)

    npz_loc = "vcf_data/{0}.npz".format(Path(vcf).stem)

    print('save vcf')
    if not Path(npz_loc).is_file():
        allel.vcf_to_npz(vcf, npz_loc, fields='*', overwrite=True,chunk_length=8192,buffer_size=8192)

    print('reload vcf')
    callset = np.load(npz_loc,allow_pickle=True)

    print('parse vcf')
    vcf = callset['calldata/GT']
    vcf_samples = callset['samples']
    chrom = callset['variants/CHROM']
    tmp_vcf = (vcf[:,:,0] + vcf[:,:,1]) / 2
    tmp_vcf[np.where(tmp_vcf == 0.5)] = 0 
    #print(tmp_vcf.shape)

    device = 'cpu' if cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    final_vcf = torch.from_numpy(tmp_vcf).float()  # .to(device)

    embedding = MDS(n_components=1,random_state=0)
    mds_data = embedding.fit_transform(tmp_vcf.T)

    pop = torch.from_numpy(mds_data).float().to(device)
    pad_pop = torch.zeros((n_samples - pop.shape[0],1)).float().to(device) 
    pop_padded = torch.cat((pad_pop,pop),0)
    # print(pop_padded)

    #n_snps = 1000 
    n_snps = final_vcf.shape[0] 
    print('final_vcf', final_vcf.shape)
    print('n_snps', n_snps)

    if not Path(pheno_path).is_file():
        print("Invalid file pheno")
        exit(1)

    pheno = pd.read_csv(pheno_path,index_col=None,sep=',')
    if not 'sample' in pheno.keys():
        raise click.ClickException('sample field missing in phenotype file')
    if not trait in pheno.keys():
        raise click.ClickException('trait field missing in phenotype file') 

    _,index_samples,index_samples_pheno = np.intersect1d(vcf_samples,pheno["sample"],return_indices=True)

    # df_ss = pd.DataFrame(ss,columns=['sample'])
    # pheno = pd.concat([df_ss,pheno],axis=0)
    final_vcf = final_vcf[:,index_samples]
    pheno = pheno.loc[index_samples_pheno].reset_index()
    #assert (pheno["sample"] == vcf_samples[index_samples]).all()
    #assert (vcf_samples == pheno['sample']).all()

    pheno_sorted = pheno.sort_values(by=[trait,"sample"],na_position='first')

    sorted_axes = np.array(pheno_sorted.index.values)
    sorted_vcf = final_vcf[:,sorted_axes]

    df_chrom = pd.DataFrame(chrom)
    chrom_labels = df_chrom[0].unique().tolist()

    input_s = torch.split(final_vcf,n_snps)
    output = torch.zeros((final_vcf.shape[0])).float().to(device)
        
    net = Net(n_snps,n_samples,1,width).to(device)
    net.load_state_dict(torch.load(model,torch.device(device))['model_state_dict'])

    net.eval()
    with torch.no_grad():
        for j in range(len(input_s)):
            input_tmp = input_s[j]
            if n_snps - input_tmp.shape[0] > 0:
                input_tmp = sorted_vcf[-n_snps:]

            pad_samples = n_samples - input_tmp.shape[1]
            pad_2 = torch.zeros((n_snps,pad_samples), device=device).float()
            input = torch.cat((pad_2,input_tmp.to(device)),1)
            input = torch.unsqueeze(input,0)

            outputs = net(input,pop_padded)

            output[j*n_snps:j*n_snps + input_s[j].shape[0]] = outputs[:,-input_s[j].shape[0]:]


    output = output.cpu()

    plt.clf()
    fig,ax = plt.subplots(1)

    current = 0

    chrom_labels.sort(key=num_sort)
    chr_loc = []


    min = 0
    
    # avr = torch.mean(output)
    # output -= avr

    print(100 * (torch.count_nonzero(output > min)/output.shape[0]).item())

    index_tmp = (output > min).nonzero().flatten().numpy()
    value_tmp = output.detach().clone()[index_tmp].flatten().numpy()

    df = pd.DataFrame({"value":value_tmp},index=index_tmp)
    df.to_csv('{0}.csv'.format(output_path))

    output[(output <= min).nonzero()] = min

    color = ""

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
    fig.savefig('{0}.png'.format(output_path))



def simulate_helper(genome_command,phenosim_command,seed,i):
    out_file = open(f'{SIM_GENOM_PATH}{i}.txt','w')
    try:
        subprocess.call(genome_command + ["{0}".format(seed[i])],stdout=out_file)
        out_file.close()
    except Exception as e:
        raise Exception(f'genome failed for {i}; with {e}')

    try:
        phenosim_command = shlex.split(phenosim_command.format(i))
        subprocess.call(phenosim_command,stdout=subprocess.DEVNULL)
    except Exception as e:
        raise Exception('phenosim failed for {i}; with {e}'.format(i=i, e=e))


@click.group()
def cli2():
    pass

@cli2.command()
@click.option('-p', '--number-of-snps','pop',required=True,type=int,help="number of SNPs in each simulation")
@click.option('-P', '--number-of-subpopulations','subpop',required=True,type=int,help="number of expected subpopulations")
@click.option('-s', '--samples','n_samples',required=True,type=int,help="number of individuals")
@click.option('-n', '--number-of-simulation','n_sim',required=True,type=int,help="number of populations to be simulated")
@click.option('-S', '--causal_snps','n_snps',default=1,type=int,help="number of causal SNPs expected per number of SNP-sites")
@click.option('-m', '--maf','maf',default=0.05,type=float,help="minor allele frequency")
@click.option('--miss','miss',default=0.03,type=float,help="proportion of missing data")
@click.option('--equal_variance/;','equal',default=False,help="set this if equal variance is expected among SNPs (ignore for single SNP)")
@click.option('--verbose/;','debug',default=False,help="increase verbosity")





def simulate(pop,subpop,n_samples,n_sim,n_snps,maf,miss,equal,debug):
    """Simulate training data"""

    # assert width < n_samples ,"image width is bigger than the number of samples"
    # assert n_samples % width == 0,"image width does not divide the number of samples"
   
    json_update(SAMPLES,n_samples)
           
    seed_arr = seed(pop)

    np.random.shuffle(seed_arr)

    cpus = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(cpus)

    genome_command, phenosim_command, samples_str = generate_commands(pop,subpop,n_samples,maf,miss)

    if n_snps > 1:

        variance = np.ones(n_snps)
        if equal:
            variance = (variance / n_snps) * 0.99
        else:
            variance = np.random.dirichlet(variance,size=1) * 0.99

        var_str = np.array2string(variance,precision=5,separator=',',formatter={'float_kind':lambda x: "%.5f" % x})
        var_str = re.sub("\[|\s*|\]","",var_str)
        phenosim_command += " -n {snps} -v {var}".format(snps=n_snps,var=var_str)
    
    try:
        if debug:
            print(f'Mapping using {cpus} CPUs')
            print(f'Output directory: {SIM_PATH}')
            print(f'Genome parameters: population={pop}, subpopulations={subpop}, samples={samples_str}')
            print(f'Phenosim parameters: maf_c={maf}, maf_r={maf}, miss={miss}')
        ss = partial(simulate_helper,genome_command,phenosim_command,seed_arr)
        pool.map(ss,range(n_sim))
    except OSError:
        if not os.path.exists(GENOME_EXE):
            raise click.ClickException('genome simulator not found') 
    if debug:
        genome_files = glob(f'{SIM_PATH}/genome*.txt')
        emma_files = glob(f'{SIM_PATH}/*.causal')
        print(f'n simulations: expected={n_sim}, genomes={len(genome_files)}, phenotype={len(emma_files)}')

@click.group()
def cli3():
    pass

@cli3.command()
@click.option('-e', '--epochs','epochs',default=100,type=int,help="number of training iterations")
# @click.option('-s', '--samples','n_samples',required=True,type=int,)
@click.option('-S', '--SNPs','n_snps',required=True,type=int,help="number of SNP sites to be randomly sampled per batch")
@click.option('-b', '--batch','batch',default=20,type=int,help="batch size") 
@click.option('-r', '--ratio','ratio',default=0.8,type=float,help="train / eval ratio")
@click.option('-w', '--width','width',default=15,type=int,help="image width must be a divisor of the number of individuals")
@click.option('--path','sim_path',required=True,type=str,help="path to the simulated data")
@click.option('--verbose/;','debug',default=False,help="increase verbosity")
@click.option('--deterministic/;','deterministic',default=False,help="set for reproducibility") 
@click.option('--cpu/;','cpu',default=False,required=False,help="force training on cpu")

def train(epochs,n_snps,batch,ratio,width,sim_path,deterministic,debug,cpu):
    """Train the model on the simulated data"""

    from net import Net
    from dataset import DatasetPhenosim
    
    json_update(WIDTH,width)
    n_samples = json_get(SAMPLES)

    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    device = 'cpu' if cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generator = configure_deterministic_behavior(deterministic)

    full_dataset = DatasetPhenosim(n_samples,n_snps,sim_path)

    if debug:
        print(full_dataset.shapes())

    dataloader_train, dataloader_test = split_and_create_dataloaders(full_dataset, ratio, batch, generator)
    
    net = Net(n_snps,n_samples,batch,width).to(device)

    debug_info(debug,net,device,n_snps,n_samples,batch,width)


    criterion = nn.MSELoss()
    criterion_test = F.mse_loss

    optimizer = optim.SGD(net.parameters(), lr=1e-2,momentum=0.9,weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.8)

    min_loss = np.Infinity
    max_accuracy = 0
    
    if not Path("models").is_dir():
        Path("models").mkdir(parents=True,exist_ok=True)

    clip_grad_norm = 5 
    e = 0

    plot_average = False 

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
        # if debug:
            # full_dataset.eval_()

        with torch.no_grad():
            for i,data in enumerate(dataloader_test):

                inputs = data['input'].float().to(device)
                pred = data['output'].float().to(device)
                pop = data['population'].float().to(device)

                outputs = net(inputs,pop)

                loss = criterion_test(outputs,pred)
                total_loss += loss.item()
                times += 1

                if debug:
                    x = inputs.detach().clone() 
                    tmp_batch,tmp_n_snps,_ = x.shape

                    if tmp_batch != batch:
                        continue 

                    x = x.view(batch,tmp_n_snps,width,-1)
                    x = x.view(batch*tmp_n_snps,width,-1)
                    x = torch.unsqueeze(x,1)

                    pop_copy = pop.detach().clone() 

                    pop_copy = pop_copy.view(tmp_batch,n_samples)
                    pop_copy = pop_copy.view(tmp_batch,width,-1)
                    pop_copy = torch.unsqueeze(pop_copy,1)

                    # plt.matshow(pop_copy[0,0,:,:].cpu())
                    # plt.savefig("results/test.png")
                    # print(torch.sigmoid(pop_copy[0,0,:,:]))
                    # exit(0)

                    pred_copy = pred.detach().clone().flatten()
                    outputs_copy = outputs.detach().clone().flatten() 


                    min = 0

                    ind_tmp = (outputs_copy >= min).nonzero()
                    ind_tmp_2 = (outputs_copy < min).nonzero()

                    pred_ind_tmp = (pred_copy >= min).nonzero()
                    pred_ind_tmp_2 = (pred_copy < min).nonzero()

                    if deterministic: # cuda problem when CUBLAS_WORKSPACE_CONFIG=":16:8"
                        outputs_copy[ind_tmp] = torch.ones(outputs_copy[ind_tmp].shape).to(device)  
                        outputs_copy[ind_tmp_2] = - torch.ones(outputs_copy[ind_tmp_2].shape).to(device) 
                        pred_copy[pred_ind_tmp] = torch.ones(pred_copy[pred_ind_tmp].shape).to(device)  
                        pred_copy[pred_ind_tmp_2] = -torch.ones(pred_copy[pred_ind_tmp_2].shape).to(device) 
                    else:
                        outputs_copy[ind_tmp] = 1.
                        outputs_copy[ind_tmp_2] = -1.
                        pred_copy[pred_ind_tmp] = 1
                        pred_copy[pred_ind_tmp_2] = -1.


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

            # print("FP: {0}, FN : {1}, TP: {2}, TN: {3}".format(n_FP,n_FN,n_TP,n_TN))
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
                accuracy = max_accuracy
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
                fig.savefig('{results_path}/matrix.png'.format(results_path=RESULTS_PATH,e=e))

     
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
            pop = data['population'].float().to(device)

            outputs = net(inputs,pop)
            
            loss = criterion(outputs, pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad_norm)
            optimizer.step()    

        if e % 100 == 0:
            scheduler.step()


    if debug:
        data = {'TP':TP_arr,'TN':TN_arr,'FP':FP_arr,'FN':FN_arr,'loss':loss_arr}
        df_stats = pd.DataFrame(data)
        df_stats.to_csv('{results_path}/stats-r{ratio}.csv'.format(results_path=RESULTS_PATH,ratio=n_snps))

cli = click.CommandCollection(sources=[cli2,cli3,cli1])

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 // 2, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

if __name__ == '__main__':
    # memory_limit()
    try:
        cli()
    except MemoryError:
        exit(1)

