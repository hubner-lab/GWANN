from typing import Tuple, Optional
from utilities import json_update, json_get
from const import WIDTH, SAMPLES, RESULTS_PATH
import os 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from net import Net
from dataset import DatasetPhenosim
import random
class Train:
    def __init__(self, epochs,n_snps,batch,ratio,width,sim_path,deterministic,debug,cpu):
        self.epochs = epochs
        self.n_snps = n_snps
        self.batch = batch
        self.ratio = ratio
        self.width = width
        self.sim_path = sim_path
        self.deterministic = deterministic
        self.debug = debug
        self.cpu = cpu

    def split_and_create_dataloaders(self,
            full_dataset: Dataset, 
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
        train_size = int(self.ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # Split the dataset into train and test datasets
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator
        )

        # Create DataLoaders for the train and test datasets
        dataloader_train = DataLoader(
            train_dataset, batch_size=self.batch, shuffle=True, num_workers=0
        )
        dataloader_test = DataLoader(
            test_dataset, batch_size=self.batch, shuffle=True, num_workers=0
        )

        return dataloader_train, dataloader_test


    def configure_deterministic_behavior(self, seed: Optional[int] = 0) -> torch.Generator:
        """
        Configures deterministic behavior for PyTorch, NumPy, and Python's random module.

        Parameters:
            deterministic (bool): Whether to enable deterministic behavior.
            seed (Optional[int]): The seed value to use for random number generators. Defaults to 0.

        Returns:
            torch.Generator: A PyTorch random number generator with the specified seed (if deterministic).
        """
        generator = torch.Generator()

        if self.deterministic:
            # Set seeds for reproducibility
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            generator.manual_seed(seed)

            # Configure PyTorch for deterministic algorithms
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        return generator

    def train(self):
        """Train the model on the simulated data"""

        
        json_update(WIDTH,self.width)
        n_samples = json_get(SAMPLES)

        if not os.path.exists(RESULTS_PATH):
            os.mkdir(RESULTS_PATH)

        device = 'cpu' if self.cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        generator = self.configure_deterministic_behavior()

        full_dataset = DatasetPhenosim(n_samples,self.n_snps,self.sim_path)

        if self.debug:
            print(full_dataset.shapes())

        dataloader_train, dataloader_test = self.split_and_create_dataloaders(full_dataset, generator)
        
        net = Net(self.n_snps,n_samples,self.batch,self.width).to(device)

        self.debug_info(n_samples,net,device)


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

        loss_arr = torch.zeros(self.epochs)

        FP_arr = torch.zeros(self.epochs,dtype=int)
        FN_arr = torch.zeros(self.epochs,dtype=int)
        TP_arr = torch.zeros(self.epochs,dtype=int)
        TN_arr = torch.zeros(self.epochs,dtype=int)


        for e in range(self.epochs):
            total_loss = 0
            times = 0

            n_FN = 0
            n_FP = 0
            n_TP = 0
            n_TN = 0
        
            avr_FN = torch.zeros((self.width,n_samples // self.width)).to(device)
            avr_FP = torch.zeros((self.width,n_samples // self.width)).to(device)
            avr_TN = torch.zeros((self.width,n_samples // self.width)).to(device)
            avr_TP = torch.zeros((self.width,n_samples // self.width)).to(device)
        

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

                    if self.debug:
                        x = inputs.detach().clone() 
                        tmp_batch,tmp_n_snps,_ = x.shape

                        if tmp_batch != self.batch:
                            continue 

                        x = x.view(self.batch,tmp_n_snps,self.width,-1)
                        x = x.view(self.batch*tmp_n_snps,self.width,-1)
                        x = torch.unsqueeze(x,1)

                        pop_copy = pop.detach().clone() 

                        pop_copy = pop_copy.view(tmp_batch,n_samples)
                        pop_copy = pop_copy.view(tmp_batch,self.width,-1)
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

                        if self.deterministic: # cuda problem when CUBLAS_WORKSPACE_CONFIG=":16:8"
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

            if self.debug:

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


        if self.debug:
            data = {'TP':TP_arr,'TN':TN_arr,'FP':FP_arr,'FN':FN_arr,'loss':loss_arr}
            df_stats = pd.DataFrame(data)
            df_stats.to_csv('{results_path}/stats-r{ratio}.csv'.format(results_path=RESULTS_PATH,ratio=self.n_snps))

            
    def debug_info(self,
        n_samples: int,
        net: torch.nn.Module, 
        device: torch.device, 
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
        if self.debug:
            print("CUDA version : {0}".format(torch.version.cuda))
            print("CUDNN version : {0}".format(torch.backends.cudnn.version()))
            print(net)
            print(device)
            print("Simulated SNP sites: {0}, Simulated sample genomes: {1}".format(self.n_snps, n_samples))
            print("Batch size: {0}, Matrix width: {1}".format(self.batch, self.width))



if __name__ == "__main__":
    train = Train(1000,1000,32,0.8,10,"simulations/simulated_data.csv",False,True,False)
    train.train()