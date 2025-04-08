from typing import Tuple, Optional
from utilities import json_update, json_get
from const import*
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
from Nimrod.dataset import DatasetPhenosim
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

    def __split_and_create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
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
        train_size = int(self.ratio * len(self.full_dataset))
        test_size = len(self.full_dataset) - train_size

        # Split the dataset into train and test datasets
        train_dataset, test_dataset = torch.utils.data.random_split(
            self.full_dataset, [train_size, test_size], self.generator
        )

        # Create DataLoaders for the train and test datasets
        dataloader_train = DataLoader(
            train_dataset, batch_size=self.batch, shuffle=True, num_workers=0
        )
        dataloader_test = DataLoader(
            test_dataset, batch_size=self.batch, shuffle=True, num_workers=0
        )

        return dataloader_train, dataloader_test


    def __configure_deterministic_behavior(self) -> None:
        """
        Configures deterministic behavior for PyTorch, NumPy, and Python's random module.

        MODE=True (bool): Whether to enable deterministic behavior.
        SEED=0 (Optional[int]): The seed value to use for random number generators.

        Returns:
            torch.Generator: A PyTorch random number generator with the specified seed (if deterministic).
        """
        self.generator = torch.Generator()

        if self.deterministic:
            # Set seeds for reproducibility
            torch.manual_seed(SEED)
            random.seed(SEED)
            np.random.seed(SEED)
            self.generator.manual_seed(SEED)
            # Configure PyTorch for deterministic algorithms
            torch.use_deterministic_algorithms(MODE)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
        
    
    def _set_criterion(self)->None:
        return nn.MSELoss(),  F.mse_loss
    
    def _set_results_path(self) -> None:
        if not os.path.exists(RESULTS_PATH):
            os.mkdir(RESULTS_PATH)
    
    def __init_data_for_training(self)->None:
             
        json_update(WIDTH,self.width)
        
        self.n_samples = json_get(SAMPLES)

        self._set_results_path()

        self.device = CPU if self.cpu else torch.device(CUDA if torch.cuda.is_available() else CPU)

        self.__configure_deterministic_behavior()

        self.full_dataset = DatasetPhenosim(self.n_samples,self.n_snps,self.sim_path)

        if self.debug:
            print(self.full_dataset.shapes())

        self.dataloader_train, self.dataloader_test = self.__split_and_create_dataloaders()
        
        self.net = Net(self.n_snps,self.n_samples,self.batch,self.width).to(self.device)

        self.__debug_info()

        self.criterion, self.criterion_test = self._set_criterion()
                                                            
        self.optimizer = optim.SGD(self.net.parameters(), lr=LR,momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=GAMMA)


    def __init_initialize_metrics(self)-> None:
        """
        Initializes metric arrays for tracking performance during training or evaluation.

        This method creates the following arrays as attributes of the class:
        - `self.loss_arr`: A tensor to store loss values for each epoch.
        - `self.FP_arr`: A tensor to store the count of False Positives (FP) for each epoch.
        - `self.FN_arr`: A tensor to store the count of False Negatives (FN) for each epoch.
        - `self.TP_arr`: A tensor to store the count of True Positives (TP) for each epoch.
        - `self.TN_arr`: A tensor to store the count of True Negatives (TN) for each epoch.

        Attributes:
            loss_arr (torch.Tensor): Zero-initialized tensor for loss values.
            FP_arr (torch.Tensor): Zero-initialized tensor for False Positives.
            FN_arr (torch.Tensor): Zero-initialized tensor for False Negatives.
            TP_arr (torch.Tensor): Zero-initialized tensor for True Positives.
            TN_arr (torch.Tensor): Zero-initialized tensor for True Negatives.

        Returns:
            None
        """
        self.loss_arr = torch.zeros(self.epochs)
        self.FP_arr = torch.zeros(self.epochs, dtype=torch.int32)
        self.FN_arr = torch.zeros(self.epochs, dtype=torch.int32)
        self.TP_arr = torch.zeros(self.epochs, dtype=torch.int32)
        self.TN_arr = torch.zeros(self.epochs, dtype=torch.int32)

    def __update_avg_metrics(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initializes and returns a tuple of four tensors for average metrics (FN, FP, TN, TP).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            A tuple containing initialized tensors for FN, FP, TN, and TP.
        """
        return tuple(torch.zeros((self.width, self.n_samples // self.width)).to(self.device) for _ in range(TOTAL_METRICS))

    def __reset_metrics(self):
        """
        Resets tracking variables for loss and metrics to their initial values.

        Returns:
            Tuple[float, int, int, int, int, int]: 
            A tuple containing initial values for total_loss, times, n_FN, n_FP, n_TP, and n_TN.
        """
        return 0.0, 0, 0, 0, 0, 0

    def _prepare_batch(self, data):
        return (data[INPUT].float().to(self.device), 
                data[OUTPUT].float().to(self.device), 
                data[POPULATION].float().to(self.device))
    
    def _plot1(self):
        """
        Initializes and returns a matplotlib figure and axes for plotting average matrices.

        This function creates a 2x2 subplot grid with labels for False Positives, 
        False Negatives, True Positives, and True Negatives. It is used for visualizing 
        the average metrics during training if the `PLOT_AVERAGE` flag is enabled.

        Returns:
            tuple: A tuple containing:
                - fig (matplotlib.figure.Figure): The matplotlib figure object.
                - ax (numpy.ndarray): A 2D array of AxesSubplot objects for the 2x2 grid.
        """
        
        fig, ax = plt.subplots(PLOT_SIZE, PLOT_SIZE)
        ax[0, 0].set_xlabel("False Positives")
        ax[0, 1].set_xlabel("False Negatives")
        ax[1, 0].set_xlabel("True Positives")
        ax[1, 1].set_xlabel("True Negatives")
        return fig, ax
        
    def _plot_average_matrices(self, e, fig, ax, avr_FP, avr_FN, avr_TP, avr_TN, n_FP, n_FN, n_TP, n_TN):
        """
        Plots and saves average matrices for False Positives, False Negatives, 
        True Positives, and True Negatives during training.

        Parameters:
            e (int): Current epoch number.
            fig (matplotlib.figure.Figure): Figure object for plotting.
            ax (numpy.ndarray): Array of AxesSubplot objects for plotting matrices.
            avr_FP, avr_FN, avr_TP, avr_TN (torch.Tensor): Tensors for average metrics.
            n_FP, n_FN, n_TP, n_TN (int): Counters for each metric.
            results_path (str): Path to save the output plot.
        """

        # Clear axes
        ax[0, 0].clear()
        ax[0, 1].clear()
        ax[1, 0].clear()
        ax[1, 1].clear()
        # Plot matrices if there are any values
        if n_FP > 0:
            ax[0, 0].matshow(avr_FP.cpu())
        if n_FN > 0:
            ax[0, 1].matshow(avr_FN.cpu())
        if n_TP > 0:
            ax[1, 0].matshow(avr_TP.cpu())
        if n_TN > 0:
            ax[1, 1].matshow(avr_TN.cpu())
        # Update and save the figure
        fig.canvas.draw()
        fig.savefig(f'{RESULTS_PATH}/matrix_{e}.png')



    def _calculate_and_print_metrics(self, e, n_TP, n_TN, n_FP, n_FN, loss):
        """
        Calculate and print the performance metrics for a given epoch.

        Parameters:
        e (int): Epoch number.
        n_TP (int): Number of true positives.
        n_TN (int): Number of true negatives.
        n_FP (int): Number of false positives.
        n_FN (int): Number of false negatives.
        loss (float): Loss value for the epoch.

        Returns:
        dict: Dictionary containing the calculated metrics.
        """
        # Calculate metrics
        accuracy = 100.0*(n_TP+n_TN)/(n_FP+n_FN+n_TN+n_TP) if n_FP+n_FN+n_TN+n_TP > 0 else 0
        recall = 100.0*(n_TP/(n_TP+n_FN)) if n_TP + n_FN >0 else 0
        precision = 100.0*(n_TP/(n_TP+n_FP)) if n_TP + n_FP >0 else 0
        F1 = 100.0*(2.0*n_TP/(2.0*n_TP+n_FP+n_FN)) if n_TP + n_FP + n_FN > 0 else 0 
        # Print metrics
        print("Epoch : {0}, Accuracy: {1:.2f}, Recall: {2:.2f}, Precision: {3:.2f}, F1: {4:.2f}, Loss: {5:.5f}".format(
            e, accuracy, recall, precision, F1, loss
        ))
        
        # Return metrics as a dictionary (optional)
        return accuracy

    def __update_matrices(self, e,total_loss, times, n_TP, n_FP, n_TN, n_FN):
        self.loss_arr[e] = total_loss / times
        self.TP_arr[e] = n_TP
        self.FP_arr[e] = n_FP
        self.TN_arr[e] = n_TN
        self.FN_arr[e] = n_FN
                
    def __debug_info(self, 
                ) -> None:
        """
        Print debugging information if debugging is enabled.

        """
        if self.debug:
            print("CUDA version : {0}".format(torch.version.cuda))
            print("CUDNN version : {0}".format(torch.backends.cudnn.version()))
            print(self.net)
            print(self.device)
            print("Simulated SNP sites: {0}, Simulated sample genomes: {1}".format(self.n_snps, self.n_samples))
            print("Batch size: {0}, Matrix width: {1}".format(self.batch, self.width))



    def _save_best_model(self):
        """
        Save the model if the current accuracy is the highest so far.

        Parameters:
        max_accuracy (float): The best accuracy recorded so far.
        
        Returns:
        None
        """
        torch.save({
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, MODEL_PATH_ACCURACY)


    def __save_min_loss_model(self,total_loss, times):
        """
        Save the model if the current loss is the minimum recorded so far.

        Parameters:
        net (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loss (float): The current average loss.
        min_loss (float): The minimum loss recorded so far.
        save_path (str): Path to save the model.
        
        Returns:
        float: The updated min_loss.
        """
        
        min_loss = total_loss/times
        torch.save({
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
        }, MODEL_PATH_MIN_LOSS)
        print(f"Model saved with minimum loss: {min_loss:.5f}")
        return min_loss

    def _debug_and_compute_metrics(
    self, inputs, pop, pred, outputs, avr_FP, avr_FN, avr_TP, avr_TN, n_FP, n_FN, n_TP, n_TN
):
        """
        Debug and compute false/true positives and negatives metrics.

        Args:
            inputs (torch.Tensor): Input tensor from the batch.
            pop (torch.Tensor): Population tensor.
            pred (torch.Tensor): Predicted values tensor.
            outputs (torch.Tensor): Output tensor.
            avr_FP (torch.Tensor): Average False Positives accumulator.
            avr_FN (torch.Tensor): Average False Negatives accumulator.
            avr_TP (torch.Tensor): Average True Positives accumulator.
            avr_TN (torch.Tensor): Average True Negatives accumulator.
            n_FP (int): Count of False Positives.
            n_FN (int): Count of False Negatives.
            n_TP (int): Count of True Positives.
            n_TN (int): Count of True Negatives.

        Returns:
            Updated metrics: avr_FP, avr_FN, avr_TP, avr_TN, n_FP, n_FN, n_TP, n_TN.
        """
        
        x = inputs.detach().clone() 
        tmp_batch,tmp_n_snps,_ = x.shape

        if tmp_batch != self.batch:
            return avr_FP, avr_FN, avr_TP, avr_TN, n_FP, n_FN, n_TP, n_TN 

        x = x.view(self.batch,tmp_n_snps,self.width,-1)
        x = x.view(self.batch*tmp_n_snps,self.width,-1)
        x = torch.unsqueeze(x,1)

        pop_copy = pop.detach().clone() 

        pop_copy = pop_copy.view(tmp_batch,self.n_samples)
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
            outputs_copy[ind_tmp] = torch.ones(outputs_copy[ind_tmp].shape).to(self.device)  
            outputs_copy[ind_tmp_2] = - torch.ones(outputs_copy[ind_tmp_2].shape).to(self.device) 
            pred_copy[pred_ind_tmp] = torch.ones(pred_copy[pred_ind_tmp].shape).to(self.device)  
            pred_copy[pred_ind_tmp_2] = -torch.ones(pred_copy[pred_ind_tmp_2].shape).to(self.device) 
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

        return avr_FP, avr_FN, avr_TP, avr_TN, n_FP, n_FN, n_TP, n_TN


    
    def train(self):
        """Train the model on the simulated data"""
        self.__init_data_for_training()

        min_loss, max_accuracy = np.Infinity, 0

        if not Path(MODELS_DIR).is_dir():
            Path(MODELS_DIR).mkdir(parents=True,exist_ok=True)

        if PLOT_AVERAGE:
            fig,ax = self._plot1()

        self.__init_initialize_metrics()

        for e in range(self.epochs):

            total_loss, times, n_FN, n_FP, n_TP, n_TN = self.__reset_metrics()

            avr_FN, avr_FP, avr_TN, avr_TP = self.__update_avg_metrics()
    
            self.net.eval()
            # if debug:
                # full_dataset.eval_()

            with torch.no_grad():
                for i,data in enumerate(self.dataloader_test):

                    inputs, pred, pop = self._prepare_batch(data)

                    outputs = self.net(inputs,pop)

                    loss = self._set_loss(inputs, pred, pop, self.criterion_test)

                    total_loss += loss.item()

                    times += 1

                    if self.debug:
                        avr_FP, avr_FN, avr_TP, avr_TN, n_FP, n_FN, n_TP, n_TN = self._debug_and_compute_metrics(inputs, pop,
                                                                                                                  pred, outputs, 
                                                                                                                  avr_FP, avr_FN, 
                                                                                                                  avr_TP, avr_TN,
                                                                                                                  n_FP, n_FN, 
                                                                                                                  n_TP, n_TN
                                                                                                                )
            if self.debug:
                self.__update_matrices(e, total_loss, times, n_TP, n_FP, n_TN, n_FN)
                # print("FP: {0}, FN : {1}, TP: {2}, TN: {3}".format(n_FP,n_FN,n_TP,n_TN))
                self.accuracy =  self._calculate_and_print_metrics(e, n_TP, n_TN, n_FP, n_FN, loss)
            
            if max_accuracy < self.accuracy:
                self.accuracy = max_accuracy
                self._save_best_model()

                if PLOT_AVERAGE and e % 20 == 0:
                    self._plot_average_matrices(e, fig, ax, avr_FP, avr_FN, avr_TP, avr_TN, n_FP, n_FN, n_TP, n_TN)
            else:
                print("Epoch: {0}".format(e))

            if total_loss/times < min_loss:
                min_loss = self.__save_min_loss_model(total_loss, times)
        
            self.full_dataset.train()
            
            self.net.train()

            for i,data in enumerate(self.dataloader_train):
                self.optimizer.zero_grad()

                inputs, pred, pop = self._prepare_batch(data)
                
                loss =  self._set_loss(inputs, pred, pop, self.criterion)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.net.parameters(), CLIP_GRAD_NORM)

                self.optimizer.step()    

            if e % 100 == 0:
                self.scheduler.step()


        if self.debug:
            self._write_to_stats()

    def _write_to_stats(self) -> None :
        data = {'TP':self.TP_arr,'TN':self.TN_arr,'FP':self.FP_arr,'FN':self.FN_arr,'loss':self.loss_arr}
        df_stats = pd.DataFrame(data)
        df_stats.to_csv('{results_path}/stats-r{ratio}.csv'.format(results_path=RESULTS_PATH,ratio=self.n_snps))

    def _set_loss(self, inputs, pred, pop, func):
        outputs = self.net(inputs,pop)
        return func(outputs, pred)

if __name__ == "__main__":
    train = Train(1000,1000,32,0.8,10,"simulations/simulated_data.csv",False,True,False)
    train.train()