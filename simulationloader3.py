import pandas as pd
import numpy as np 
from sklearn.manifold import MDS
from utilities import json_get
from const import TOTAL_SNPS 
class SimulationDataReader:
    SUFFIX_EMMA_GENO = '0.emma_geno'
    SUFFIX_CAUSAL = '0.causal'
    SUFFIX_EMMA_PHENO = '0.emma_pheno'
    COLUMN = 1 
    count_once = False
    SNP_total = 0
    def __init__(self, base_path:str, mds_flag:bool):
        self.base_path = base_path

        self.sorted_causal_indices = None
        self.geno_data_pd = None
        self.causal_snp_data_pd = None
        self.pheno_data_pd = None

        
        self.geno_data_np = None
        self.causal_snp_data_np = None
        self.pheno_data_np = None


        self.phone_arg_sorted = None
        self.mds_arg_sorted = None

        # self.pheno_data_sorted = None
        self.geno_data_reordered_pheno = None
        self.geno_data_reordered_mds = None

        self.causal_sorted = None
        self.mds_data_sorted = None
        self.labels = None
        self.mds_data = None
        self.cache = dict()
        self.mds_flag = mds_flag

    def run(self, index, total_samples):
        self.load(index, total_samples)
        self.sort_phenotype()
        self.reorder_geno_data_according_to_pheno_sorted_order()

        self.mds_data = self.apply_mds_transformation()

        if self.mds_flag:
            self.sort_mds()

            # Get non-causal SNPs in current matrix
            non_causal_snps_indices = np.where(self.labels == 0)[0]

            self.reorder_geno_data_according_to_MDS_sorted_order(non_causal_snps_indices)

            # Duplicate non-causal data to make TNs
            labels = np.concatenate((self.labels, np.zeros(len(non_causal_snps_indices), dtype=int)), axis=0)
            inputMat = np.concatenate((self.geno_data_reordered_pheno, self.geno_data_reordered_mds), axis=0)

            return {
                'input': inputMat,
                'labels': labels,
            }
        else:
            return {
                'input': self.geno_data_reordered_pheno,
                'labels': self.labels,
            }

    
    def sort_mds(self):
        "Sort the vector that was given by the MDS transformation, and set mds_arg_sorted to the indices order after sorting"
        if self.mds_data is not None and self.mds_data.size > 0 :
            sorted_indices = np.argsort(self.mds_data[0, :])
            self.mds_data_sorted = self.mds_data[:, sorted_indices]
            self.mds_arg_sorted = sorted_indices

    def reorder_geno_data_according_to_MDS_sorted_order(self, non_causal_snps_indices):
        """Reorder the genotype Matrix columns according to the order of mds sorted indices order
            Consider only the passed snps indices
            This is done to add TN samples (in addition to the total samples)
            The process done to avoid the population structure and the way its effect the GWANN"""
        if self.mds_data is not None and self.mds_data.size > 0 :
            temp = self.geno_data_np[non_causal_snps_indices,:] # Take the specific snps  sub-Matrix 
            self.geno_data_reordered_mds  = temp[:, self.mds_arg_sorted] # reorder sub-Matrix columns according to the mds sorted indices 


    def load(self, index, total_samples=None):
        emma_geno_file = f'{self.base_path}{index}{self.SUFFIX_EMMA_GENO}'
        causal_file = f'{self.base_path}{index}{self.SUFFIX_CAUSAL}'
        pheno_file = f'{self.base_path}{index}{self.SUFFIX_EMMA_PHENO}'

        self.causal_snp_data_np = pd.read_csv(causal_file, header=None, sep='\t').to_numpy()
        self.sort_causal_snps()
        causal_indices = self.get_causal_snps_indices_after_sorting()
        n_causal = len(causal_indices)
        
        all_indices = np.arange(json_get(TOTAL_SNPS))
        non_causal_indices = np.setdiff1d(all_indices, causal_indices)
        if total_samples is None:
            total_samples = n_causal + len(non_causal_indices)
        n_non_causal = total_samples - n_causal
        sampled_non_causal = np.random.choice(non_causal_indices, n_non_causal, replace=False)
        all_sampled_indices = np.concatenate([causal_indices, sampled_non_causal])
        np.random.shuffle(all_sampled_indices)

        geno_data_list = []
        size = 0
        self.mapped_indices = dict()
        for i, line in enumerate(open(emma_geno_file)):
            if i in all_sampled_indices:
                geno_data_list.append([float(x) if x not in ('', 'NA', 'NaN') else -1 for x in line.strip().split('\t')])
                self.mapped_indices[i] = size
                size += 1
            if len(geno_data_list) == total_samples:
                break
        self.geno_data_np = np.array(geno_data_list)

        self.pheno_data_np = pd.read_csv(pheno_file, header=None, sep='\t').to_numpy()

        self.labels = np.zeros(total_samples, dtype=int)
        for causal_index in causal_indices:
            self.labels[self.mapped_indices[causal_index]] = 1



    def sort_phenotype(self):
        """Sort the phenotype array, and set the phone_arg_sorted to the indices order after sorting"""
        if self.pheno_data_np is not None and self.pheno_data_np.size > 0 :
            self.phone_arg_sorted = np.argsort(self.pheno_data_np).squeeze()
            # self.pheno_data_sorted = np.sort(self.pheno_data_np)

    def sort_causal_snps(self):
        "sort the causal snp file according to the indices"
        if self.causal_snp_data_np is not None and self.causal_snp_data_np.size > 0 :
            # Get sorted indices based on the first column
            sorted_causal_indices = np.argsort(self.causal_snp_data_np[:, 0])
            self.causal_sorted = self.causal_snp_data_np[sorted_causal_indices ]

    def reorder_geno_data_according_to_pheno_sorted_order(self):
        "Reorder the genotype Matrix columns according to the order of phenotype sorted indices order"
        if self.geno_data_np is not None and self.geno_data_np.size > 0  :
            self.geno_data_reordered_pheno = self.geno_data_np[:, self.phone_arg_sorted]

    def apply_labeling(self, causal_snps_indices, is_sigmoid:bool = False):
        "Label the SNPs, causal ==> 1, None-causal ==> 0"
        self.labels = np.empty(self.geno_data_reordered_pheno.shape[0])
        if is_sigmoid: 
            self.labels[:] = 0 
            self.labels[causal_snps_indices] = 1 
        else:
            self.labels[:] = -1 
            self.labels[causal_snps_indices] = 1 
    
    
    def get_causal_snps_indices_after_sorting(self):
        """Return the indices of the causal snps after sorting the causal data"""
        return (self.causal_sorted[:,self.COLUMN]).astype(int)
    
    def add_non_causal_snps_samples_randomly(self, total_samples, causal_snps_indices):
        """The parameter samples that was passed to the program will determine the number of SNPs to take from the total simulation
            Here we includes first the causal SNPs and later we add None causal SNPs ( the result total(causal snps) + total(None causal snps taken) = samples )
            The none causal snps are randomly taken
        """
        n = total_samples - len(causal_snps_indices) 

        sampled_indices_non_causal_snps = np.random.choice(np.setdiff1d(range(self.geno_data_reordered_pheno.shape[0]),causal_snps_indices), n, replace=False)  

        all_sampled_indices = np.concatenate((sampled_indices_non_causal_snps,causal_snps_indices))

        np.random.shuffle(all_sampled_indices)

        return all_sampled_indices, sampled_indices_non_causal_snps
    
        
    def apply_mds_transformation(self):
        """Apply MDS algorithm on the Matrix columns
            returns : vector (1XN) where N is number of individuals in the simulation """
        embedding = MDS(n_components=1, random_state=0, normalized_stress="auto")
        return embedding.fit_transform(self.geno_data_reordered_pheno.T).T
    


if __name__ == "__main__":
    data_reader = SimulationDataReader('./simulation/data/', True)
    data_reader.run(0, 200)