import pandas as pd
import numpy as np 
from sklearn.manifold import MDS
from utilities import json_get
from const import TOTAL_SNPS


class SimulationDataReader:

    ALLELE_MAPPING = {'A': 1, 'T': 0, '0': -1}
    GENMODEL={"minor" : 0, "major" : 2, "heterozygote" : 1, "missing" : -1}
    PED_COLUMNS = ['Family_ID', 'Individual_ID', 'Paternal_ID', 'Maternal_ID', 'Sex', 'Phenotype']
    SUFFIX_GENO_PED = '0.ped'
    SUFFIX_CAUSAL = '0.causal'
    COLUMN = 1 


    def __init__(self, base_path:str, mds_flag:bool):
        self.base_path = base_path
        self.geno_data_pd = None
        self.geno_data_np = None
        self.causal_snp_data_np = None
        self.pheno_data_np = None
        self.phone_arg_sorted = None
        self.mds_arg_sorted = None
        self.geno_data_reordered_pheno = None
        self.geno_data_reordered_mds = None
        self.causal_sorted = None
        self.mds_data_sorted = None
        self.labels = None
        self.mds_data = None
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


    def process_ped_data(self, all_snps):
        alleles_columns = [f'allel{i}' for i in range(all_snps*2) ]
        self.geno_data_pd.columns = self.PED_COLUMNS + alleles_columns

        matrix = self.geno_data_pd[alleles_columns].to_numpy()
        mapped_matrix = np.vectorize(self.ALLELE_MAPPING.get)(matrix)
        matrix1 = mapped_matrix[:, ::2]
        matrix2 = mapped_matrix[:, 1::2]

        self.geno_data_np = matrix1 + matrix2

        self.geno_data_np[self.geno_data_np == 1] = self.GENMODEL[json_get("gene_model")]
        self.geno_data_np[self.geno_data_np == -2] = -1 
        self.geno_data_np = (self.geno_data_np.T + 1) * 1/3

        self.pheno_data_np = self.geno_data_pd['Phenotype'].to_numpy()

        
    def load(self, index, total_samples=None):
        ped_geno_file = f'{self.base_path}{index}{self.SUFFIX_GENO_PED}'
        causal_file = f'{self.base_path}{index}{self.SUFFIX_CAUSAL}'

        all_snps = json_get(TOTAL_SNPS)
        all_indices = np.arange(all_snps)



        self.geno_data_pd = pd.read_csv(
                ped_geno_file,
                sep='\s+',
                engine='c',
                header=None,
                dtype=str,  # Treat all columns as strings to avoid type issues
                skip_blank_lines=True,
                na_values=None,  # Prevent empty strings from being treated as NaN
                keep_default_na=False
            )

        self.process_ped_data(all_snps)
        self.causal_snp_data_np = pd.read_csv(causal_file, header=None, sep='\t').to_numpy()
        self.sort_causal_snps()
        causal_indices = self.get_causal_snps_indices_after_sorting()
        n_causal = len(causal_indices)
        
    
        non_causal_indices = np.setdiff1d(all_indices, causal_indices)
        if total_samples is None:
            total_samples = n_causal + len(non_causal_indices)
        n_non_causal = total_samples - n_causal
        sampled_non_causal = np.random.choice(non_causal_indices, n_non_causal, replace=False)
        all_sampled_indices = np.concatenate([causal_indices, sampled_non_causal])
        # randomly shuffle the combined indices
        np.random.shuffle(all_sampled_indices)
        # mix the order of the indices to avoid any bias
        self.geno_data_np = self.geno_data_np[all_sampled_indices, :]

        self.labels = np.zeros(total_samples, dtype=int)
        # label the causal SNPs as 1
        for mapped_index, origin_index in enumerate(all_sampled_indices):
            if origin_index in causal_indices :
                self.labels[mapped_index] = 1


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
    
        
    def apply_mds_transformation(self):
        """Apply MDS algorithm on the Matrix columns
            returns : vector (1XN) where N is number of individuals in the simulation """
        embedding = MDS(n_init=4, n_components=1, random_state=0, normalized_stress="auto")
        return embedding.fit_transform(self.geno_data_reordered_pheno.T).T
    


if __name__ == "__main__":
    data_reader = SimulationDataReader('./simulation/data/', True)
    data_reader.run(0, 20)
