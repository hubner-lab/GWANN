import pandas as pd
import numpy as np 
from sklearn.manifold import MDS

class SimulationDataReader:
    SUFFIX_EMMA_GENO = '0.emma_geno'
    SUFFIX_CAUSAL = '0.causal'
    SUFFIX_EMMA_PHENO = '0.emma_pheno'
    COLUMN = 1 
    def __init__(self, base_path:str):
        self.base_path = base_path

        self.geno_data_pd = None
        self.causal_snp_data_pd = None
        self.pheno_data_pd = None

        
        self.geno_data_np = None
        self.causal_snp_data_np = None
        self.pheno_data_np = None


        self.phone_arg_sorted = None

        # self.pheno_data_sorted = None
        self.geno_data_reordered = None
        self.causal_sorted = None

        self.labels = None

        self.cache = dict()

    def run(self, index, total_samples):
        self.load(index)
        self.sort_phenotype()
        self.sort_causal_snps()
        self.reorder_geno_data_according_to_pheno_sorted_order()
        causal_snps_indices = self.get_causal_snps_indices_after_sorting()
        self.apply_labeling(causal_snps_indices, is_sigmoid=True)
        mds_data = self.apply_mds_transformation()
        sampled_indices= self.add_non_causal_snps_samples_randomly(total_samples, causal_snps_indices)

        return { 
                         'input':self.geno_data_reordered[sampled_indices,:], 
                         'labels': self.labels[sampled_indices],
                         'population': mds_data
                }

    def load(self, index):
        emma_geno_file  = f'{self.base_path}{index}{self.SUFFIX_EMMA_GENO}'
        causal_file =  f'{self.base_path}{index}{self.SUFFIX_CAUSAL}'
        pheno_file = f'{self.base_path}{index}{self.SUFFIX_EMMA_PHENO}'

        self.geno_data_np = pd.read_csv(emma_geno_file,index_col=None,header=None,sep='\t').fillna(-1).to_numpy()
        self.causal_snp_data_np = pd.read_csv(causal_file,index_col=None,header=None,sep='\t').to_numpy()
        self.pheno_data_np = pd.read_csv(pheno_file,index_col=None,header=None,sep='\t').to_numpy()



    def sort_phenotype(self):
        if self.pheno_data_np is not None and self.pheno_data_np.size > 0 :
            self.phone_arg_sorted = np.argsort(self.pheno_data_np).squeeze()
            # self.pheno_data_sorted = np.sort(self.pheno_data_np)

    def sort_causal_snps(self):
        if self.causal_snp_data_np is not None and self.causal_snp_data_np.size > 0 :
            # Get sorted indices based on the first column
            sorted_indices = np.argsort(self.causal_snp_data_np[:, 0])
            self.causal_sorted = self.causal_snp_data_np[sorted_indices]

    def reorder_geno_data_according_to_pheno_sorted_order(self):
        if self.geno_data_np is not None and self.geno_data_np.size > 0  :
            self.geno_data_reordered = self.geno_data_np[:, self.phone_arg_sorted]

    def apply_labeling(self, causal_snps_indices, is_sigmoid:bool = False):
        self.labels = np.empty(self.geno_data_reordered.shape[0])
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
        n = total_samples - len(causal_snps_indices)

        sampled_indices = np.random.choice(np.setdiff1d(range(self.geno_data_reordered.shape[0]),causal_snps_indices), n, replace=False)  

        sampled_indices = np.concatenate((sampled_indices,causal_snps_indices))

        np.random.shuffle(sampled_indices)

        return sampled_indices 
    
        
    def apply_mds_transformation(self):
        embedding = MDS(n_components=1, random_state=0, normalized_stress="auto")
        return embedding.fit_transform(self.geno_data_reordered.T).T

if __name__ == "__main__":
    data_reader = SimulationDataReader('./simulation/data/')
    data_reader.run(0, 200)