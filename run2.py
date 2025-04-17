import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utilities import json_get, createImages
from mylogger import Logger
import os 
from scipy.special import logit
from const import VCF_DATA_DIR




class Run:
    def __init__(self,vcf, pheno_path, trait,model, output_path, cpu):
        self.vcf = vcf
        self.pheno_path = pheno_path
        self.trait = trait
        self.model_path = model
        self.output_path = output_path
        self.cpu = cpu
        self.width = json_get("width")
        self.sim_indivduals = json_get("samples")



    def load_and_parse_data(self):
        if not Path(VCF_DATA_DIR).is_dir():
            Path(VCF_DATA_DIR).mkdir(parents=True, exist_ok=True)

        npz_loc = f"{VCF_DATA_DIR}/{Path(self.vcf).stem}.npz"
        Logger(f'Message:', f"{os.environ['LOGGER']}").info(f"Loading VCF file: {self.vcf}")

        callset = np.load(npz_loc, allow_pickle=True)
        Logger(f'Message:', f"{os.environ['LOGGER']}").info(f"Parsing VCF file: {self.vcf}")

        if not Path(self.pheno_path).is_file():
            print("Invalid phenotype file")
            exit(1)

        Logger(f'Message:', f"{os.environ['LOGGER']}").info(f"Loading phenotype file: {self.pheno_path}")
        pheno = pd.read_csv(self.pheno_path, index_col=None, sep=',')
        if 'sample' not in pheno.keys():
            raise ValueError('Sample field missing in phenotype file')
        if self.trait not in pheno.keys():
            raise ValueError('Trait field missing in phenotype file')
        
        return callset['calldata/GT'], callset['samples'], callset['variants/CHROM'], pheno 

    def calc_avg_vcf(self, vcf_data):
        tmp_vcf = (vcf_data[:, :, 0] + vcf_data[:, :, 1]) / 2
        tmp_vcf[np.where(tmp_vcf == 0.5)] = 0
        return tmp_vcf
    

    def load_model_and_predict(self, sorted_vcf, pop):
        """Load the model and make predictions on the data
        Args:
            sorted_vcf (np.ndarray): The sorted VCF data.
            pop (np.ndarray): population structure.
        Returns: 
        """
        # Reshape input to match model expectatons
        X_input = np.expand_dims(createImages(self.width,  sorted_vcf, self.sim_indivduals), axis=-1)  # Add channel dim (height, width, 1)

        pop = np.repeat(pop, X_input.shape[0], axis=0) # repeat the population structure to match the number of samples

        Logger(f'Message:', f"{os.environ['LOGGER']}").info(f"Loading trained TensorFlow model: {self.model_path}...")
        model = load_model(self.model_path)

        Logger(f'Message:', f"{os.environ['LOGGER']}").info("Running inference on the model...")
        # Pass both inputs as a list to predict
        predictions = model.predict([X_input, pop], batch_size=4096)
        output = predictions.flatten()

        EPSILON = 0.1
        clipoutput = np.clip(output, EPSILON, 1 - EPSILON)
        logitoutput = logit(clipoutput)

        df = pd.DataFrame({"value": predictions.flatten()}, index=range(len(output)))
        df.to_csv(f"{self.output_path}.csv")

        return logitoutput

    def plot_data(self, chrom, logitoutput):
        chrom_arr = np.unique(chrom)
        chrom_labels =  chrom_arr[ np.argsort([int(x[2:]) for x in chrom_arr])] 

        Logger(f'Message:', f"{os.environ['LOGGER']}").info(f"Generating scatter plot...")
        plt.figure(figsize=(12, 5))

        x_ticks = []
        x_tick_labels = []
        current_position = 0

        for i, chr_label in enumerate(chrom_labels):
            chr_indices = np.where(chrom == chr_label)[0]
            chr_outputs = logitoutput[chr_indices]

            x_chr = np.arange(current_position, current_position + len(chr_indices))
            
            # Alternate color: even index = blue, odd index = black
            color = 'blue' if i % 2 == 0 else 'red'

            plt.scatter(x_chr, chr_outputs, s=3, alpha=0.6, color=color, label=chr_label)

            x_ticks.append(current_position + len(chr_indices) // 2)
            x_tick_labels.append(chr_label)

            current_position += len(chr_indices)

        plt.xticks(x_ticks, x_tick_labels, rotation=45)
        plt.xlabel("Chromosome")
        plt.ylabel("Prediction Logit Score")
        plt.title("Prediction of SNPs associated with the trait.")
        plt.ylim(0, max(logitoutput))
        plt.tight_layout()
        plt.legend(markerscale=6, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        plt.savefig(f"{self.output_path}.png")
        Logger(f'Message:', f"{os.environ['LOGGER']}").info(f"Scatter plot saved to {self.output_path}.png")

    def start(self):
        """Run on real data using a trained TensorFlow model"""


        vcf_data, vcf_samples, chromosomes, pheno = self.load_and_parse_data()

        tmp_vcf = self.calc_avg_vcf(vcf_data)

        embedding = MDS(n_components=1, random_state=0, normalized_stress="auto")
        pop = embedding.fit_transform(tmp_vcf.T).T

        # Pad the vector pop with zeros to match the length of samples
        if pop.shape[1] < self.sim_indivduals:
            pad_width = self.sim_indivduals - pop.shape[1]
            pop = np.pad(pop, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)

        _, index_samples, index_samples_pheno = np.intersect1d(vcf_samples, pheno["sample"], return_indices=True)
        final_vcf = tmp_vcf
        final_vcf = final_vcf[:, index_samples]
        pheno = pheno.loc[index_samples_pheno].reset_index()

        pheno_sorted = pheno.sort_values(by=[self.trait, "sample"], na_position='first')
        sorted_axes = np.array(pheno_sorted.index.values)
        sorted_vcf = final_vcf[:, sorted_axes]


        logitoutput = self.load_model_and_predict(sorted_vcf, pop)

        self.plot_data(chromosomes, logitoutput)



if __name__ == '__main__':
    pass