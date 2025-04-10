import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from genomeToImage import GenomeImage
from utilities import json_get
from mylogger import Logger
import os 
from scipy.special import logit

def createImages(columns, data, sim_indivduals):
    X_list = []
    individuals = data.shape[1]
    if individuals  < sim_indivduals:
        pad = np.zeros((data.shape[0], sim_indivduals - individuals))
        data = np.concatenate((data, pad), axis=1)
    rows = int(sim_indivduals / columns)
    genomeImage = GenomeImage(rows, columns)
    for sample in data:
        image = genomeImage.transform_to_image(sample)
        X_list.append(image)  # Append image to the list
    X = tf.convert_to_tensor(X_list, dtype=tf.float32)  # Adjust dtype if needed
    return X



class Run:
    def __init__(self,vcf, pheno_path, trait,model, output_path, cpu):
        self.vcf = vcf
        self.pheno_path = pheno_path
        self.trait = trait
        self.model_path = model
        self.output_path = output_path
        self.cpu = cpu


    def start(self):
        """Run on real data using a trained TensorFlow model"""
        width = json_get("width")
        sim_indivduals = json_get("samples")
        if not Path("vcf_data").is_dir():
            Path("vcf_data").mkdir(parents=True, exist_ok=True)

        npz_loc = f"vcf_data/{Path(self.vcf).stem}.npz"


        Logger(f'Message:', os.environ['LOGGER']).info(f"Loading VCF file: {self.vcf}")
        callset = np.load(npz_loc, allow_pickle=True)

        Logger(f'Message:', os.environ['LOGGER']).info(f"Parsing VCF file: {self.vcf}")
        vcf_data = callset['calldata/GT']
        vcf_samples = callset['samples']
        chrom = callset['variants/CHROM']

        tmp_vcf = (vcf_data[:, :, 0] + vcf_data[:, :, 1]) / 2
        tmp_vcf[np.where(tmp_vcf == 0.5)] = 0


        # print('Running MDS for population structure...')
        # embedding = MDS(n_components=1, random_state=0)
        # mds_data = embedding.fit_transform(tmp_vcf.T)

        # pop = np.expand_dims(mds_data, axis=-1)  # Ensure correct shape for the model

        if not Path(self.pheno_path).is_file():
            print("Invalid phenotype file")
            exit(1)

        Logger(f'Message:', os.environ['LOGGER']).info(f"Loading phenotype file: {self.pheno_path}")
        pheno = pd.read_csv(self.pheno_path, index_col=None, sep=',')
        if 'sample' not in pheno.keys():
            raise ValueError('Sample field missing in phenotype file')
        if self.trait not in pheno.keys():
            raise ValueError('Trait field missing in phenotype file')

        _, index_samples, index_samples_pheno = np.intersect1d(vcf_samples, pheno["sample"], return_indices=True)
        final_vcf = tmp_vcf
        final_vcf = final_vcf[:, index_samples]
        pheno = pheno.loc[index_samples_pheno].reset_index()

        pheno_sorted = pheno.sort_values(by=[self.trait, "sample"], na_position='first')
        sorted_axes = np.array(pheno_sorted.index.values)
        sorted_vcf = final_vcf[:, sorted_axes]

        chrom_arr = np.unique(chrom)
        chrom_labels =  chrom_arr[ np.argsort([int(x[2:]) for x in chrom_arr])] 

        # Reshape input to match model expectations

        X_input = np.expand_dims(createImages(width,  sorted_vcf, sim_indivduals), axis=-1)  # Add channel dim (height, width, 1)

        Logger(f'Message:', os.environ['LOGGER']).info(f"Loading trained TensorFlow model: {self.model_path}...")
        model = load_model(self.model_path)


        Logger(f'Message:', os.environ['LOGGER']).info(f"Running inference on the model...")
        predictions = model.predict(X_input, batch_size=4096)  # batch_size to make the prediction process run faster, multi process created 

        output = predictions.flatten() 

        EPSILON = 0.1
        clipoutput = np.clip(output, EPSILON, 1 - EPSILON)  # log-odds
        logitoutput = logit(clipoutput)

        # sigmoid_th = 0.5 
        # print(f"Thresholded predictions: {100 * (np.sum(output > sigmoid_th) / output.shape[0]):.2f}%")

        df = pd.DataFrame({"value": predictions.flatten() }, index=range(len(output)))
        df.to_csv(f"{self.output_path}.csv")


        Logger(f'Message:', os.environ['LOGGER']).info(f"Generating scatter plot...")
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



if __name__ == '__main__':
    pass