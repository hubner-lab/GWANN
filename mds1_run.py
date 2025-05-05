import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import MDS
from tensorflow.keras.models import load_model
from utilities import json_get, createImages
from mylogger import Logger
import os 
from scipy.special import logit
from const import VCF_DATA_DIR
import numpy as np
import plotly.graph_objects as go


def tanh_map(output, scale=10):
    return np.tanh(scale * (output - 0.5))


def logit_map(output):
        EPSILON = 0.1
        clipoutput = np.clip(output, EPSILON, 1 - EPSILON)
        return logit(clipoutput)


def log_map(output):
    EPSILON  = 0.0001
    clipoutput = np.clip(output, EPSILON, 1 - EPSILON)  # to avoid log(1-1) = log(0)
    res =  -np.log(1-clipoutput)
    resNorm = res / np.max(res) # normalize to 0-1
    SCALE = 100
    return  SCALE* resNorm


class Run:
    def __init__(self,vcf, pheno_path, trait,model, output_path, cpu, func, th):
        self.vcf = vcf
        self.pheno_path = pheno_path
        self.trait = trait
        self.model_path = model
        self.output_path = output_path
        self.cpu = cpu
        self.width = json_get("width")
        self.sim_indivduals = json_get("samples")
        self.logger = Logger(f'Message:', f"{os.environ['LOGGER']}")
        self.func = func
        self.th = th


    def load_and_parse_data(self):
        if not Path(VCF_DATA_DIR).is_dir():
            Path(VCF_DATA_DIR).mkdir(parents=True, exist_ok=True)

        npz_loc = f"{VCF_DATA_DIR}/{Path(self.vcf).stem}.npz"
        self.logger.info(f"Loading VCF file: {self.vcf}")

        callset = np.load(npz_loc, allow_pickle=True)
        self.logger.info(f"Parsing VCF file: {self.vcf}")

        if not Path(self.pheno_path).is_file():
            print("Invalid phenotype file")
            exit(1)

        self.logger.info(f"Loading phenotype file: {self.pheno_path}")
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
    

    def get_output_modified(self,output, funcName = ""):
        """
        Get the output of a function by modifying the function name.
        Args:
            funcName (str): The name of the function to modify.
        Returns:
            str: The modified function name.
        """
        if funcName == "tan":
            return tanh_map(output)
        elif funcName == "logit":
            return logit_map(output)
        elif funcName == "log":
            return log_map(output)
        else:
            SCALE = 100
            return SCALE* output
        
    
    def load_model_and_predict(self, sorted_vcf, pop, funcName=""):
        """Load the model and make predictions on the data
        Args:
            sorted_vcf (np.ndarray): The sorted VCF data.
            pop (np.ndarray): population structure.
        Returns: 
        """
        # Reshape input to match model expectatons
        X_input = np.expand_dims(createImages(self.width,  sorted_vcf, self.sim_indivduals), axis=-1)  # Add channel dim (height, width, 1)

        pop = np.repeat(pop, X_input.shape[0], axis=0) # repeat the population structure to match the number of samples

        self.logger.info(f"Loading trained TensorFlow model: {self.model_path}...")
        model = load_model(self.model_path)

        self.logger.info("Running inference on the model...")
        # Pass both inputs as a list to predict
        predictions = model.predict([X_input, pop], batch_size=4096)
        output = predictions.flatten()

        df = pd.DataFrame({"value": predictions.flatten()}, index=range(len(output)))
        df.to_csv(f"{self.output_path}.csv")
        self.logger.info(f"Output saved to {self.output_path}.csv")
        return  self.get_output_modified(output, funcName)


    def plot_data(self, chrom, output):
        chrom_arr = np.unique(chrom)
        chrom_labels = chrom_arr[np.argsort([int(x[2:]) for x in chrom_arr])]

        self.logger.info(f"Generating scatter plot with Plotly...")

        x_ticks = []
        x_tick_labels = []
        current_position = 0
        fig = go.Figure()

        for i, chr_label in enumerate(chrom_labels):
            chr_indices = np.where(chrom == chr_label)[0]
            chr_outputs = output[chr_indices]

            x_chr = np.arange(current_position, current_position + len(chr_indices))

            # Alternate color: even index = blue, odd index = black
            color = 'blue' if i % 2 == 0 else 'red'

            fig.add_trace(go.Scattergl(
                x=x_chr,
                y=chr_outputs,
                mode='markers',
                name=chr_label,
                marker=dict(size=3, opacity=0.6, color=color)
            ))

            x_ticks.append(current_position + len(chr_indices) // 2)
            x_tick_labels.append(chr_label)

            current_position += len(chr_indices)

        fig.update_layout(
            title="Prediction of SNPs associated with the trait.",
            xaxis=dict(tickmode='array', tickvals=x_ticks, ticktext=x_tick_labels, title="Chromosome"),
                yaxis=dict(title="Prediction(%)", range=[self.th, np.ceil(output)]),
            showlegend=True
        )

        fig.write_html(f"{self.output_path}.html")
        self.logger.info(f"Scatter plot saved to {self.output_path}.html")

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


        output = self.load_model_and_predict(sorted_vcf, pop, self.func)

        self.plot_data(chromosomes, output)



if __name__ == '__main__':
    pass