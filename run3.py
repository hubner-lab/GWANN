import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from utilities import json_get, createImages
from mylogger import Logger
import os 
from scipy.special import logit
import numpy as np
import plotly.graph_objects as go
import allel
import re
from scipy.ndimage import laplace


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
    GeneModel = {'recessive':0,'dominant':1,'additive':0.5,'noHet':-1}
    def __init__(self,vcf, pheno_path, trait,model, output_path, cpu, func, th, gm):
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
        self.gm = gm


    def load_and_parse_data(self):
        npz_loc = "vcf_data/{0}.npz".format(Path(self.vcf).stem)

        if not Path(npz_loc).is_file():
            Logger(f'Message:', os.environ['LOGGER']).info(f"Converting VCF to NPZ: {self.vcf} to {npz_loc}")
            Logger(f'Message:', os.environ['LOGGER']).info(f"It may take a while to convert VCF to NPZ, please wait...")
            allel.vcf_to_npz(self.vcf, npz_loc, fields='*', overwrite=True,chunk_length=8192,buffer_size=8192)

        callset = np.load(npz_loc, allow_pickle=True)
        Logger(f'Message:', os.environ['LOGGER']).info(f"Parsing VCF file: {self.vcf}")

        if not Path(self.pheno_path).is_file():
            print("Invalid phenotype file")
            exit(1)

        Logger(f'Message:', os.environ['LOGGER']).info(f"Loading phenotype file: {self.pheno_path}")
        pheno = pd.read_csv(self.pheno_path, index_col=None, sep=',')
        if 'sample' not in pheno.keys():
            raise ValueError('Sample field missing in phenotype file')
        if self.trait not in pheno.keys():
            raise ValueError('Trait field missing in phenotype file')
        return callset['calldata/GT'], callset['samples'], callset['variants/CHROM'], pheno 

    def calc_avg_vcf(self, vcf_data):

        tmp_vcf = vcf_data[:, :, 0] + vcf_data[:, :, 1]

        tmp_vcf[tmp_vcf == -2] = -1  
        tmp_vcf = (tmp_vcf + 1) * 1/3

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
        
    def load_model_and_predict(self, sorted_vcf, funcName=""):
        

        X_input = np.expand_dims(createImages(self.width,  sorted_vcf, self.sim_indivduals), axis=-1)  # Add channel dim (height, width, 1)

        Logger(f'Message:', os.environ['LOGGER']).info(f"Loading trained TensorFlow model: {self.model_path}...")
        model = load_model(self.model_path)


        Logger(f'Message:', os.environ['LOGGER']).info(f"Running inference on the model...")
        predictions = model.predict(X_input, batch_size=4096)  # batch_size to make the prediction process run faster, multi process created 

        output =  predictions[:, 1]
        df = pd.DataFrame({"value": output}, index=range(len(output)))
        df.to_csv(f"{self.output_path}.csv")
        self.logger.info(f"Output saved to {self.output_path}.csv")

        return  self.get_output_modified(output, funcName)

    
    def filter_chrom(self, chrom):
        chrom_arr = np.unique(chrom)

        filtered = []
        for ch in chrom_arr:
            l = len(ch)
            build_chrom = ""
            if  ch[l-1].isalpha():
                ch = ch[:l-1]
            for c in ch[::-1]:
                if c.isalpha():
                    break
                build_chrom = c + build_chrom
            filtered.append("chrome"+build_chrom)

        sort_keys = [] 
        for label in filtered:
            match = re.search(r'chrome(\d+)', label)  
            if match:
                num_part = int(match.group(1))
                sort_keys.append(num_part)
            else:
                sort_keys.append(float('inf'))
        sorted_labels = [label for _, label in sorted(zip(sort_keys, chrom_arr))]       
        return sorted_labels
    
    def plot_data(self, chrom, output):
    
        chrom_labels = self.filter_chrom(chrom)

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

        # Adding the 50% threshold line, aligned with the chromosome x-axis, and color-coded (green)
        fig.update_layout(
            title="Prediction of SNPs associated with the trait.",
            xaxis=dict(tickmode='array', tickvals=x_ticks, ticktext=x_tick_labels, title="Chromosome"),
            yaxis=dict(title="Prediction(%)", range=[self.th, 100]),
            showlegend=True,
            shapes=[
                dict(
                    type='line',
                    x0=0,
                    x1=current_position,  # spans the full x-axis range aligned with chromosomes
                    y0=50,
                    y1=50,
                    line=dict(color='green', width=2, dash='dash'),
                    name="50% Threshold Line"
                )
            ],
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=1,
                    y=1,
                    buttons=[
                        dict(
                            label="Show 50% Line",
                            method="relayout",
                            args=[{"shapes[0].visible": True}]
                        ),
                        dict(
                            label="Hide 50% Line",
                            method="relayout",
                            args=[{"shapes[0].visible": False}]
                        )
                    ]
                )
            ]

        )

        fig.write_html(f"{self.output_path}.html")
        self.logger.info(f"Scatter plot saved to {self.output_path}.html")

    def start(self):
        """Run on real data using a trained TensorFlow model"""


        vcf_data, vcf_samples, chrom, pheno = self.load_and_parse_data()

        tmp_vcf = self.calc_avg_vcf(vcf_data)

        _, index_samples, index_samples_pheno = np.intersect1d(vcf_samples, pheno["sample"], return_indices=True)
        final_vcf = tmp_vcf
        final_vcf = final_vcf[:, index_samples]
        pheno = pheno.loc[index_samples_pheno].reset_index()

        pheno_sorted = pheno.sort_values(by=[self.trait, "sample"], na_position='first')
        sorted_axes = np.array(pheno_sorted.index.values)
        sorted_vcf = final_vcf[:, sorted_axes]


        output = self.load_model_and_predict(sorted_vcf, self.func)
        causal_snps = len(np.where(output/100 > 0.5)[0])
        percentage = causal_snps/len(output)
        print(f'Causal SNPs(%):{100*percentage}')
        self.plot_data(chrom, output)



if __name__ == '__main__':
    pass