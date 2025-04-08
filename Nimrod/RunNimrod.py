from utilities import json_get
from pathlib import Path
import allel
import numpy as np
import torch
from sklearn.manifold import MDS
import pandas as pd
import click
import matplotlib.pyplot as plt
import re 
from net import Net
from const import *
class Run:
    def __init__(self, vcf:str ,pheno_path:str ,trait:str ,model:str ,output_path:str ,cpu:bool):
        self.vcf = vcf
        self.pheno_path = pheno_path
        self.trait = trait
        self.model = model
        self.output_path = output_path
        self.cpu = cpu


    def run(self):
        """Run on real data"""

        width = json_get('width')
        n_samples = json_get('samples')

        if not Path("vcf_data").is_dir():
            Path("vcf_data").mkdir(parents=True,exist_ok=True)

        npz_loc = "vcf_data/{0}.npz".format(Path(self.vcf).stem)

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

        device = CPU if self.cpu else torch.device(CUDA if torch.cuda.is_available() else CPU)
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

        if not Path(self.pheno_path).is_file():
            print("Invalid file pheno")
            exit(1)

        pheno = pd.read_csv(self.pheno_path,index_col=None,sep=',')
        if not 'sample' in pheno.keys():
            raise click.ClickException('sample field missing in phenotype file')
        if not self.trait in pheno.keys():
            raise click.ClickException('trait field missing in phenotype file') 

        _,index_samples,index_samples_pheno = np.intersect1d(vcf_samples,pheno["sample"],return_indices=True)

        # df_ss = pd.DataFrame(ss,columns=['sample'])
        # pheno = pd.concat([df_ss,pheno],axis=0)
        final_vcf = final_vcf[:,index_samples]
        pheno = pheno.loc[index_samples_pheno].reset_index()
        #assert (pheno["sample"] == vcf_samples[index_samples]).all()
        #assert (vcf_samples == pheno['sample']).all()

        pheno_sorted = pheno.sort_values(by=[self.trait,"sample"],na_position='first')

        sorted_axes = np.array(pheno_sorted.index.values)
        sorted_vcf = final_vcf[:,sorted_axes]

        df_chrom = pd.DataFrame(chrom)
        chrom_labels = df_chrom[0].unique().tolist()

        input_s = torch.split(final_vcf,n_snps)
        output = torch.zeros((final_vcf.shape[0])).float().to(device)
            
        net = Net(n_snps,n_samples,1,width).to(device)
        net.load_state_dict(torch.load(self.model,torch.device(device))['model_state_dict'])

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

        chrom_labels.sort(key=self.num_sort)
        chr_loc = []


        min = 0
        
        # avr = torch.mean(output)
        # output -= avr

        print(100 * (torch.count_nonzero(output > min)/output.shape[0]).item())

        index_tmp = (output > min).nonzero().flatten().numpy()
        value_tmp = output.detach().clone()[index_tmp].flatten().numpy()

        df = pd.DataFrame({"value":value_tmp},index=index_tmp)
        df.to_csv('{0}.csv'.format(self.output_path))

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
        fig.savefig('{0}.png'.format(self.output_path))

    def num_sort(self, test_string: str)->int:
        return list(map(int, re.findall(r'\d+', test_string)))[0]



if __name__ == "__main__":
    Run('data.vcf','data.csv','trait','model.pth','output',True).run()