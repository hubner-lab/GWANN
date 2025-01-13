
from typing import List, Tuple
from const import SAMPLES, SIM_PATH, GENOME_EXE
from utilities import  json_update
import numpy as np
import multiprocessing
from glob import glob
import re 
from functools import partial
import os
import click 
from const import SIM_GENOM_PATH, LOWER_BOUND, UPPER_BOUND
import shlex
import subprocess


class Simulate:
    def __init__(self, pop: int, subpop: int, n_samples: int, n_sim: int, n_snps: int, maf: float, miss: float, equal: bool, debug: bool):
        """
        Initialize the simulation parameters.

        Args:
            pop (int): Number of SNPs in each simulation.
            subpop (int): Number of expected subpopulations.
            n_samples (int): Number of individuals.
            n_sim (int): Number of populations to be simulated.
            n_snps (int): Number of causal SNPs expected per number of SNP-sites.
            maf (float): Minor allele frequency.
            miss (float): Proportion of missing data.
            equal (bool): Set this if equal variance is expected among SNPs (ignore for single SNP)
            debug (bool): Flag to enable verbose logging for debugging.
        """
        self.pop = pop
        self.subpop = subpop
        self.n_samples = n_samples
        self.n_sim = n_sim
        self.n_snps = n_snps
        self.maf = maf
        self.miss = miss
        self.equal = equal
        self.debug = debug
        

    def simulate(self):

        # assert width < n_samples ,"image width is bigger than the number of samples"
        # assert n_samples % width == 0,"image width does not divide the number of samples"
    
        json_update(SAMPLES,self.n_samples)
            
        seed_arr = self.seed()

        np.random.shuffle(seed_arr)
        
        # Knowing the number of CPU cores can help you decide how many processes to create for parallel execution.
        cpus = multiprocessing.cpu_count()
        # Create a pool of worker processes to execute tasks concurrently. Here's a breakdown of what it does:
        pool = multiprocessing.Pool(cpus)
        # TODO: Ask Sariel about the parameters 
        genome_command, phenosim_command, samples_str = self.generate_commands()

        try:
            self.debug_message1(cpus,samples_str)
            self.create_simulations(pool, genome_command, phenosim_command, seed_arr)
        except OSError:
            if not os.path.exists(GENOME_EXE):
                raise click.ClickException('genome simulator not found') 
        self.debug_message2()

    
    def create_simulations(self,pool, genome_command, phenosim_command, seed_arr):
            # This is a utility from the functools module. It creates a new function (ss) by pre-filling some arguments of the original function (self.simulate_helper).
            ss = partial(self.simulate_helper,genome_command,phenosim_command,seed_arr)
            # pool.map: This is a multiprocessing function that applies ss (the worker function) to each element in the provided iterable (range(self.n_sim)).
            pool.map(ss,range(self.n_sim))
    
    def create_var(self) -> np.ndarray:
        variance = np.ones(self.n_snps)
        # set this if equal variance is expected among SNPs (ignore for single SNP)
        if self.equal:
            variance = (variance / self.n_snps) * 0.99
        else:
            # Function is being used to generate a sample from a Dirichlet distribution
            variance = np.random.dirichlet(variance,size=1) * 0.99
        var_str = np.array2string(variance,precision=5,separator=',',formatter={'float_kind':lambda x: "%.5f" % x})
        return re.sub("\[|\s*|\]","",var_str)

    def seed(self) -> np.ndarray:
        """
        Generates a seeded population array with random integers added to each element.

        Returns:
        np.ndarray: An array of integers where each element is a unique integer from 0 to pop-1 
                    with a random integer between LOWER_BOUND=1 and UPPER_BOUND=1000000 added to it.
        """
        return np.array(list(range(self.pop))) + np.random.randint(LOWER_BOUND, UPPER_BOUND)

        
    def generate_samples_str(self) -> str:
        """
            Generate a string representation of the sample distribution.
            
            Args:
            - n_samples: Total number of samples.
            - subpop: Number of subpopulations.
            
            Returns:
            - str: Space-separated string of samples distribution.
        """              
        tmp = (self.n_samples // self.subpop)
        tmp2 = self.n_samples - tmp * self.subpop
        # convert the number of samples from an array of strings to one string
        return  " ".join([str(tmp)] * (self.subpop - 1) + [str(tmp + tmp2)])
    


    def generate_genome_command(self) -> Tuple[List[str], str]:
        samples_str = self.generate_samples_str()
        # In the provided code, the shlex.split function is used to split a command string into a list of arguments, similar to how a shell would parse the command.
        genome_command = shlex.split(f"{GENOME_EXE} -s {self.pop} -pop {self.subpop} {samples_str} -seed")
        return genome_command, samples_str

    def generate_phenosim_command(self) -> str:
        phenosim_command = (
            f"python2 simulation/phenosim/phenosim.py -i G "
            f"-f {SIM_PATH}/genome{{0}}.txt --outfile {SIM_PATH}/{{0}} "
            f"--maf_r {self.maf},1.0 --maf_c {self.maf} --miss {self.miss}"
        )
        return phenosim_command
    

    def generate_commands(self) -> Tuple[List[str], str]:
        """
        Generate genome and phenosim commands based on given parameters.

        Returns:
            tuple: A tuple containing the genome command (list) and the phenosim command (str).
        """
        genome_command, samples_str = self.generate_genome_command()
        phenosim_command = self.generate_phenosim_command()
        if self.n_snps > 1:
            var_str = self.create_var()
            phenosim_command += " -n {snps} -v {var}".format(snps=self.n_snps,var=var_str)
        return genome_command, phenosim_command, samples_str
    


    def simulate_helper(self, genome_command,phenosim_command,seed,i):

        out_file = open(f'{SIM_GENOM_PATH}{i}.txt','w')
        try:
            subprocess.call(genome_command + ["{0}".format(seed[i])],stdout=out_file)
            out_file.close()
        except Exception as e:
            raise Exception(f'genome failed for {i}; with {e}')

        try:
            phenosim_command = shlex.split(phenosim_command.format(i))
            subprocess.call(phenosim_command,stdout=subprocess.DEVNULL)
        except Exception as e:
            raise Exception('phenosim failed for {i}; with {e}'.format(i=i, e=e))
        
    
    def debug_message1(self, cpus:int, samples_str:str):
        if self.debug:
            print(f'Mapping using {cpus} CPUs')
            print(f'Output directory: {SIM_PATH}')
            print(f'Genome parameters: population={self.pop}, subpopulations={self.subpop}, samples={samples_str}')
            print(f'Phenosim parameters: maf_c={self.maf}, maf_r={self.maf}, miss={self.miss}')

    def debug_message2(self):
        if self.debug:
            genome_files = glob(f'{SIM_PATH}/genome*.txt')
            emma_files = glob(f'{SIM_PATH}/*.causal')
            print(f'n simulations: expected={self.n_sim}, genomes={len(genome_files)}, phenotype={len(emma_files)}')
        
if __name__ == '__main__':
    sim = Simulate(100, 5, 100, 100, 1000, 0.05, 0.05, False, True)
    sim.simulate()