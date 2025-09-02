
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
from const import SIM_GENOM_PATH, LOWER_BOUND, UPPER_BOUND, SIMULATIONS, TOTAL_SNPS
import shlex
import subprocess
from mylogger import Logger


class Simulate:

    DEPOLIDS = 2
    
    def __init__(self, SNPs: int, subpop: int, n_samples: int, n_sim: int, causalSNPs: int, maf: float, miss: float, equal: bool, debug: bool, delete: bool):
        """
        Initialize the simulation parameters.

        Args:
            SNPs (int): Number of SNPs in each simulation.
            subpop (int): Number of expected subpopulations.
            n_samples (int): Number of individuals.
            n_sim (int): Number of populations to be simulated.
            causalSNPs (int): Number of causal SNPs expected per number of SNP-sites.
            maf (float): Minor allele frequency.
            miss (float): Proportion of missing data.
            equal (bool): Set this if equal variance is expected among SNPs (ignore for single SNP)
            debug (bool): Flag to enable verbose logging for debugging.
            delete (bool): Flag to delete the current simulated files.
        """
                
        self.SNPs = SNPs
        self.subpop = subpop
        self.n_samples = n_samples * self.DEPOLIDS
        self.n_sim = n_sim
        self.causalSNPs = causalSNPs
        self.maf = maf
        self.miss = miss
        self.equal = equal
        self.debug = debug
        self.delete = delete
        self.logger = Logger(f'Message:', f"{os.environ['LOGGER']}")
        json_update(SAMPLES,n_samples)
        json_update(SIMULATIONS,n_sim)
        json_update(TOTAL_SNPS,SNPs)


    def delete_files_in_directory(self, directory_path):
        if not os.path.exists(directory_path):
            return
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) and filename != '.gitignore':
                os.remove(file_path)
        

    def simulate(self):
        if not os.path.exists(SIM_PATH):
            os.makedirs(SIM_PATH)

        if self.delete:
            self.delete_files_in_directory(SIM_PATH)

        seed_arr = self._seed()
        
        np.random.shuffle(seed_arr)
        

        # Create a pool of worker processes to execute tasks concurrently. Here's a breakdown of what it does:
    
        self.logger.info(f"Generating commands...")
        genome_command, phenosim_command, samples_str = self._generate_commands()
        self.logger.info(f"Commands generated with the following parameters:\ngenome_command: {genome_command}\nphenosim_command: {phenosim_command}")


        # Knowing the number of CPU cores can help you decide how many processes to create for parallel execution.
        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpus)
        self.logger.info(f"Pool of workers created with {cpus} CPU cores.")
        try:
            self._debug_message1(cpus,samples_str)
            self.logger.info(f"Creating simulations...")
            self._create_simulations(pool, genome_command, phenosim_command, seed_arr)
            self.logger.info(f"Simulations created successfully.")
            pool.close()
        except OSError:
            if not os.path.exists(GENOME_EXE):
                raise click.ClickException('genome simulator not found, please install it from README.md under Dependencies section, or adjust the GENOME_EXE variable in const.py') 
        self._debug_message2()

    
    def _create_simulations(self,pool, genome_command, phenosim_command, seed_arr):
            # This is a utility from the functools module. It creates a new function (ss) by pre-filling some arguments of the original function (self.simulate_helper).
            ss = partial(self._simulate_helper,genome_command,phenosim_command,seed_arr)
            # pool.map: This is a multiprocessing function that applies ss (the worker function) to each element in the provided iterable (range(self.n_sim)).
            pool.map(ss,range(self.n_sim))
    

    def _create_var(self) -> np.ndarray:
        variance = np.ones(self.causalSNPs)
        # set this if equal variance is expected among SNPs (ignore for single SNP)
        if self.equal:
            variance = (variance / self.causalSNPs) * 0.99
        else:
            # Function is being used to generate a sample from a Dirichlet distribution
            variance = np.random.dirichlet(variance,size=1) * 0.99
        var_str = np.array2string(variance,precision=5,separator=',',formatter={'float_kind':lambda x: "%.5f" % x})
        return re.sub("\[|\s*|\]","",var_str)


    def _seed(self) -> np.ndarray:
        """
        Generates a seeded population array with random integers added to each element.

        Returns:
        np.ndarray: An array of integers where each element is a unique integer from 0 to number of simulations
                    with a random integer between LOWER_BOUND=1 and UPPER_BOUND=1000000 added to it.
        """
        return np.array(list(range(self.n_sim))) + np.random.randint(LOWER_BOUND, UPPER_BOUND)

        
    def _generate_samples_str(self) -> str:
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
        return  " ".join([str(tmp)] * (self.subpop - 1) + [str(tmp + tmp2)])

    
    def _generate_genome_command(self) -> Tuple[List[str], str]:
        samples_str = self._generate_samples_str()
        genome_command = shlex.split(f"{GENOME_EXE} -c {1} -s {self.SNPs} -pop {self.subpop} {samples_str} -seed")
        return genome_command, samples_str


    def _generate_phenosim_command(self) -> str:
        phenosim_command = (
            f"python2 simulation/phenosim/phenosim.py -i G "
            f"-d 1 -o P -f {SIM_PATH}/genome{{0}}.txt --outfile {SIM_PATH}/{{0}} "
            f"--maf_r {self.maf},1.0 --maf_c {self.maf} --miss {self.miss}"
        )
        return phenosim_command
    

    def _generate_commands(self) -> Tuple[List[str], str]:
        """
        Generate genome and phenosim commands based on given parameters.

        Returns:
            tuple: A tuple containing the genome command (list) and the phenosim command (str).
        """
        genome_command, samples_str = self._generate_genome_command()
        phenosim_command = self._generate_phenosim_command()
        if self.causalSNPs > 1:
            var_str = self._create_var()
            phenosim_command += " -n {snps} -v {var}".format(snps=self.causalSNPs,var=var_str)
        return genome_command, phenosim_command, samples_str
    

    def _simulate_helper(self, genome_command,phenosim_command,seed,i):
        """
        Helper function to simulate genome and phenotype data.

        This function runs genome and phenotype simulation commands using subprocess calls.
        It handles exceptions and logs errors if the commands fail.

        Args:
            genome_command (list): The command to run the genome simulation.
            phenosim_command (str): The command to run the phenotype simulation.
            seed (list): A list of seed values for the simulations.
            i (int): The index of the current simulation.

        Raises:
            Exception: If the genome simulation command fails.
            Exception: If the phenotype simulation command fails.
        """
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


    def _debug_message1(self, cpus:int, samples_str:str):
        if self.debug:
            print(f'Mapping using {cpus} CPUs')
            print(f'Output directory: {SIM_PATH}')
            print(f'Genome parameters: population={self.SNPs}, subpopulations={self.subpop}, samples={samples_str}')
            print(f'Phenosim parameters: maf_c={self.maf}, maf_r={self.maf}, miss={self.miss}')


    def _debug_message2(self):
        if self.debug:
            genome_files = glob(f'{SIM_PATH}/genome*.txt')
            emma_files = glob(f'{SIM_PATH}/*.causal')
            print(f'n simulations: expected={self.n_sim}, genomes={len(genome_files)}, phenotype={len(emma_files)}')
    

if __name__ == '__main__':
    sim = Simulate(100, 5, 100, 100, 1000, 0.05, 0.05, False, True, True)
    sim.simulate()