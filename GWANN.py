import click
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Simulate import Simulate
# from train6 import Train  # worked well
# from train11 import Train  # improved since data is extremely imbalanced
# from train11_BN import Train # for testing 
from train13 import Train  
from run3 import Run
# from run3_BN import Run
from utilities import json_get, json_update
import resource
from const import SIMULATIONS, LOGGER_DIR
import datetime
from mylogger import Logger
import time
import resource


def log_resource_usage(start_time, logger, label=""):
    end_time = time.time()
    usage = resource.getrusage(resource.RUSAGE_SELF)

    runtime = end_time - start_time
    mem_usage = usage.ru_maxrss / 1024  # MB
    user_cpu = usage.ru_utime
    sys_cpu = usage.ru_stime

    # Colored console output using logger (assumes StreamHandler supports color)
    logger.info(f"\033[96m{label} - Runtime: {runtime:.2f}s\033[0m")        
    logger.info(f"\033[93m{label} - Max Memory Usage: {mem_usage:.2f} MB\033[0m") 
    logger.info(f"\033[94m{label} - User CPU time: {user_cpu:.2f}s\033[0m")   
    logger.info(f"\033[94m{label} - System CPU time: {sys_cpu:.2f}s\033[0m")  

class CLIManager:
    def __init__(self):
        """Initialize the CLI Manager.
        
        This class is responsible for managing the CLI commands and running the 
        application. It includes the commands for running, simulating, and training 
        the model for GWANN analysis.
        """
        self.cli = click.Group()


    @staticmethod
    @click.command()
    @click.option('-v', '--vcf', 'vcf', required=True, help='path to the VCF file')
    @click.option('-p', '--pheno','pheno_path',required=True,help='path to the phenotype file (comma seperated csv file)')
    @click.option('-t', '--trait','trait',required=True,help='name of the trait (header in the phenotype file)')
    @click.option('--model','model',default=None,help="path to the network model generated in the training step")
    @click.option('--output','output_path',default="results/GWAS",help="prefix of output plot and causative SNPs indexes in the VCF")
    @click.option('--transform', '--f', 'func', default="", type=str, help="The name of the function to modify the output (tanh_map, logit_map, log_map)")
    @click.option('--threshold', '--th', 'th', default=50, type=int, help="Causal classification if  >= threshold (% Prediction)")
    def run(
        vcf: str,
        pheno_path:str,
        trait:str,
        model:str,
        output_path:str,
        func:str,
        th:int,
        )-> None:
        """Run GWANN analysis.

        This command performs the GWANN analysis using the provided VCF file, phenotype
        file, trait name, and trained model. It generates output files containing plots
        and SNP indexes.
        
        Args:
            vcf (str): Path to the VCF file with SNP data.
            pheno_path (str): Path to the phenotype file (CSV format).
            trait (str): Name of the trait to analyze.
            model (str): Path to the trained model.
            output_path (str): Output file prefix for plots and SNP indexes.
            cpu (bool): Flag to force CPU usage for computation.
            func: The name of the function to modify the output (tanh_map, logit_map, log_map)
            th: Plot resolution begin from this threshold (% Prediction)
            gm: Choose one of the four models, where a value of 0.5 is interpreted as follows: recessive = 0, dominant = 1, additive = 0.5, noHet = -1
        """
        start_time = time.time()
        modelPath = json_get("model_name") 
        modelPath = modelPath if model is None else model
        json_update("current_command", 'run')
        LOGGER_FILE = "run"
        os.environ['LOGGER'] = f'{LOGGER_DIR}/{LOGGER_FILE}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
        Logger(f'Message:', f"{os.environ['LOGGER']}").debug(f"Running GWANN analysis with VCF: {vcf}, phenotype path: {pheno_path}, trait: {trait}, model: {modelPath}, output path: {output_path}, threshold: {th}, function: {func}")
        Run(vcf, pheno_path, trait, modelPath, output_path, func, th).start()
        log_resource_usage(start_time,Logger(f'Message:', f"{os.environ['LOGGER']}"),"Run")
        pass


    @staticmethod
    @click.command()
    @click.option('-p', '--number-of-snps', 'pop', required=True, type=int, help="Number of SNPs in each simulation")
    @click.option('-P', '--number-of-subpopulations', 'subpop', required=True, type=int, help="Number of expected subpopulations")
    @click.option('-s', '--samples', 'n_samples', required=True, type=int, help="Number of individuals")
    @click.option('-n', '--number-of-simulation', 'n_sim', required=True, type=int, help="Number of populations to be simulated")
    @click.option('-S', '--causal-snps', 'n_snps', default=1, type=int, help="Number of causal SNPs expected per number of SNP-sites")
    @click.option('-m', '--maf', 'maf', default=0.05, type=float, help="Minor allele frequency")
    @click.option('--miss', 'miss', default=0.03, type=float, help="Proportion of missing data")
    @click.option('--equal-variance', 'equal', default=False, is_flag=True, help="Set this if equal variance is expected among SNPs (ignore for single SNP)")
    @click.option('--verbose', 'debug', default=False, is_flag=True, help="Increase verbosity")
    @click.option('--delete', 'delete', default=True, is_flag=True, help="Delete the current simulated files")
    def simulate(
        pop: int,
        subpop: int,
        n_samples: int,
        n_sim: int, 
        n_snps: int, 
        maf: float, 
        miss: float, 
        equal: bool, 
        debug: bool,
        delete: bool
        )-> None:
        """Simulate genetic data for GWANN analysis.

        This command simulates genetic data based on the specified parameters such as 
        the number of SNPs, subpopulations, individuals, and causal SNPs. The simulation 
        is performed to generate synthetic data for model training or analysis.
        
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
        start_time = time.time()
        LOGGER_FILE = "simulate"
        os.environ['LOGGER'] = f'{LOGGER_DIR}/{LOGGER_FILE}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
        Logger(f'Message:', f"{os.environ['LOGGER']}").debug(f"Simulating {n_sim} populations with {pop} SNPs, {subpop} subpopulations, {n_samples} samples, {n_snps} causal SNPs, MAF: {maf}, missing data: {miss}, equal variance: {equal}")
        Simulate(pop, subpop, n_samples, n_sim, n_snps, maf, miss, equal, debug, delete).simulate()
        log_resource_usage(start_time,Logger(f'Message:', f"{os.environ['LOGGER']}"), "Simulate")


    @staticmethod
    @click.command()
    @click.option('-M', '--MN', 'model_name', required=True, type=str, help="Model name to be saved")
    @click.option('-e', '--epochs', 'epochs', default=100, type=int, help="Number of training iterations")
    @click.option('-S', '--SNPs', 'n_snps', required=True, type=int, help="Number of SNP sites to be randomly sampled per batch")
    @click.option('-b', '--batch', 'batch', default=64, type=int, help="Batch size")
    @click.option('-l', '--lrate', 'lr', default=0.01, type=float, help="learning rate for the model")
    @click.option('-r', '--ratio', 'ratio', default=0.8, type=float, help="Train/Test ratio")
    @click.option('-w', '--width', 'width', default=15, type=int, help="Image width must be a divisor of the number of individuals")
    @click.option('--path', 'sim_path', required=True, type=str, help="Path to the simulated data")
    @click.option('--mds', 'mds', default=False,is_flag=True, type=bool, help="Apply mds transformation on the phenotype matrix, add TN to avoid population structure")
    def train(
        model_name: str,
        epochs: int, 
        n_snps: int, 
        batch: int, 
        ratio: float, 
        width: int,
        lr: float,
        sim_path: str,
        mds,
    ) -> None:
        """Train the model for GWANN analysis.

        This command trains a model using the provided simulated genetic data. The 
        model is trained for the specified number of epochs with the given batch size 
        and training/evaluation split ratio.

        Args:
            epochs (int): Number of epochs (training iterations).
            n_snps (int): Number of SNPs to sample per batch.
            batch (int): Batch size for training.
            ratio (float): Ratio of training data to evaluation data (train/eval split).
            width (int): Image width for data processing.
            sim_path (str): Path to the directory containing the simulated data.
            debug (bool): Flag to enable verbose logging for debugging.
            deterministic (bool): Flag to ensure deterministic results for reproducibility.
            cpu (bool): Flag to force training on CPU.
        
        Returns:
            None: This function does not return any value.
        """
        start_time = time.time()
        LOGGER_FILE = "train"
        os.environ['LOGGER'] = f'{LOGGER_DIR}/{LOGGER_FILE}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
        Logger(
            f'Message:',
            f"{os.environ['LOGGER']}").debug(f"Training with {epochs} epochs, {n_snps} sampled-SNPs, batch-size: {batch}, ratio: {ratio}, width: {width}, path: {sim_path}, model-name: {model_name}, mds: {mds}, lr: {lr}")
        total_simulations = json_get(SIMULATIONS)
        Train(model_name, total_simulations, n_snps, width, batch,lr ,epochs,mds, sim_path, ratio).run()
        log_resource_usage(start_time,Logger(f'Message:', f"{os.environ['LOGGER']}"), "Train")
        

    def register_commands(self):
        """Register all CLI commands with the main CLI manager.

        This method registers the individual commands (run, simulate, and train) to 
        the CLI manager, allowing them to be called from the command line interface.
        """
        self.cli.add_command(self.run)
        self.cli.add_command(self.simulate)
        self.cli.add_command(self.train)


    def start(self):
        """Run the CLI application."""
        self.register_commands()
        self.cli()


    def memory_limit(self):
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (self.get_memory() * 1024 // 2, hard))

 
    def get_memory(self):
        with open('/proc/meminfo', 'r') as mem:
            free_memory = 0
            for i in mem:
                sline = i.split()
                if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                    free_memory += int(sline[1])
        return free_memory



# Entry point for the CLI application
if __name__ == "__main__":
    manager = CLIManager()
    manager.start()
