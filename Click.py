import click
from Simulate import Simulate
from Train import Train
from Run import Run
import resource

class CLIManager:
    def __init__(self):
        """Initialize the CLI Manager."""
        self.cli = click.Group()

    @staticmethod
    @click.command()
    @click.option('-v', '--vcf', 'vcf', required=True, help='path to the VCF file')
    @click.option('-p', '--pheno','pheno_path',required=True,help='path to the phenotype file (comma seperated csv file)')
    @click.option('-t', '--trait','trait',required=True,help='name of the trait (header in the phenotype file)')
    @click.option('--model','model',default="models/net.pt",help="path to the network model generated in the training step")
    @click.option('--output','output_path',default="results/GWAS",help="prefix of output plot and causative SNPs indexes in the VCF")
    @click.option('--cpu/;','cpu',default=False,required=False,help="force on cpu")
    def run(vcf, pheno_path, trait, model, output_path, cpu):
        """Run command for GWANN analysis."""
        Run(vcf, pheno_path, trait, model, output_path, cpu).run()

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
    def simulate(pop, subpop, n_samples, n_sim, n_snps, maf, miss, equal, debug):
        Simulate(pop, subpop, n_samples, n_sim, n_snps, maf, miss, equal, debug).simulate()

    @staticmethod
    @click.command()
    @click.option('-e', '--epochs', 'epochs', default=100, type=int, help="Number of training iterations")
    @click.option('-S', '--SNPs', 'n_snps', required=True, type=int, help="Number of SNP sites to be randomly sampled per batch")
    @click.option('-b', '--batch', 'batch', default=20, type=int, help="Batch size")
    @click.option('-r', '--ratio', 'ratio', default=0.8, type=float, help="Train / eval ratio")
    @click.option('-w', '--width', 'width', default=15, type=int, help="Image width must be a divisor of the number of individuals")
    @click.option('--path', 'sim_path', required=True, type=str, help="Path to the simulated data")
    @click.option('--verbose', 'debug', default=False, is_flag=True, help="Increase verbosity")
    @click.option('--deterministic', 'deterministic', default=False, is_flag=True, help="Set for reproducibility")
    @click.option('--cpu', 'cpu', default=False, is_flag=True, help="Force training on CPU")
    def train(epochs, n_snps, batch, ratio, width, sim_path, debug, deterministic, cpu):
        """CLI command for training."""
        Train(epochs, n_snps, batch, ratio, width, sim_path, debug, deterministic, cpu).train()
        

    def register_commands(self):
        """Register CLI commands."""
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
