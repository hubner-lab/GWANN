
GENOME: coalescent-based whole genome simulator
(c) 2006-2009 Liming Liang, Goncalo Abecasis



DISCLAIMER
==========

This is version of GENOME is provided to you free of charge.  It comes with
no guarantees. If you report bugs and provide helpful comments, the next
version will be better.


DISTRIBUTION
============

The latest version of GENOME is available at:

   http://www.sph.umich.edu/csg/liang/genome

If you use GENOME please register by e-mailing lianglim@umich.edu. 

Code for GENOME is subject to specific
license conditions -- please see file LICENSE.GENOME for additional
information. 

Code for the Mersenne Twister random number generator is subject to specific
license conditions -- please see file LICENSE.twister for additional
information.




BRIEF INSTRUCTIONS
==================

A complete manual is available at:

     http://www.sph.umich.edu/csg/liang/genome


Parameters and default values are listed below:

      -pop       number of subpopulations and size of subsamples [ 2 10 10 ] 
      -N         effective size of each subpopulation or the filename for population profile [10000] 
      -c         number of independent regions (chromosomes) to simulate [1] 
      -pieces    number of fragments per independent region [100] 
      -len       length in base of each fragment [10000] 
      -s         fixed number of SNPs per independent region (chromosome), -1 = number of SNPs follows Poisson distribution [-1] 
      -rec       recombination rate bewteen consecutive fragments or the filename for recombination rate distribution [0.0001] 
      -mut       mutation rate per generation per base pair [1e-08] 
      -mig       migration rate per generation per individual [0.00025] 
      -seed      random seed, -1 = use time as the seed [-1] 
      -tree      1=draw the genealogy trees, 0=do not output [0] 
      -maf       Output SNPs with minor allele frequence greater than [0] 
      -prop      To keep this proportion of SNPs with MAF < the value of -maf parameter [1] 



An example command line is:

     genome -pop 2 3 5 -N 100 -tree 1

This command will simulate 3 and 5 sequences from two populations respectively.
Each population is of size 100. Other parameters will take the default values listed in "[]".
The genealogy trees in Newick format will be output.

recombination.txt contains an example of the recombination rates profile.
population.txt contains an example of the population history profile. 
For detail of these two files please refer to the -N and -rec parameters in the manual.


REFERENCE
=========
Liang L., Zollner S., Abecasis G.R. (2007) GENOME: a rapid coalescent-based whole genome simulator.
Bioinformatics 23(12):1565-7



Comments and suggestions are welcome!

Liming Liang
lianglim@umich.edu
