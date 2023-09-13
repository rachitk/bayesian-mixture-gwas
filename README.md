# SABER (Statistical Analysis for Bayesian Estimation of Regions)

This repository contains all of the code needed to run SABER (Statistical Analysis for Bayesian Estimation of Regions), which is a tool that is designed to take in GWAS summary statistics and identify the bounds of genomic regions of interest. 

It does so by computing ratios for relevant genomic positions that represent the likelihood that the position is the lower or upper bound of a genomic region (locus) of interest.

## Installation

To install all the required dependencies, you can clone this repository anywhere that you would like. Afterwards, run the following (ideally inside of a virtual/Conda environment):

```pip install -r requirements.txt```

to install the required dependencies. 

If you want to have a progress bar, you can also run 

```pip install -r optional_requirements.txt```

to install TQDM (though this is not required). 

## Execution

To run the tool, you simply need to run 

```python bmm_code.py {ARGUMENTS}```

where {ARGUMENTS} are as described below:

```
--window: default 100KB. Number of bases below and above each position to look for variants.
--chr: default -1 (all chromosomes). Specify the chromosome to analyze, if not all chromosomes (-1).
--sig-thresh: default 5e-8. Threshold for identifying "significant" variants for plotting and filtered fitting.
--ratio-cutoff: default 90%. Threshold outside which ratios will be highlighted in the plots.
--[beta/p/chr/pos]-col: default is [BETA/P/CHR/POS], respectively. Define column-specific headers. Useful if using summary stats from a variety of tools. 
--out-dir: default "./results". Directory where results should be saved.
```

There are additional parameters made available that may be useful to a user who wants to specify more specific behavior:
```
--no-filtered-fit: flag (no value needed, just the argument) to disable filtering of significant and downsampling of insignificant variants before the BMM fit.
--ratio-regularization: default 1e-8. This is the value used to regularize the ratio computations to prevent precision issues.
--ratio-thresh: default 1e-5. Only positions around variants that have a p-value below this threshold will be computed (drastically saves computation time). Positions not included by this method will be given a default value of 0.
--out-per-sig: flag (no value needed, just the argument) to force the output of plots around every variant that meets the significance threshold. 
--out-per-ratio-thresh: flag (no value needed, just the argument) to force the output of plots around every variant that meets the ratio threshold. 
--seed: default 9. A value that allows the user to set the deterministic random seed of the algorithm (which affects the variant filtering and BMM fitting). Note this is set to 9 by default, so running the algorithm twice without passing this flag should produce the same results on the same environment.
```