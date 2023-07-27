import argparse

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

import os
import random

# Plotting
import matplotlib.pyplot as plt

# Debugging
import ipdb
from tqdm import tqdm
from functools import partial, partialmethod



def main(args):

    # Set the random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Disable TQDM progress bar if user does not explicitly request it
    tqdm.pandas()
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.enable_progress_bar)

    # Load DF and remove missing values, then extract out the features (P-values, BETA effect sizes)
    # TODO: LOG10P instead of p-value?
    # Note this is a 2D mixture, not a 1D mixture (which is different from the Bayesian project you did)
    gwas_ss_df = pd.read_csv(args.input, sep='\t').dropna().reset_index(drop=True)
    gwas_ss_df_features = gwas_ss_df[['BETA', 'P']].abs().to_numpy()


    # Note that this model fit is done for every chromosome, not each chromosome separately
    print("Calculating Bayesian Gaussian Mixture parameters now...\n")


    # Define BGM model and fit
    estim = BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_distribution",
        n_components=2,
        init_params="k-means++",
        random_state=args.seed,
        n_init=3,
        max_iter=100,
        verbose=args.verbose_fit,
        verbose_interval=10,
        # weight_concentration_prior=1e-9,
        # mean_precision_prior=1e-9
        )

    estim.fit(gwas_ss_df_features)


    # Print the weights of each distribution and their respective means
    print("\nWeights")
    print(estim.weights_)

    if(estim.weights_[0] >= estim.weights_[1]):
        majority_comp = 0
    else:
        majority_comp = 1

    print("\nMeans")
    print(estim.means_)

    print("(Column 1 is BETA, Column 2 is P)\n")


    # TODO: Plot the components and their distributions on top of the 2D plot of the P and BETA values
    # Use matplotlib (and sklearn - 'plot_ellipses' from here: https://scikit-learn.org/stable/auto_examples/mixture/plot_concentration_prior.html)


    # Use built-in (hidden) sklearn function to avoid imprecision issues
    # Note that this log is the NATURAL LOG, not LOG10. 
    # The overall final scaling of Bayesian factors is only minorly different (base e) 
    # and can be easily changed using a change of bases if needed.
    _, proba_mix = estim._estimate_log_prob_resp(gwas_ss_df_features)

    if(majority_comp == 1):
        proba_df = pd.DataFrame(proba_mix, columns=['minor_comp', 'major_comp'])
    else:
        proba_df = pd.DataFrame(proba_mix, columns=['major_comp', 'minor_comp'])

    gwas_ss_df = gwas_ss_df.join(proba_df)

    # And then add a column indicating that all the variants here are "real" (not used now)
    #gwas_ss_df['actual_variant'] = 1


    # TODO: Develop sliding/rolling window function based on the position of the variants
    # Need to do on a per-chromosome basis
    # Then for every position, compute the Bayes factor of before and after that position values

    if(args.chr == -1):
        # Do all chromosomes
        unique_chr = gwas_ss_df['CHR'].unique()

    else:
        # Do selected chromosome
        unique_chr = [args.chr]


    for chr_num in unique_chr:
        print(f"Computing values for chromosome {chr_num}...")
        chr_select_df = gwas_ss_df[gwas_ss_df['CHR'] == chr_num]

        BF_calcs = compute_BF_for_chr(chr_select_df, args.window)

        print(f"\tPlotting positional ratios for chromosome {chr_num}...")

        ax = plot_ratios(chr_select_df, BF_calcs, args.window, args.sig_thresh, args.ratio_cutoff)


    ipdb.set_trace()

    

# Plotting function
def plot_ratios(data_df,BF_calcs,window,sig_thresh,ratio_cutoff):
    BF_calcs = BF_calcs.reindex(range(BF_calcs.index[0], BF_calcs.index[-1]+1), fill_value=0)

    #STUB/TODO: not complete for all chromosomes yet
    # Plot the data
    ax = BF_calcs.plot(color='b')


    # Add highlighting based on percentile cutoffs and significant SNPs:

    # GWAS significant SNPs
    sig_snps = data_df[data_df['P'] <= sig_thresh]
    if(not sig_snps.empty):
        #sig_pos = get_relevant_indices(sig_snps['POS'], window=args.window)
        plt.fill_between(BF_calcs.index, y1=BF_calcs.min(), y2=BF_calcs.max(), where=BF_calcs.index.isin(sig_snps['POS']), color='g', alpha=0.2)
    else:
        print(f"\t\tNo significant variants found with a cutoff of {sig_thresh}...")


    # Mark regions with ratios within the interval indicated by user
    int_low = (1. - args.ratio_cutoff)/2.
    int_high = 1 - int_low
    low_perc, high_perc = BF_calcs.quantile([int_low, int_high])


    # Perform highlights of nth percentile cutoffs using fill_between
    plt.fill_between(BF_calcs.index, y1=BF_calcs.min(), y2=BF_calcs.max(), where=((BF_calcs <= low_perc) | (BF_calcs >= high_perc)), color='r', alpha=0.1)
    
    plt.show()

    return ax


# Chromosome-specific computation function
def compute_BF_for_chr(df,window):
    # Turn the position into the index (for faster conditional logic)
    df = df.set_index('POS')

    BF_calcs = roll_upper_low_BF(df, window)

    return BF_calcs


# Based on responses from and modified for our purposes
# https://stackoverflow.com/questions/14300768/pandas-rolling-computation-with-window-based-on-values-instead-of-counts
def roll_upper_low_BF(df,window):
    # Get all positions within a window of an actual variant
    # (all unique positions like this to loop over instead of every position as it is currently)
    print("\t\tGetting relevant indices...")
    index_inds = get_relevant_indices(df.index, window=window)

    curr_high_indexer = df.index.slice_indexer(0,0,1)
    curr_low_indexer = df.index.slice_indexer(0,0,1)
    curr_BF = 0

    def applyToWindow_calcBFRatio(val):
        # Get indices associated with window above and window below (the position itself is included in the higher window)
        high_indexer = df.index.slice_indexer(val,val+window,1)
        low_indexer = df.index.slice_indexer(val-window-1,val-1,1)

        # Check if either side is empty - if so, then the value is meaningless (no ratio to create)
        if((low_indexer.stop == low_indexer.start) or (high_indexer.stop == high_indexer.start)):
            return 0


        # Define as nonlocal to be able to access the outer-scope variables
        # Otherwise Python will assume these are local variables inside this function scope
        nonlocal curr_high_indexer
        nonlocal curr_low_indexer
        nonlocal curr_BF

        # Check if the last set of indexers is the same as this one (common when variants are in isolation)
        # if so, then no need to recompute - just return the last BF again
        if((high_indexer == curr_high_indexer) & (low_indexer == curr_low_indexer)):
            return curr_BF
        else:
            curr_high_indexer = high_indexer
            curr_low_indexer = low_indexer


        # Get ratio of the product of major_comp of upper to product of major_comp of lower
        # We want to scale by the number of variants, so we take the nth root
        # Note that in log scale, this is equivalent to taking the average of the log probabilities and then subtracting
        high_logprob = df[high_indexer]['major_comp'].mean()
        low_logprob = df[low_indexer]['major_comp'].mean()

        bayes_factor_log = low_logprob - high_logprob

        # Update existing BF value for check above
        curr_BF = bayes_factor_log

        return bayes_factor_log


    print(f"\t\tComputing Bayes factors for {len(index_inds)} positions...")
    rolled = index_inds.progress_apply(applyToWindow_calcBFRatio)
    return rolled



# Function to only get positions that will actually be relevant when indexing for the Bayes factor (within one window of any position)
def get_relevant_indices(all_ind,window, keep_between=True):

    # Using a max approach (aka basic math) since every position should be in ascending sequence anyways
    # Basically, get the max of the current list and then use a max comparison of that and the lower bound for arange
    # This allows us to preallocate a list of a size and then concatenate all the resulting np arrays together at the end
    arr_list = [None] * len(all_ind)

    curr_max = 0

    for l_ind,ind in enumerate(tqdm(all_ind)):
        lb = max(curr_max+1, ind-window-1)
        ub = ind+window+1
        if(keep_between):
            arr_list[l_ind] = np.arange(lb, ub)
        else:
            arr_list[l_ind] = np.array([lb, ub])
        curr_max = ub-1

    index_set_list = np.concatenate(arr_list).tolist()

    return pd.Series(index_set_list, index=index_set_list)


    # Old method that made use of sets - not very efficient
    # Much slower than the current method of using basic math to realize that computing nonoverlapping sets is easy

    # index_set = set()

    # for ind in tqdm(all_ind):
    #     index_set.update(set(np.arange(ind-window-1, ind+window+1).tolist()))

    # index_set_list = list(index_set)

    # return pd.Series(index_set_list, index=index_set_list)


    # Alternative using np.unique and concatenate
    # This is generally about 20% of the speed as using sets and 0.01% of the speed of using the current method

    # index_set = np.array([])

    # for ind in tqdm(all_ind):
    #     index_set = np.unique(np.concatenate((np.arange(ind-window-1, ind+window+1), index_set)))

    # index_set_list = index_set.tolist()

    # return pd.Series(index_set_list, index=index_set_list)





# Argument parsing for proper main call
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load the input data and perform the Bayesian mixture model analysis on it')


    # Input arguments

    parser.add_argument("--input", type=str, 
            default='./data/GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_ALL',
            help="Input GWAS summary stats")

    parser.add_argument("--window", type=int,
            default=100000,
            help="Window of positions to calculate the Bayes factor for."
            "We use a value of 100kb because generally this works well. "
            "Note, smaller loci (especially adjacent ones) may require smaller windows to detect."
            "Larger window sizes will require substantially more computation time.")

    parser.add_argument("--chr", type=int,
            default=-1,
            help="Chromosome to evaluate in this run."
            "A value of -1 (which is the default) will run all chromosomes.")

    parser.add_argument("--sig-thresh", type=float,
            default=5e-8,
            help="Significance threshold (p-value) to use on GWAS SNPs (for highlighting the plot). "
            "Note this will highlight windows that include significant variants below this threshold.")

    parser.add_argument("--ratio-cutoff", type=float,
            default=0.99,
            help="The percentile interval that ratios need to fall outside of to be highlighted. Default is 0.99."
            "Example: 0.95 means that only the ratios outside of [0.025, 0.975] will be highlighted. "
            "Note this will highlight positions where the ratio is outside of this interval.")


    # Output arguments

    parser.add_argument("--out-dir", type=str, 
            default='./results/',
            help="Output directory.")


    # Miscellaneous arguments (plotting, etc.)

    parser.add_argument("--enable-progress-bar", action='store_true', 
            help="Enable the TQDM progress bar. "
            "Note that TQDM does add some overhead, so disabling it is better on headless cluster systems. "
            "But this can be enabled for debugging/monitoring progress.")

    parser.add_argument("--verbose-fit", action='store_true', 
            help="Enables verbosity on the Bayesian Gaussian mixture fitting process.")

    parser.add_argument("--seed", type=int,
            default=9,
            help="Random seed for training and evaluation.")







    # Parse arguments and handle conflicts/other issues with arguments
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)


    print(args)
    print()
    print("Running Bayesian mixture model on provided GWAS data...")
    main(args)

