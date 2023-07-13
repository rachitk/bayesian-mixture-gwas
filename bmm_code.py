import argparse

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

import os
import random

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


    # Define BGM model and fit
    estim = BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_distribution",
        n_components=2,
        init_params="k-means++",
        random_state=args.seed)

    estim.fit(gwas_ss_df_features)


    # Print the weights of each distribution and their respective means
    print("Weights")
    print(estim.weights_)

    print("\nMeans")
    print(estim.means_)

    print("(Column 1 is BETA, Column 2 is P)\n")


    # TODO: Plot the components and their distributions on top of the 2D plot of the P and BETA values
    # Use matplotlib (and sklearn - 'plot_ellipses' from here: https://scikit-learn.org/stable/auto_examples/mixture/plot_concentration_prior.html)


    print("Calculating Bayesian Gaussian Mixture parameters now...\n")

    # Get the probabilities of each sample in each distribution and add as new columns to the original DF
    proba_mix = estim.predict_proba(gwas_ss_df_features)
    proba_df = pd.DataFrame(proba_mix, columns=['prob_comp1', 'prob_comp2'])
    proba_df['prob_comp1_log'] = np.log(proba_df['prob_comp1'])

    gwas_ss_df = gwas_ss_df.join(proba_df)

    # And then add a column indicating that all the variants here are "real" (not used now)
    #gwas_ss_df['actual_variant'] = 1


    # TODO: Develop sliding/rolling window function based on the position of the variants
    # Need to do on a per-chromosome basis
    # Then for every position, compute the Bayes factor of before and after that position values

    unique_chr = gwas_ss_df['CHR'].unique()

    for chr_num in unique_chr:
        print(f"\tComputing values for chromosome {chr_num}...")
        gwas_ss_df_chr = gwas_ss_df[gwas_ss_df['CHR'] == chr_num]

        # Turn the position into the index (for faster conditional logic)
        gwas_ss_df_chr = gwas_ss_df_chr.set_index('POS')

        BF_calcs = roll_upper_low_BF(gwas_ss_df_chr, args.window)

        ipdb.set_trace()


# Based on responses from and modified for our purposes
# https://stackoverflow.com/questions/14300768/pandas-rolling-computation-with-window-based-on-values-instead-of-counts
def roll_upper_low_BF(df,window):
    # Get all positions within a window of an actual variant
    # (all unique positions like this to loop over instead of every position as it is currently)
    index_inds = get_relevant_indices(df.index, window=window)

    def applyToWindow_calcBFRatio(val):
        # Get indices associated with window above and window below (the position itself is included in the upper window)
        high_indexer = df.index.slice_indexer(val,val+window,1)
        low_indexer = df.index.slice_indexer(val-window-1,val-1,1)

        # Get ratio of the product of prob_comp1 of upper to product of prob_comp1 of lower
        # We want to scale by the number of variants, so we take the nth root
        # Note that in log scale, this is equivalent to taking the average of the log probabilities and then subtracting
        high_logprob = df[high_indexer]['prob_comp1_log'].mean()
        low_logprob = df[low_indexer]['prob_comp1_log'].mean()

        bayes_factor_log = high_logprob - low_logprob

        return bayes_factor_log

    rolled = index_inds.progress_apply(applyToWindow_calcBFRatio)
    return rolled



# Function to only get positions that will actually be relevant when indexing for the Bayes factor (within one window of any position)
def get_relevant_indices(all_ind,window):
    index_set = set()

    for ind in tqdm(all_ind):
        index_set.update(set(np.arange(ind-window-1, ind+window+1)))

    index_set_list = list(index_set)

    return pd.Series(index_set_list, index=index_set_list)





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
            help="Window of positions to calculate the Bayes factor for")


    # Output arguments

    parser.add_argument("--out-dir", type=str, 
            default='./results/',
            help="Output directory.")


    # Miscellaneous arguments (plotting, etc.)

    parser.add_argument("--enable-progress-bar", action='store_true', 
            help="Enable the TQDM progress bar. "
            "Note that TQDM does add some overhead, so disabling it is better on headless cluster systems. "
            "But this can be enabled for debugging/monitoring progress.")

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

