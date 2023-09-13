import argparse

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

import os
import random

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# Progress bar
try:
    from tqdm import tqdm
    tqdm_avail = True
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator
    tqdm_avail = False

from functools import partialmethod


def main(args):

    # Set the random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Disable TQDM progress bar if user does not explicitly request it and is available
    if(tqdm_avail):
        tqdm.pandas()
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.enable_progress_bar)

    # Load DF and remove missing values, then extract out the features (P-values, BETA effect sizes)
    # Note this is a 2D mixture, not a 1D mixture (slightly different from expected)
    print(f"Reading in data from {args.input}")
    gwas_ss_df = pd.read_csv(args.input, sep='\t').dropna(axis=1, how='all').dropna().reset_index(drop=True)
    print("\tCompleted reading in data")


    gwas_ss_df = gwas_ss_df.rename(columns={args.beta_col: 'BETA', 
        args.p_col: 'P', 
        args.chr_col: 'CHR', 
        args.pos_col: 'POS'})
    
    print("Sorting data by chromosome and position (to improve efficiency)...")
    gwas_ss_df = gwas_ss_df.sort_values(by=['CHR', 'POS'])

    print("Checking for multiallelic variants (will drop any multiallelic variants)")
    n_multivar = (gwas_ss_df.groupby(['CHR', 'POS']).size() > 1).sum()

    if(n_multivar > 0):
        print(f"\tFound {n_multivar} multiallelic variants... Will drop all such variants.\n")
        gwas_ss_df = gwas_ss_df.drop_duplicates(subset=['CHR', 'POS'], keep=False)
    else:
        print(f"\tFound 0 multiallelic variants... Continuing with original data.\n")

    #gwas_ss_df['LOG10P'] = np.log10(gwas_ss_df['P'])
    gwas_ss_df_features_raw = gwas_ss_df[['BETA', 'P']].abs().to_numpy()


    if(not args.no_filtered_fit):
        # Use a filtered subset of the data to fit
        # TODO: Consider filtering so the betas of the sampled insignificant variants
        # also needs to be smaller than the median/mean of the significant ones
        # to try to enforce the L-shaped ideal and also prevent skewing
        gwas_ss_df['_is_sig'] = gwas_ss_df['P'] <= args.sig_thresh
        n_sig = gwas_ss_df['_is_sig'].sum()

        gwas_ss_df_fitdata = gwas_ss_df[gwas_ss_df['_is_sig']]
        gwas_ss_df_fitdata = pd.concat([gwas_ss_df_fitdata, gwas_ss_df[~gwas_ss_df['_is_sig']].sample(n=n_sig, random_state=args.seed)])

        gwas_ss_df_features_fit = gwas_ss_df_fitdata[['BETA', 'P']].abs().to_numpy()

    else:
        # Use all the data to fit
        gwas_ss_df_features_fit = gwas_ss_df_features_raw



    # Note that this model fit is done for every chromosome, not each chromosome separately
    print("Calculating Bayesian Gaussian Mixture parameters now...\n")


    # Define BGM model and fit
    estim = BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_process",
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

    estim.fit(gwas_ss_df_features_fit)


    # Print the weights of each distribution (if using filtered fit, this should be close to 50/50) and their respective means
    print("\nWeights of components (in the data used to fit)")
    print(estim.weights_)

    print("\nMeans of components in passed data")
    print(estim.means_.T)

    print("(Row 1 is BETA, Row 2 is P)")
    print("(Col 1 is Comp 1, Col 2 is Comp 2)\n")


    # Create scatter plot of BETA and P as xy to show "clusters"
    print("Plotting the BMM fit against all of the data...")
    scatter_ax = plot_BMM_results(estim, gwas_ss_df['BETA'].abs().to_numpy(), gwas_ss_df['P'].abs().to_numpy(), save_loc=args.out_dir)
    print("\t Finished plotting the BMM component fits.\n")

    plt.close()


    # Use built-in (hidden) sklearn function to avoid imprecision issues
    # Note that this log is the NATURAL LOG, not LOG10. 
    # The overall final scaling of Bayesian factors is only minorly different (base e) 
    # and can be easily changed using a change of bases if needed.
    _, proba_mix = estim._estimate_log_prob_resp(gwas_ss_df_features_raw)

    # Determine which component represents the majority of the data (significant variants extremely likely to be the minority)
    # If the mean of the argmax is leq 0.5, then component 0 is the majority class; otherwise component 1 is the majority class
    if(proba_mix.argmax(axis=1).mean() <= 0.5):
        majority_comp = 0
    else:
        majority_comp = 1


    if(majority_comp == 1):
        print("Using component 2 (out of 2) as the 'majority' component.\n")
        proba_df = pd.DataFrame(proba_mix, columns=['minor_comp', 'major_comp'])
    else:
        print("Using component 1 (out of 2) as the 'majority' component.\n")
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

        out_loc = os.path.join(args.out_dir)
        chr_loc = os.path.join(args.out_dir, f'chr{chr_num}')
        os.makedirs(chr_loc, exist_ok=True)
        csv_save_loc = os.path.join(chr_loc, f'BF_values_chr{chr_num}.csv')

        preexist_file = os.path.isfile(csv_save_loc)

        if(preexist_file and not args.force_recompute):
            print(f"\tFound {csv_save_loc} and --force-recompute not passed. Will use the preexisting calculations to plot!")
            # Need to do some strange hackery to read this as a Series and not a DF
            BF_calcs = pd.read_csv(csv_save_loc, index_col=0, header=None)[1]
        else:
            print(f"\tEither {csv_save_loc} not found or --force-recompute was passed. Will recompute BF values...")
            BF_calcs = roll_upper_low_BF(chr_select_df, args.window, args.ratio_regularization, args.ratio_thresh)

        if((not preexist_file) or (args.force_recompute)):
            BF_calcs.to_csv(csv_save_loc, header=False)

        print(f"\tPlotting positional ratios for chromosome {chr_num}...")
        ax = plot_ratios(chr_select_df, BF_calcs, args.window, args.sig_thresh, args.ratio_thresh, args.ratio_cutoff, show_plot=False, save_name=f'chr{chr_num}', save_loc=out_loc, per_sig=args.out_per_sig, per_rthresh=args.out_per_ratio_thresh)

        print(f"Done with chromosome {chr_num}!\n")



# Plotting functions for BMM fit
# Taken from https://scikit-learn.org/stable/auto_examples/mixture/plot_concentration_prior.html
def plot_ellipses(ax, weights, means, covars):
    for n in range(means.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(
            means[n], eig_vals[0], eig_vals[1], angle=180 + angle, edgecolor="black"
        )
        ell.set_clip_box(ax.bbox)
        #ell.set_alpha(weights[n])
        ell.set_alpha(0.5)
        #ell.set_facecolor("#56B4E9")
        ell.set_facecolor("#f4dea6")
        ax.add_artist(ell)

# Plotting functions for BMM fit
# Taken from https://scikit-learn.org/stable/auto_examples/mixture/plot_concentration_prior.html
def plot_BMM_results(estimator, data_x, data_y, save_loc=None):
    plt.scatter(data_x, data_y, s=0.1, marker=".", alpha=1, linewidths=0)
    ax1 = plt.gca()
    ax1.set_ylim(-0.1, 1.1)
    plot_ellipses(ax1, estimator.weights_, estimator.means_, estimator.covariances_)

    ax1.set_xlabel("ABS(BETA)")
    ax1.set_ylabel("P")

    if(save_loc is not None):
        plt.savefig(os.path.join(save_loc, f'fitted_bmm.png'), dpi=600)

    return ax1


# Plotting function of ratios
def plot_ratios(data_df,BF_calcs,window,sig_thresh,ratio_thresh,ratio_cutoff, show_plot=True, save_name=None, save_loc=None, per_sig=False, per_rthresh=False):
    is_emptydata = BF_calcs.isnull().all()

    BF_calcs = BF_calcs.reindex(range(BF_calcs.index[0], BF_calcs.index[-1]+1), fill_value=0)

    BF_calcs_noinf = BF_calcs.replace([np.inf, -np.inf], np.nan)

    BF_calcs_min = BF_calcs_noinf.min()
    BF_calcs_max = BF_calcs_noinf.max()

    BF_calcs = BF_calcs.replace([np.inf], BF_calcs_max+1)
    BF_calcs = BF_calcs.replace([-np.inf], BF_calcs_min-1)

    #STUB/TODO: not complete for all chromosomes yet
    # Plot the data
    ax = BF_calcs.plot(color='b')

    plt.axhline(y=0, color='c', linestyle='dotted')


    # Add highlighting based on percentile cutoffs and significant SNPs:

    # GWAS significant SNPs
    sig_snps = data_df[data_df['P'] <= sig_thresh]
    if(not sig_snps.empty):
        #sig_pos = get_relevant_indices(sig_snps['POS'], window=args.window)
        plt.fill_between(BF_calcs.index, y1=BF_calcs.min(), y2=BF_calcs.max(), where=BF_calcs.index.isin(sig_snps['POS']), color='g', alpha=0.1)
    else:
        print(f"\t\tNo significant variants found with a cutoff of {sig_thresh}...")

    # Variants that meet the threshold for ratio computation
    ratio_snps = data_df[data_df['P'] <= ratio_thresh]
    if(not ratio_snps.empty):
        #sig_pos = get_relevant_indices(sig_snps['POS'], window=args.window)
        plt.fill_between(BF_calcs.index, y1=BF_calcs.min(), y2=BF_calcs.max(), where=BF_calcs.index.isin(ratio_snps['POS']), color='y', alpha=0.1)
    else:
        print(f"\t\tNo other variants found with a cutoff of {ratio_thresh}...")

    # Mark regions with ratios within the interval indicated by user
    # Only consider nonzero values when computing the quantiles (the zeros are filled in)
    int_low = (1. - ratio_cutoff)/2.
    int_high = 1 - int_low
    BF_calcs_onlynonzero = BF_calcs.replace(0, np.nan)
    low_perc, high_perc = BF_calcs_onlynonzero.quantile([int_low, int_high])

    # If this was an empty series - don't need to draw lines
    if(not (is_emptydata or np.isnan(low_perc) or np.isnan(high_perc))):
        # Perform highlights of nth percentile cutoffs using fill_between
        plt.fill_between(BF_calcs.index, y1=BF_calcs.min(), y2=BF_calcs.max(), where=((BF_calcs <= low_perc) | (BF_calcs >= high_perc)), color='r', alpha=0.05)
    
    if((save_loc is not None) and (save_name is not None)):
        plt.savefig(os.path.join(save_loc, f'{save_name}.png'), dpi=600)

        # TODO: refactor both into a single function since the underlying logic is basically the same
        # Just need to point to and iterate over a different dataframe

        # Per significant SNP region saving
        if(per_sig and not sig_snps.empty):
            print(f"\t\tSaving figures for nonoverlapping windows of significant variants (cutoff {sig_thresh})...")
            persig_dir = os.path.join(save_loc, save_name, 'sigSNPwindows')
            os.makedirs(persig_dir, exist_ok=True)

            sig_regions = get_relevant_indices(sig_snps['POS'].sort_values(), window, keep_between=False)
            sig_regions = sig_regions.drop_duplicates(keep=False)

            for sig_pos_ind in tqdm(range(0, len(sig_regions), 2)):
                plt.xlim([sig_regions.iloc[sig_pos_ind]-5*window, sig_regions.iloc[sig_pos_ind+1]+5*window])
                plt.savefig(os.path.join(persig_dir, f'snp_window_{sig_regions.iloc[sig_pos_ind]}_{sig_regions.iloc[sig_pos_ind+1]}.png'), dpi=600)

        # Per ratio SNP region saving
        if(per_rthresh and not ratio_snps.empty):
            print(f"\t\tSaving figures for nonoverlapping windows of variants that met the threshold for ratios (cutoff {ratio_thresh})...")
            perrthresh_dir = os.path.join(save_loc, save_name, 'threshratioSNPwindows')
            os.makedirs(perrthresh_dir, exist_ok=True)

            ratio_regions = get_relevant_indices(ratio_snps['POS'].sort_values(), window, keep_between=False)
            ratio_regions = ratio_regions.drop_duplicates(keep=False)

            for ratio_pos_ind in tqdm(range(0, len(ratio_regions), 2)):
                plt.xlim([ratio_regions.iloc[ratio_pos_ind]-5*window, ratio_regions.iloc[ratio_pos_ind+1]+5*window])
                plt.savefig(os.path.join(perrthresh_dir, f'snp_window_{ratio_regions.iloc[ratio_pos_ind]}_{ratio_regions.iloc[ratio_pos_ind+1]}.png'), dpi=600)

    if(show_plot):
        plt.show()
    else:
        plt.close()

    return ax



# Based on responses from and modified for our purposes
# https://stackoverflow.com/questions/14300768/pandas-rolling-computation-with-window-based-on-values-instead-of-counts
def roll_upper_low_BF(df,window, reg=1e-8, index_pthresh=1):
    # Turn the position into the index (for faster logic)
    df = df.sort_values(by='POS').set_index('POS')

    # Get all positions within a window of an actual variant
    # (all unique positions like this to loop over instead of every position as it is currently)
    # User decides whether to only do this around "significant variants" or not
    print("\tGetting relevant indices...")
    if(index_pthresh < 1.0):
        print(f"\t\tWill only compute BF around variants with p-values <= {index_pthresh}")
        thresh_df = df[df['P'] <= index_pthresh]
        if(thresh_df.empty):
            print(f"\t\tNo variants below the threshold {index_pthresh} found! Returning an empty set of ratios...")
            return pd.Series(data=[np.nan, np.nan], index=[df.index[0]-1, df.index[-1]+1])

        index_inds = get_relevant_indices(df[df['P'] <= index_pthresh].index, window=window, keep_between=True)
    else:
        index_inds = get_relevant_indices(df.index, window=window, keep_between=True)

    # Old, deprecated because didn't realize variants might have the same position (multiallelic?). Requires unique indices.
    curr_high_indexer = df.index.slice_indexer()
    curr_low_indexer = df.index.slice_indexer()

    # # Option if do care about multiallelics
    # curr_high_indexes = pd.Index([-1])
    # curr_low_indexes = pd.Index([-1])

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

        high_df = df[high_indexer]
        low_df = df[low_indexer]


        # Massive commented block below is old, deprecated because decided not to handle multiallelics. Does not need unique indices in POS.

        # # Get DF of above and below variants within window and their indexes
        # high_df = df.loc[(val):(val+window)]
        # low_df = df.loc[(val-window-1):(val-1)]

        # high_indexes = high_df.index
        # low_indexes = low_df.index

        # # Check if either DF is empty; if so, no ratio to create (and log(1) = 0)
        # if(high_df.empty or low_df.empty):
        #     return 0

        # # Access variables out of this scope within outer function
        # nonlocal curr_high_indexes
        # nonlocal curr_low_indexes
        # nonlocal curr_BF

        # # Check if the last set of indexes is the same as this one (common when variants are in isolation)
        # # if so, then no need to recompute - just return the last BF again
        # if((high_indexes.equals(curr_high_indexes)) & (low_indexes.equals(curr_low_indexes))):
        #     return curr_BF
        # else:
        #     curr_high_indexes = high_indexes
        #     curr_low_indexes = low_indexes



        # Get log ratio of the (1- product of major_comp of upper) to (1- product of major_comp of lower)
        # We want to scale by the number of variants, so we take the nth root
        # Note that in log scale, this is equivalent to taking the average of the log probabilities and then subtracting
        high_logprob = high_df['major_comp'].mean()
        low_logprob = low_df['major_comp'].mean()

        high_prob_atleastonesig = (1 - np.exp(high_logprob))
        low_prob_atleastonesig = (1 - np.exp(low_logprob))

        ratio_logprob = np.log( (high_prob_atleastonesig + reg) / (low_prob_atleastonesig + reg) )

        # Update existing BF value for check above
        # bayes_factor_log = low_logprob - high_logprob
        # curr_BF = bayes_factor_log
        # return bayes_factor_log

        curr_BF = ratio_logprob
        return ratio_logprob


    print(f"\tComputing ratios for {len(index_inds)} positions...")
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
            "Note this will highlight windows that include significant variants below this threshold. "
            "If --filtered-fit is passed, this will also determine the threshold used to create and downsample the filtered data for BMM fitting.")

    parser.add_argument("--no-filtered-fit", action='store_true',
            help="Will not filter the data for fitting the BMM."
            "Specifically,  if this is not passed, will use --sig-thresh (default 5e-8), filter the data to only include significant variants, "
            "and then sample an equal number of insignificant variants to include in fitting.")

    parser.add_argument("--ratio-cutoff", type=float,
            default=0.90,
            help="The percentile interval that ratios need to fall outside of to be highlighted. Default is 0.90."
            "Example: 0.95 means that only the ratios outside of [0.025, 0.975] will be highlighted. "
            "Note this will highlight positions where the ratio is outside of this interval.")

    parser.add_argument("--ratio-regularization", type=float,
            default=1e-8,
            help="A regularization value (default: 1e-8) that is applied to the ratios to avoid log(0) or log(inf).")

    parser.add_argument("--ratio-thresh", type=float, default=1e-5,
            help="Compute ratios only in windows around variants above the given threshold. "
            "Note that this can be different from the significance threshold used to fit the BMM. "
            "Default is a nominal threshold of 1e-5 (0.00001)")

    # Output arguments

    parser.add_argument("--out-dir", type=str, 
            default='./results/',
            help="Output directory.")

    parser.add_argument("--out-per-sig", action='store_true', 
            help="Outputs a windowed plot per significant SNP (based on --sig-thresh). "
            "Note that this is quite slow, but pretty useful if exploring various regions. "
            "Do note that if there are a lot of significant SNPs, this may take a long time. ")

    parser.add_argument("--out-per-ratio-thresh", action='store_true', 
            help="Outputs a windowed plot per SNP that meets the ratio threshold. "
            "Note that this is quite slow, but pretty useful if exploring various regions. "
            "Do note that if there are a lot of such SNPs, this may take a long time. ")


    # Miscellaneous arguments (plotting, etc.)

    parser.add_argument("--enable-progress-bar", action='store_true', 
            help="Enable the TQDM progress bar. "
            "Note that TQDM does add some overhead, so disabling it is better on headless cluster systems. "
            "But this can be enabled for debugging/monitoring progress.")

    parser.add_argument("--verbose-fit", action='store_true', 
            help="Enables verbosity on the Bayesian Gaussian mixture fitting process.")

    parser.add_argument("--force-recompute", action='store_true', 
            help="Will force a recomputation of the BF calculations if passed."
            "Otherwise, will look for [out-dir]/[chr]/BF_values.csv and load that instead.")

    parser.add_argument("--seed", type=int,
            default=9,
            help="Random seed for training and evaluation.")

    parser.add_argument("--beta-col", type=str,
            default='BETA',
            help="String for the header of the column with BETA values.")

    parser.add_argument("--p-col", type=str,
            default='P',
            help="String for the header of the column with P values.")

    parser.add_argument("--chr-col", type=str,
            default='CHR',
            help="String for the header of the column with chromosome values.")

    parser.add_argument("--pos-col", type=str,
            default='POS',
            help="String for the header of the column with position values.")







    # Parse arguments and handle conflicts/other issues with arguments
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)


    print(args)
    print()
    print("Running Bayesian mixture model on provided GWAS data...")
    main(args)

