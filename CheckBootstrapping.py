import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import boost_histogram as bh
import os
import boost_histogram as bh
import matplotlib.pyplot as plt
import h5py
from scipy.special import rel_entr
import matplotlib.lines as mlines

plt.style.use(hep.style.CMS)

def parse_args():
    parser = argparse.ArgumentParser(description="Perform copula analysis on Monte Carlo samples")
    parser.add_argument('-n', type=int, required=True, help="Number of entries to load")
    parser.add_argument('-d', '--data_type', type=str, choices=['train', 'val', 'full'], required=True, help="Data type (train, val, or full)")
    parser.add_argument('-g', '--legend', type=str, required=True, help="Legend name (e.g., HDamp, etc.)")
    parser.add_argument('-w', '--weight_type', type=str, choices=['CARL', 'Morphing'], required=True, help="Weight type (CARL or Morphing)")
    parser.add_argument('-f', '--filter_type', type=str, required=True, help="Filter type (filter or no_filter)")

    # Parse arguments
    args = parser.parse_args()
    return args

def load_data(n, data_type, legend, dir_label='NoBootstrapping'):
    """
    Loads data from an HDF5 file instead of .npy files.
    Returns lists of x0, w0, w_CARL, x1, and w1 arrays.
    """
    def load_hdf5_data(split):
        hdf5_file_path = f'/data/dust/user/griesinl/carl-torch/{legend}/{dir_label}/{legend}_data_{n}.h5'
        print(f"Loading {split} data from {hdf5_file_path}...")

        x0_list, w0_list, w_CARL_list, x1_list, w1_list = [], [], [], [], []

        with h5py.File(hdf5_file_path, "r") as h5f:
            group = h5f[split]
            indices = [str(i) for i in range(50) if str(i) in group]  # Now considering all 100

            for idx in indices:
                x0_list.append(group[f"{idx}/X0"][()])
                w0_list.append(group[f"{idx}/w0"][()].flatten())
                w_CARL_list.append(group[f"{idx}/w_CARL"][()].flatten())
                x1_list.append(group[f"{idx}/X1"][()])
                w1_list.append(group[f"{idx}/w1"][()].flatten())

        return x0_list, w0_list, w_CARL_list, x1_list, w1_list
    
    # Load metadata separately
    metadata_file = f"{legend}_0_metaData_{n}.pkl"
    metadata_path = f"/data/dust/user/griesinl/carl-torch/{legend}/{dir_label}/{metadata_file}"

    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    else:
        print(f"Warning: Metadata file not found - {metadata_path}")
        metadata = None  # Set to None if missing

    if data_type == 'full':
        x0_train, w0_train, w_CARL_train, x1_train, w1_train = load_hdf5_data('train')
        x0_val, w0_val, w_CARL_val, x1_val, w1_val = load_hdf5_data('val')

        # Combine train and val per index
        x0_list = [np.vstack((x0_train[i], x0_val[i])) for i in range(len(x0_train))]
        w0_list = [np.concatenate((w0_train[i], w0_val[i])) for i in range(len(w0_train))]
        w_CARL_list = [np.concatenate((w_CARL_train[i], w_CARL_val[i])) for i in range(len(w_CARL_train))]
        x1_list = [np.vstack((x1_train[i], x1_val[i])) for i in range(len(x1_train))]
        w1_list = [np.concatenate((w1_train[i], w1_val[i])) for i in range(len(w1_train))]
    else:
        x0_list, w0_list, w_CARL_list, x1_list, w1_list = load_hdf5_data(data_type)

    print(f"Loaded {len(x0_list)} x0 arrays, {len(w0_list)} w0 arrays, {len(w_CARL_list)} w_CARL arrays, {len(x1_list)} x1 arrays, and {len(w1_list)} w1 arrays.")
    print("Loading from HDF5 done!")

    return x0_list, x1_list, w0_list, w1_list, w_CARL_list, metadata

def prep_weights(w0_list, w_CARL):
    return [w0 * w_carl for w0, w_carl in zip(w0_list, w_CARL)]

def create_masks(x0_list, x1_list, filter_type):
    """
    Efficiently creates masks for all x0 and x1 arrays using NumPy vectorization.
    """
    # Stacked x0 and x1 arrays into 3D arrays (for each set of arrays)
    x0_array = np.stack(x0_list)  # Shape: (num_arrays, num_events, num_features)
    x1_array = np.stack(x1_list)  # Shape: (num_arrays, num_events, num_features)

    if filter_type == "filter":
        # Use 3D slicing to apply the filters correctly across all arrays
        default_masks = (x0_array[:, :, 2] < 400) & (x0_array[:, :, 3] > 0.2)
        variation_masks = (x1_array[:, :, 2] < 400) & (x1_array[:, :, 3] > 0.2)
        return default_masks, variation_masks
    
    elif filter_type == 'fitTopMass':
        # Apply filtering to the relevant columns for all arrays
        default_masks = (x0_array[:, :, 3] > 0.2) & (x0_array[:, :, 2] > 130) & (x0_array[:, :, 2] < 350)
        variation_masks = (x1_array[:, :, 3] > 0.2) & (x1_array[:, :, 2] > 130) & (x1_array[:, :, 2] < 350)
        return default_masks, variation_masks
    
    elif filter_type == 'Rb':
        # Apply filtering to the relevant columns for all arrays
        default_masks = (x0_array[:, :, 3] > 0.2) & (x0_array[:, :, 1] < 4)
        variation_masks = (x1_array[:, :, 3] > 0.2) & (x1_array[:, :, 1] < 4)
        return default_masks, variation_masks
    
    elif filter_type == 'Mlb':
        # Apply filtering to the relevant columns for all arrays
        default_masks = (x0_array[:, :, 3] < 0.2) & (x0_array[:, :, 0] < 300)
        variation_masks = (x1_array[:, :, 3] < 0.2) & (x1_array[:, :, 0] < 300)
        return default_masks, variation_masks
    
    elif filter_type == 'recoWMass':
        # Apply filtering to the relevant columns for all arrays
        default_masks = (x0_array[:, :, 3] > 0.2) & (x0_array[:, :, 4] > 63) & (x0_array[:, :, 4] < 110)
        variation_masks = (x1_array[:, :, 3] > 0.2) & (x1_array[:, :, 4] > 63) & (x1_array[:, :, 4] < 110)
        return default_masks, variation_masks
    
    elif filter_type == 'Mlb_over_fitTopMass':
        # Apply filtering to the relevant columns for all arrays
        default_masks = (x0_array[:, :, 3] > 0.2) & (x0_array[:, :, 5] < 1.5)
        variation_masks = (x1_array[:, :, 3] > 0.2) & (x1_array[:, :, 5] < 1.5)
        return default_masks, variation_masks

def plot_and_convert_to_1DHistograms(x0_list, x1_list, w0_list, w_mod_list, w1_list, args, metadata, output_dir, filter_type='no_filter', with_bs=False):
    """
    Create 2D histograms for Default, Modified, and Variation samples, then convert them into flattened 1D histograms and plot them.
    """
    num_observables = x0_list[0].shape[1]  # Using x0_list[0] as reference
    observable_names = list(metadata.keys())

    # Mapping of observable names to LaTeX labels
    latex_labels = {"Mlb": r"$M_{lb}$","Rb": r"$R_b$","fitTopMass": r"$m_t^{fit}$","prob": r"$P_{gof}$","recoWMass": r"$m_W^{reco}$"}
    
    for i in range(num_observables):
        for j in range(num_observables):
            print(f"{observable_names[i]} vs. {observable_names[j]}")

            default_flattened = []
            modified_flattened = []
            variation_flattened = []
            errors_default = []
            errors_modified = []
            errors_variation = []
            
            for x0, w0, x_mod, w_mod, x1, w1 in zip(x0_list, w0_list, x0_list, w_mod_list, x1_list, w1_list):
                # Compute percentile bins for the current (x0, w0)
                positive_mask = w0 > 0
                x0_filtered = x0[positive_mask]
                w0_filtered = w0[positive_mask]

                percentiles_i = np.percentile(x0_filtered[:, i], np.arange(0, 101, 10), weights=w0_filtered, method='inverted_cdf')
                percentiles_j = np.percentile(x0_filtered[:, j], np.arange(0, 101, 10), weights=w0_filtered, method='inverted_cdf')

                def compute_flattened_histogram(data, weights):
                    hist = bh.Histogram(bh.axis.Variable(percentiles_i), bh.axis.Variable(percentiles_j), storage=bh.storage.Weight())
                    hist.fill(data[:, i], data[:, j], weight=weights)
                    values_flat = np.concatenate([hist.view().T['value'][i, :] for i in range(hist.view().T.shape[0])])
                    errors_flat = np.sqrt(np.concatenate([hist.view().T['variance'][i, :] for i in range(hist.view().T.shape[0])]))
                    return values_flat, errors_flat

                # Compute histograms for Default, Modified, and Variation using the same bin edges
                values_default, errors_default_hist = compute_flattened_histogram(x0, w0)
                values_modified, errors_modified_hist = compute_flattened_histogram(x_mod, w_mod)
                values_variation, errors_variation_hist = compute_flattened_histogram(x1, w1)

                default_flattened.append(values_default)
                modified_flattened.append(values_modified)
                variation_flattened.append(values_variation)
                errors_default.append(errors_default_hist)
                errors_modified.append(errors_modified_hist)
                errors_variation.append(errors_variation_hist)

            # Compute mean and standard deviation if with_bs is True, else compute mean of errors
            if with_bs:
                default_stack = np.stack(default_flattened, axis=0)
                modified_stack = np.stack(modified_flattened, axis=0)
                variation_stack = np.stack(variation_flattened, axis=0)

                values_default_mean = np.mean(default_stack, axis=0)
                errors_default_combined = np.std(default_stack, axis=0)

                values_modified_mean = np.mean(modified_stack, axis=0)
                errors_modified_combined = np.std(modified_stack, axis=0)

                values_variation_mean = np.mean(variation_stack, axis=0)
                errors_variation_combined = np.std(variation_stack, axis=0)
            else:
                # Compute the mean of the values, but keep the errors as computed in compute_flattened_histogram
                values_default_mean = np.mean(default_flattened, axis=0)
                errors_default_combined = np.mean(errors_default, axis=0)

                values_modified_mean = np.mean(modified_flattened, axis=0)
                errors_modified_combined = np.mean(errors_modified, axis=0)

                values_variation_mean = np.mean(variation_flattened, axis=0)
                errors_variation_combined = np.mean(errors_variation, axis=0)

            # Plot histograms
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
            
             # Add CMS style text
            axs[0].text(0.01, 1.01, "Private work\n(CMS simulation)", fontsize=14, verticalalignment='bottom', transform=axs[0].transAxes)
            hep.cms.lumitext("36.3 fb$^{-1}$ (13 TeV)", ax=axs[0], fontsize=14)
            
            bin_edges = np.arange(101)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Use a colormap that goes from white to light grey with 10 distinct colors
            colormap = plt.get_cmap("Greys")
            percentile_colors = [colormap(i / 10) for i in range(1, 11)]  # Shades from white to light grey

            # Create the percentile bins for 10 regions
            percentiles_range = np.linspace(0, 100, len(percentile_colors) + 1)
            
            for k, color in enumerate(percentile_colors):
                #axs[0].axvspan(percentiles_range[k], percentiles_range[k+1], facecolor=color, alpha=0.4)
                axs[0].axvline(percentiles_range[k], color='black', linestyle='--', linewidth=0.5, alpha = 0.4)  # Add grid lines
            
            # Plot the Default histogram
            axs[0].stairs(values_default, bin_edges, label="Default", color='blue', linestyle='-', alpha=0.5, linewidth=1.5, baseline=0, zorder=1)
            axs[0].errorbar(bin_centers, values_default, yerr=errors_default_combined, fmt='o', color='blue', alpha=0.5, markersize=1.5, linestyle='None')

            # Plot the Variation histogram
            axs[0].stairs(values_variation, bin_edges, label=f"{args.legend}", color='red', linestyle='-', alpha=0.5, linewidth=1.5, baseline=0, zorder=3)
            axs[0].errorbar(bin_centers, values_variation, yerr=errors_variation_combined, fmt='o', color='red', alpha=0.5, markersize=1.5, linestyle='None')
        
            # Plot the Modified histogram
            axs[0].stairs(values_modified, bin_edges, label=f"Default * {args.weight_type}", color='orange', linestyle='-', alpha=0.5, linewidth=1.5, baseline=0, zorder=2)
            axs[0].errorbar(bin_centers, values_modified, yerr=errors_modified_combined, fmt='o', color='orange', markersize=1.5, alpha=0.5, linestyle='None')
            
            ymax = max(values_default.max(), values_variation.max(), values_modified.max()) * 1.1  # Add 20% margin
            # Find the midpoint of the x-axis
            midpoint = len(bin_centers) // 2

            # Find the maximum and minimum y-values in each half (left and right)
            left_max = np.max([np.max(values_default[:midpoint]), np.max(values_variation[:midpoint]), np.max(values_modified[:midpoint])])
            right_max = np.max([np.max(values_default[midpoint:]), np.max(values_variation[midpoint:]), np.max(values_modified[midpoint:])])
            left_min = np.min([np.min(values_default[:midpoint]), np.min(values_variation[:midpoint]), np.min(values_modified[:midpoint])])
            right_min = np.min([np.min(values_default[midpoint:]), np.min(values_variation[midpoint:]), np.min(values_modified[midpoint:])])

            # Determine the side with the maximum value
            if right_max > left_max:
                # Peak is on the right side, place the legend on the left side
                legend_side = 'left'
            else:
                # Peak is on the left side, place the legend on the right side
                legend_side = 'right'

            # Determine the vertical position of the legend
            if legend_side == 'left':
                # Check if the left side has more values in the upper or lower half
                left_half_peak = "upper" if left_max < (ymax / 2) else "lower"
                left_half_min = "upper" if left_min < (ymax / 2) else "lower"
                legend_loc = 'upper left' if left_half_peak == "upper" and left_half_min == "upper" else 'lower left'
            else:
                # Check if the right side has more values in the upper or lower half
                right_half_peak = "upper" if right_max < (ymax / 2) else "lower"
                right_half_min = "upper" if right_min < (ymax / 2) else "lower"
                legend_loc = 'upper right' if right_half_peak == "upper" and right_half_min == "upper" else 'lower right'
            
            # Add the observable combination to the legend
            handles, labels = axs[0].get_legend_handles_labels()
            if with_bs:
                title = f"{args.data_type} sample\n{latex_labels.get(observable_names[i], observable_names[i])} in {latex_labels.get(observable_names[j], observable_names[j])} bins (w/ BS)"
            else:
                title = f"{args.data_type} sample\n{latex_labels.get(observable_names[i], observable_names[i])} in {latex_labels.get(observable_names[j], observable_names[j])} bins (w/o BS)"
            legend = axs[0].legend(handles=handles, labels=labels, loc=legend_loc,
            title=title,
            title_fontsize= 12, fontsize = 10, framealpha = 1, fancybox = False, frameon = True, edgecolor = 'white')
            legend.get_title().set_ha('center')

            axs[0].set_ylim(0, ymax )  # Add space for the legend height
            axs[0].set_ylabel("Entries", loc = 'center', fontsize =14)
            axs[0].tick_params(axis='y', which='major', labelsize=12)
            axs[0].set_xlim(0, 100.)
            
            # Compute ratios
            ratio_mod_default = values_modified_mean / values_default_mean
            ratio_err_mod_default = errors_modified_combined / values_default_mean
            
            ratio_var_default = values_variation_mean / values_default_mean
            ratio_err_var_default = errors_variation_combined / values_default_mean
            
            nonzero_bins = (values_default_mean > 0)
            
           # Print the mean ratio, standard deviation, and mean error on ratio
            mean_ratio_mod_default = np.mean(ratio_mod_default[nonzero_bins])
            std_ratio_mod_default = np.std(ratio_mod_default[nonzero_bins])
            mean_error_mod_default = np.mean(ratio_err_mod_default[nonzero_bins])
            
            mean_ratio_var_default = np.mean(ratio_var_default[nonzero_bins])
            std_ratio_var_default = np.std(ratio_var_default[nonzero_bins])
            mean_error_var_default = np.mean(ratio_err_var_default[nonzero_bins])

            print(f"Mean Ratio (Modified/Default) for {observable_names[i]} vs. {observable_names[j]}: {mean_ratio_mod_default:.4f}")
            print(f"Std of Ratio (Modified/Default) for {observable_names[i]} vs. {observable_names[j]}: {std_ratio_mod_default:.4f}")
            print(f"Mean Error on Ratio (Modified/Default) for {observable_names[i]} vs. {observable_names[j]}: {mean_error_mod_default:.4f}")
            
            print(f"Mean Ratio (Variation/Default) for {observable_names[i]} vs. {observable_names[j]}: {mean_ratio_var_default:.4f}")
            print(f"Std of Ratio (Variation/Default) for {observable_names[i]} vs. {observable_names[j]}: {std_ratio_var_default:.4f}")
            print(f"Mean Error on Ratio (Variation/Default) for {observable_names[i]} vs. {observable_names[j]}: {mean_error_var_default:.4f}")
            
            
            # Calculate the largest deviation from y = 1 for both NoBootstrapping and Bootstrapping ratios
            min_ratio = min(ratio_mod_default)
            max_ratio = max(ratio_mod_default)

            # Make sure min_ratio and max_ratio are not NaN or Inf
            if np.isnan(min_ratio) or np.isnan(max_ratio) or np.isinf(min_ratio) or np.isinf(max_ratio):
                print("Warning: Invalid ratio values (NaN or Inf) detected")
                # Handle the case appropriately, for example:
                min_ratio = 1.025
                max_ratio = 1.025

            # Calculate the largest deviation from y = 1
            max_deviation = max(abs(min_ratio - 1), abs(max_ratio - 1))
            # Set the y-limits to center around 1 with some buffer
            buffer = 0.05  # adjust this based on your needs
            ymin = 1 - max_deviation - buffer
            ymax = 1 + max_deviation + buffer

            axs[1].errorbar(bin_centers[nonzero_bins], ratio_mod_default[nonzero_bins], yerr=ratio_err_mod_default[nonzero_bins], fmt='o', color='#DAA520', markersize=3, 
                            label=f"Default * {args.weight_type} (w/ BS)" if with_bs else f"Default * {args.weight_type} (w/o BS)")
            axs[1].errorbar(bin_centers[nonzero_bins], ratio_var_default[nonzero_bins], yerr=ratio_err_var_default[nonzero_bins], fmt='o', color='orangered', markersize=3, 
                            label=f"{args.legend} (w/ BS)" if with_bs else f"{args.legend} (w/o BS)")
            axs[1].axhline(1, color='black', linestyle='--')
            axs[1].set_xlabel("Bin Index", loc='right', fontsize=12)
            axs[1].set_ylabel("Ratio Sample / Default", loc = 'center', fontsize = 10)
            axs[1].set_xlim(0, 100)
            axs[1].set_ylim(ymin, ymax)
            axs[1].grid(True, which='major', axis='y', linestyle='--', color='gray', alpha=0.5)
            axs[1].tick_params(axis='y', which='major', labelsize=10)
            axs[1].tick_params(axis='y', which='minor', left=False, right = False) 
            
            axs[1].set_xticks(bin_centers[::10])  # Set ticks every 10 bins
            axs[1].set_xticklabels(np.arange(0, 100, 10))
            axs[1].tick_params(axis='x', which='major', labelsize=12)
             
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"1D_Hist_{args.weight_type}_{args.data_type}_{observable_names[i]}_{observable_names[j]}_{args.n}_{args.filter_type}.png")
            plt.savefig(plot_path, dpi = 300)
            plt.close()
                
def plot_1DHistograms(x0_list, x1_list, w0_list, w1_list, w_mod_list, args, metadata, output_dir, filter_type='no_filter', with_bs = False):
    """
    Create 1D histograms to compare Default, Modified, and Variation for all observables, using percentile bins and normalizing by bin width. 
    Plots the ratios of these histograms, considering the mean and standard deviation for each bin.
    """
    num_observables = x0_list[0].shape[1]
    observable_names = list(metadata.keys())

    for i in range(num_observables):
        all_bin_contents_default = []
        all_bin_contents_modified = []
        all_bin_contents_variation = []
        
        all_bin_errors_default = []
        all_bin_errors_modified = []
        all_bin_errors_variation = []
        
        all_ratios_mod_default = []
        all_ratios_var_def = []
        
        latex_labels_hist = {"Mlb": r"$M_{lb}$ [GeV]", "Rb": r"$R_b$", "fitTopMass": r"$m_t^{fit}$ [GeV]", "prob": r"$P_{gof}$", "recoWMass": r"$m_W^{reco}$ [GeV]"}
        
        for x0_Default, x1_Variation, w0, w1, w_mod in zip(x0_list, x1_list, w0_list, w1_list, w_mod_list):
            
            x0_Default_percentile = x0_Default[w0 > 0]
            w0_percentile = w0[w0> 0]

            # Calculate percentile bins using Default data (x0_Default)
            percentiles = np.percentile(x0_Default_percentile[:, i], np.arange(0, 101, 10), weights=w0_percentile, method='inverted_cdf')
            
            # Create histograms using boost-histogram
            bin_edges = bh.axis.Variable(percentiles)

            # Define the histograms for Default, Modified, and Variation samples
            hist_default = bh.Histogram(bin_edges, storage=bh.storage.Weight())
            hist_modified = bh.Histogram(bin_edges, storage=bh.storage.Weight())
            hist_variation = bh.Histogram(bin_edges, storage=bh.storage.Weight())

            hist_default.fill(x0_Default[:, i], weight=w0)
            hist_modified.fill(x0_Default[:, i], weight=w_mod)
            hist_variation.fill(x1_Variation[:, i], weight=w1)

            # Calculate the bin contents for the histograms
            bin_contents_default = hist_default.values()
            bin_contents_modified = hist_modified.values()
            bin_contents_variation = hist_variation.values()
            
            # Compute ratio per iteration
            ratio_mod_default = bin_contents_modified / bin_contents_default
            all_ratios_mod_default.append(ratio_mod_default)
            
            ratio_var_def = bin_contents_variation / bin_contents_default
            all_ratios_var_def.append(ratio_var_def)

            all_bin_contents_default.append(bin_contents_default)
            all_bin_contents_modified.append(bin_contents_modified)
            all_bin_contents_variation.append(bin_contents_variation)

            bin_errors_default = np.sqrt(hist_default.variances())
            bin_errors_modified = np.sqrt(hist_modified.variances())
            bin_errors_variation = np.sqrt(hist_variation.variances())
            
            all_bin_errors_default.append(bin_errors_default)
            all_bin_errors_modified.append(bin_errors_modified)
            all_bin_errors_variation.append(bin_errors_variation)
        
        if with_bs:
            default_stack = np.stack(all_bin_contents_default, axis=0)
            modified_stack = np.stack(all_bin_contents_modified, axis=0)
            variation_stack = np.stack(all_bin_contents_variation, axis=0)

            values_default_mean = np.mean(default_stack, axis=0)
            errors_default_combined = np.std(default_stack, axis=0)

            values_modified_mean = np.mean(modified_stack, axis=0)
            errors_modified_combined = np.std(modified_stack, axis=0)

            values_variation_mean = np.mean(variation_stack, axis=0)
            errors_variation_combined = np.std(variation_stack, axis=0)
        else:
            values_default_mean = np.mean(all_bin_contents_default, axis=0)
            errors_default_combined = np.mean(all_bin_errors_default, axis=0)

            values_modified_mean = np.mean(all_bin_contents_modified, axis=0)
            errors_modified_combined = np.mean(all_bin_errors_modified, axis=0)

            values_variation_mean = np.mean(all_bin_contents_variation, axis=0)
            errors_variation_combined = np.mean(all_bin_errors_variation, axis=0)
            
        # Normalize by bin width
        bin_widths = np.diff(hist_default.axes[0].edges)
        bin_heights_default = values_default_mean / bin_widths
        bin_heights_modified = values_modified_mean / bin_widths
        bin_heights_variation = values_variation_mean / bin_widths

        # Create a figure with two subplots: histogram and ratio plot
        fig, (ax_hist, ax_ratio) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Plot histograms
        ax_hist.bar(hist_default.axes[0].edges[:-1], bin_heights_default, width=bin_widths, alpha=0.5, edgecolor='black', linewidth=1.5, label="Default", color='blue', align='edge', zorder=1)
        ax_hist.bar(hist_variation.axes[0].edges[:-1], bin_heights_variation, width=bin_widths, alpha=0.5, edgecolor='black', linewidth=1.5, label=f"{args.legend}", color='red', align='edge', zorder=3)
        ax_hist.bar(hist_modified.axes[0].edges[:-1], bin_heights_modified, width=bin_widths, alpha=0.5, edgecolor='black', linewidth=1.5, label=f"Default * {args.weight_type}", color='orange', align='edge', zorder=2)
        ax_hist.set_ylim(bottom=0)

        ax_hist.text(0.01, 1.01, "Private work\n(CMS simulation)", fontsize=14, verticalalignment='bottom', transform=ax_hist.transAxes)
        hep.cms.lumitext("36.3 fb$^{-1}$ (13 TeV)", ax=ax_hist, fontsize=14)

        print(f"{observable_names[i]}")

        # Compute mean and standard deviation across iterations
        all_ratios_mod_default = np.array(all_ratios_mod_default)
        ratio_mod_default_mean = np.mean(all_ratios_mod_default, axis=0)
        ratio_mod_default_std = np.std(all_ratios_mod_default, axis=0)

        all_ratios_var_def = np.array(all_ratios_var_def)
        ratio_var_def = np.mean(all_ratios_var_def, axis = 0)
        ratio_err_var_def = np.std(all_ratios_var_def, axis = 0)
        
         # Select the bin to analyze (e.g., middle bin)
        selected_bin = 0 # Middle bin
        # Collect values for the selected bin across iterations
        default_values_bin = np.array([contents[selected_bin] for contents in all_bin_contents_default])
        modified_values_bin = np.array([contents[selected_bin] for contents in all_bin_contents_modified])
        
        # Print results
        print("\n### Debug Output for Selected Bin ###")
        print(f"Selected Bin Index: {selected_bin}")
        print(f"Default Values (50 Iterations): {default_values_bin}")
        print(f"Modified (CARL * Default) Values (50 Iterations): {modified_values_bin}")
        print(f"Mean from Default: {np.mean(default_values_bin):.5f}, Variance from Default: {np.var(default_values_bin):.5f}")
        print(f"Mean from Modified: {np.mean(modified_values_bin):.5f}, Variance from Modified: {np.var(modified_values_bin):.5f}")
        print(f"Mean Ratio: {ratio_mod_default_mean[selected_bin]}, Variance Ratio: {ratio_mod_default_std[selected_bin]:.5f}")
        
        bin_centers = (hist_default.axes[0].edges[:-1] + hist_default.axes[0].edges[1:]) / 2

        ax_ratio.errorbar(bin_centers, ratio_mod_default_mean, yerr=ratio_mod_default_std, fmt='o', color='#DAA520', markersize=3, label=f"Default * {args.weight_type}")
        ax_ratio.errorbar(bin_centers, ratio_var_def, yerr=ratio_err_var_def, fmt='o', color='orangered', markersize=3, label=f"{args.legend}")

        # Add labels and legend
        ax_ratio.set_xlabel(f"{latex_labels_hist.get(observable_names[i], observable_names[i])}", loc='right', fontsize = 14)
        ax_hist.set_ylabel("Normalized Entries", loc = 'center', fontsize = 14)
        ax_hist.tick_params(axis='both', which='major', labelsize=12)
        ax_ratio.tick_params(axis='y', which='major', labelsize=10)
        ax_ratio.tick_params(axis='x', which='major', labelsize=12)
        ax_ratio.grid(True, which='major', axis='y', linestyle='--', color='gray', alpha=0.5)
        ax_ratio.set_ylabel("Ratio Sample / Default", loc = 'center', fontsize = 10)    
        ax_ratio.axhline(1, color='black', linestyle='--')
        ax_ratio.grid(True, which='major', axis='y', linestyle='--', color='gray', alpha=0.5)
        ax_ratio.set_ylabel("Ratio Sample / Default", loc = 'center', fontsize = 10)

        # Add legend for the histogram
        handles, labels = ax_hist.get_legend_handles_labels()
        if with_bs:
            title = f"{args.data_type} sample (w/ BS)"
        else:
            title=f"{args.data_type} sample (w/o BS)"
            
        ax_hist.legend(handles=handles, labels=labels,  loc='best', title=title, title_fontsize= 12, fontsize=10)
        
        # Save the plot
        output_path = os.path.join(output_dir, f'Hist_1D_{args.weight_type}_{args.data_type}_{observable_names[i]}_{args.n}_{filter_type}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi = 300)
        plt.close()
        print(f"Saved histogram and ratio plots for {observable_names[i]} to {output_path}")
        
        # Now, save the ratio plot separately
        output_path_ratio = os.path.join(output_dir, f'Ratio_1D_{args.weight_type}_{args.data_type}_{observable_names[i]}_{args.n}_{filter_type}.png')

        fig_ratio = plt.figure(figsize=(8, 4))  # Create a new figure for the ratio plot
        ax_ratio_new = plt.gca()  # Get the current axis of the new figure

        # Place text in the top-left corner of the plot
        ax_ratio_new.text(0.01, 1.01, "Private work\n(CMS simulation)", fontsize=14, verticalalignment='bottom', transform=ax_ratio_new.transAxes)
        hep.cms.lumitext("36.3 fb$^{-1}$ (13 TeV)", ax=ax_ratio_new, fontsize=14)

        # Plot the error bars with the correct labels
        ax_ratio_new.errorbar(bin_centers, ratio_mod_default_mean, yerr=ratio_mod_default_std, fmt='o', color='#DAA520', markersize=3, label=f"Default * {args.weight_type}")
        ax_ratio_new.errorbar(bin_centers, ratio_var_def, yerr=ratio_err_var_def, fmt='o', color='orangered', markersize=3, label=f"{args.legend}")

        # Customize the ratio plot as before
        ax_ratio_new.set_xlabel(f"{latex_labels_hist.get(observable_names[i], observable_names[i])}", loc='right', fontsize=14)
        ax_ratio_new.set_ylabel("Ratio Sample / Default", loc='center', fontsize=10)
        ax_ratio_new.axhline(1, color='black', linestyle='--')
        ax_ratio_new.tick_params(axis='both', which='major', labelsize=12)
        ax_ratio_new.grid(True, which='major', axis='y', linestyle='--', color='gray', alpha=0.5)
        ax_ratio_new.tick_params(axis='y', which='major', labelsize=10)

        # Set the correct legend
        ax_ratio_new.legend(loc='lower right', title=title, title_fontsize=12, fontsize=10, frameon = True)

        # Save the ratio figure
        plt.tight_layout()
        plt.savefig(output_path_ratio, dpi=300)
        plt.close(fig_ratio)

        print(f"Saved histogram and ratio plots for {observable_names[i]} to {output_path}")
        print(f"Saved ratio plot separately to {output_path_ratio}")


def main():
    
    args = parse_args()
    
    output_dir = f'plots_NoBootstrappingV3_CMS_1D_plots/'
    os.makedirs(output_dir, exist_ok=True)
    
    x0_list, x1_list, w0_list, w1_list, w_CARL, metadata = load_data(n = args.n, data_type= args.data_type, legend= args.legend, dir_label='NoBootstrappingV3') # Run_without_bs or BootstrappedV2 
    
    # Assuming w0 and w_CARL are already loaded
    w_mod_list = prep_weights(w0_list, w_CARL)

    print("Prepared the modified event weights!")
    if args.filter_type != "no_filter":
        default_masks, variation_masks = create_masks(x0_list, x1_list, args.filter_type)

        # Apply masks correctly
        x0_list_filtered = [x0[mask] for x0, mask in zip(x0_list, default_masks)]
        w0_list_filtered = [w0[mask] for w0, mask in zip(w0_list, default_masks)]
        w_mod_list_filtered = [w_mod[mask] for w_mod, mask in zip(w_mod_list, default_masks)]
        x1_list_filtered = [x1[mask] for x1, mask in zip(x1_list, variation_masks)]
        w1_list_filtered = [w1[mask] for w1, mask in zip(w1_list, variation_masks)]
    else:
        x0_list_filtered = x0_list
        w0_list_filtered = w0_list
        w_mod_list_filtered = w_mod_list
        x1_list_filtered = x1_list
        w1_list_filtered = w1_list


    print("Data filtering completed.")
    
    plot_1DHistograms(x0_list_filtered, x1_list_filtered, w0_list_filtered, w1_list_filtered, w_mod_list_filtered, args, metadata, output_dir, filter_type=args.filter_type, with_bs = False)
    
    #plot_and_convert_to_1DHistograms(x0_list_filtered, x1_list_filtered, w0_list_filtered, w_mod_list_filtered, w1_list_filtered, 
    #                                args, metadata, output_dir, filter_type=args.filter_type, with_bs=True)
    
if __name__ == "__main__":
    main()