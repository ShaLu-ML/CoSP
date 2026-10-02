from config import XRAY_FILES

import os
from pathlib import Path
import pickle
import random
import time
from collections import defaultdict
import re
import pandas as pd
import torch
import matplotlib.gridspec as gridspec
from explanation import predict_proba, select_samples

from config import N_FREQ, N_CHN_PAIR, WORK_PATH, PAT_LIST_ALL, PAT_LIST, PAT_ID_LIST, RST_PATH, CONFIG, LIME_PATH, \
    INFO_PATH, FREQ_BANDS
from data import compute_coherence, compute_coherence_one_segment
from models import get_model
from util import get_saved_info, get_final_model_id, sigmoid, load_tst_coh, my_save
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image, lime_tabular

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable

#=========================== Explaination with LimeTabularExplainer ===============
def get_combined_feature_name():
    # generate flattened feature names from 2D shape of (n_frequency, n_channel_pair)
    channel_names = []
    for i in range(1, 17):
        for j in range(i + 1, 17):
            channel_names.append(f"{i}-{j}")
    n_chn = len(channel_names)
    freq_names = np.loadtxt(RST_PATH / f'coherence_frequencies_170.csv', delimiter=',').astype('int')
    freq_names[0] = 0.5
    n_freq = len(freq_names)

    feature_names = []
    for idx in range(n_freq * n_chn):
        freq_idx, channel_idx = np.unravel_index(idx, (n_freq, n_chn))
        feature_names.append(f"C{channel_names[channel_idx]}_F{freq_names[freq_idx]}")

    return feature_names

def convert_coh_2d_array(coh, n_sample=10000, pred_df=None):
    """
    Convert `coh` dictionary into a 2D array, selecting up to `n_sample` samples.

    Args:
        coh (dict): Dictionary containing tensor data.
        n_sample (int): Number of samples to return.
        pred_df (pd.DataFrame): DataFrame with labels (needs 'seg_epoch' and 'label').

    Returns:
        X_data (np.ndarray): Flattened 2D array of selected samples.
        y_data (np.ndarray): Corresponding labels.
    """
    # Ensure valid input if sampling is required
    print(f"Preparing coherence to 1D array")
    if n_sample > 0 and pred_df is None:
        raise ValueError(f"ERROR: pred shouldn't be None when n_sample > 0: n_sample={n_sample}")

    # Collecting preictal (y=1) and interictal (y=0) separately
    preictal_samples = []
    interictal_samples = []

    # Randomize the order of keys from coh to avoid time-based bias
    random_keys = list(coh.keys())
    random.shuffle(random_keys)

    # Iterate over randomized coh keys to collect data
    for key in random_keys:
        value = coh[key] # shape of value (109, 120)
        matching_row = pred_df.loc[pred_df['seg_epoch'] == key, 'label']
        if matching_row.empty:
            continue

        label = matching_row.values[0]
        flattened_sample = value.numpy().ravel()

        # Add sample to the appropriate class list (y=1 or y=0)
        if label == 1:
            preictal_samples.append((flattened_sample, label))
        elif label == 0:
            interictal_samples.append((flattened_sample, label))

        # Stop once we have collected enough samples
        if len(preictal_samples) >= n_sample // 2 and len(interictal_samples) >= n_sample // 2:
            break

    # Randomly sample from the collected preictal and interictal samples
    if len(preictal_samples) < n_sample // 2 or len(interictal_samples) < n_sample // 2:
        print(f"Warning: Not enough samples collected. Preictal={len(preictal_samples)}, Interictal={len(interictal_samples)}")

    selected_preictal = random.sample(preictal_samples, min(len(preictal_samples), n_sample // 2))
    selected_interictal = random.sample(interictal_samples, min(len(interictal_samples), n_sample // 2))

    # Combine the samples and labels
    selected_samples = selected_preictal + selected_interictal
    random.shuffle(selected_samples)  # Shuffle to mix preictal and interictal samples

    # Split samples and labels
    X_data = np.array([sample for sample, label in selected_samples])
    y_data = np.array([label for sample, label in selected_samples])

    return X_data, y_data


def explain_lime_tabular(pat_id, pre_itvl=45, n_sample=30, n_base_sample=10000, n_exp_feature=5000,
                         type='top', target_class=1,sz_id=-1, save=True, load_saved=True):
    # load model
    mid = get_final_model_id(pat_id, pre_itvl)
    mid_dict = {'model_id': mid, 'round_id': 1}
    config = get_saved_info(mid_dict, 'cfg')
    cosp = get_model(model_id=mid_dict, config=config, device='cpu')
    cosp.eval()

    # Load predictions
    # inter_itvl = 1440 if type == 'top-tp' else -1
    inter_itvl = 1440
    file_path = RST_PATH / f'detail/model{mid}_round1_pred_{inter_itvl}_tst.csv'
    pred = pd.read_csv(file_path)
    pred['probability'] = sigmoid(pred['prediction'])

    # Filter predictions based on sz_id
    if sz_id != -1:
        pred = pred[pred['sz_id'] == sz_id].reset_index()
    # Choose samples to be explained according to type
    sel_samples = select_samples(type, n_sample, pred, target_class)
    if sel_samples.empty:
        return None

    # Load coherence
    coh = load_tst_coh(pat_id)
    X_test, y_test = convert_coh_2d_array(coh, n_sample=n_base_sample, pred_df=pred)
    feature_names = get_combined_feature_name()

    # Instantiate the LIME Tabular Explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_test),  # The tabular data used for training or evaluation
        mode='regression',  # Set to 'regression' since it's a regression task
        feature_names=feature_names,  # Column names for the tabular data
        discretize_continuous=True  # LIME will discretize continuous features for better interpretability
    )

    # Instantiate the SHAP explainer for the CNN model (DeepExplainer)
    # explainer = shap.DeepExplainer(cosp, torch.tensor(X_test).float())

    # Choose an instance to explain
    imp_signed = defaultdict(float)
    imp_abs = defaultdict(float)
    imp_norm = defaultdict(float)
    count_p = defaultdict(int)
    count_n = defaultdict(int)
    criteria_count = defaultdict(int)
    for idx, row in sel_samples.iterrows():
        print(f"Pat{pat_id}, sz{row['sz_id']}: generating LIME explanation {idx + 1}/{len(sel_samples)}, "
              f"type={type}, class={target_class}...")
        f_exp = LIME_PATH / (f"Pat{pat_id}/tabular/Pat{pat_id}_pre{pre_itvl}_{row['seg_epoch']}"
                             f"_exp_class-{target_class}_lime.pkl")
        f_exp.parent.mkdir(parents=True, exist_ok=True)

        if f_exp.exists() and load_saved:
            with open(f_exp, 'rb') as f:
                explanation = pickle.load(f)
        else:
            instance = coh.get(row['seg_epoch'], None)
            if instance is None:
                instance = compute_coherence_one_segment(pat_id, row['seg_epoch'], row['file_name'])
            if instance is None:
                continue
            if isinstance(instance, torch.Tensor):
                instance = instance.cpu().numpy()
            instance = instance.reshape(-1)

            # # Generate SHAP values for the given instance
            # shap_values = explainer.shap_values(torch.tensor([instance]).float())[target_class]
            # # Extract SHAP importance values and accumulate them
            # explanation_values = list(zip(feature_names, shap_values[0]))
            # Create a summary plot for global feature importance
            # shap.summary_plot(shap_values, X_test, feature_names=feature_names)

            # Explain the prediction for the chosen instance
            explanation = explainer.explain_instance(
                data_row=instance,
                predict_fn=lambda x: predict_proba(x, cosp, 'cpu'),
                labels=(target_class,),
                num_samples=5000,
                num_features=n_exp_feature
            )
            with open(f_exp, 'wb') as f:
                pickle.dump(explanation, f)

        explanation_values = explanation.as_list()
        max_signed = max(importance for _, importance in explanation_values)
        min_signed = min(importance for _, importance in explanation_values)
        print(f"Pat{pat_id}, sz{row['sz_id']}: Instance {row['seg_epoch']} -len(explanation)={len(explanation_values)}, "
              f"Max_Importance={max_signed}, Min_Importance: {min_signed}")

        # Normalize importance within this sample using min-max normalization
        imp_abs_values = [abs(importance) for _, importance in explanation_values]
        imp_abs_sum = sum(imp_abs_values)
        imp_abs_min = min(imp_abs_values)
        imp_abs_max = max(imp_abs_values)

        # Extract importance for each feature and accumulate
        for criteria, importance in explanation.as_list():
            imp_signed[criteria] += importance
            imp_abs[criteria] += abs(importance)
            norm_one = (abs(importance) - imp_abs_min) / (imp_abs_max - imp_abs_min) if imp_abs_max - imp_abs_min > 0 else 0
            imp_norm[criteria] += norm_one
            criteria_count[criteria] += 1
            count_n[criteria] += 0  # keep to the same length as criteria_count
            count_p[criteria] += 0
            if importance > 0:
                count_p[criteria] += 1
            elif importance < 0:
                count_n[criteria] += 1

    # Compute averages for features
    avg_imp_signed = {f: imp_signed[f] / criteria_count[f] for f in imp_signed}
    avg_imp_abs = {f: imp_abs[f] / criteria_count[f] for f in imp_abs}
    avg_imp_nor = {f: imp_norm[f] / criteria_count[f] for f in imp_norm}

    imp_df = pd.DataFrame({
        "Criteria": imp_signed.keys(),
        "Importance_Signed": avg_imp_signed.values(),
        "Importance_Absolute": avg_imp_abs.values(),
        "Importance_Absolute_Normalized": avg_imp_nor.values(),
        "Counts_Positive": count_p.values(),
        "Counts_Negative": count_n.values(),
        "Counts": criteria_count.values()
    })

    # Split X_test into preictal (label == 1) and interictal (label == 0)
    preictal_samples = X_test[y_test == 1]
    interictal_samples = X_test[y_test == 0]

    avg_feature_values = []
    avg_feature_values_pre = []  # For label == 1 (preictal)
    avg_feature_values_inter = []  # For label == 0 (interictal)
    for criteria in imp_df["Criteria"]:
        fail = False
        # Extract the feature name without the conditions (e.g., C1-9_F6 > 0.11)
        feature_name = [f for f in criteria.split(' ') if f.startswith('C')]
        if len(feature_name) > 0:
            feature_name = feature_name[0]  # Get the actual feature name (e.g., f6_c2-8)
            if feature_name in feature_names:
                feature_idx = feature_names.index(feature_name)
                # Compute the average value of the feature across all instances in X_test
                avg_value = np.mean(X_test[:, feature_idx])
                avg_feature_values.append(avg_value)

                # Compute the average values for preictal and interictal samples
                avg_value_pre = np.mean(preictal_samples[:, feature_idx])
                avg_value_inter = np.mean(interictal_samples[:, feature_idx])

                avg_feature_values_pre.append(avg_value_pre)
                avg_feature_values_inter.append(avg_value_inter)
            else:
                print(f"Feature {feature_name} does not exist in feature_names.")
                fail = True
        else:
            print(f"Could not extract feature name from {criteria}.")
            fail = True

        if fail:
            avg_feature_values.append(np.nan)
            avg_feature_values_pre.append(np.nan)
            avg_feature_values_inter.append(np.nan)

    imp_df["Importance_Weighted"] = imp_df["Importance_Absolute_Normalized"] * imp_df["Counts"] / n_sample
    # Add the average feature values to imp_df
    imp_df["Feature_Value_Mean"] = avg_feature_values
    imp_df["Feature_Value_Mean_Preictal"] = avg_feature_values_pre
    imp_df["Feature_Value_Mean_Interictal"] = avg_feature_values_inter

    imp_df = imp_df.sort_values(by="Counts", ascending=False).reset_index(drop=True)
    if save:
        f_avg_imp = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/tabular/"
                                f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance"
                                f"_sz{sz_id}_{type}_class{target_class}_{n_sample}.csv")
        my_save(imp_df, f_avg_imp)
        print(f"Feature importance saved to {f_avg_imp}")


# Function to extract features from 'Criteria'
def extract_features(criteria):
    feature_name = [f for f in criteria.split(' ') if f.startswith('C')][0]
    pattern = r'C(\d+)-(\d+)_F(\d+)'  # Adjust the regex to extract necessary components
    match = re.match(pattern, feature_name)
    if match:
        # Extract C1, C2, and F1
        c1, c2, f1 = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return c1, c2, f1  # Return zero-based indices for c1, c2, and f1
    else:
        return None, None, None


# Function to plot the importance matrix as a heatmap
def plot_weighted_imp_tabular(pat_id, pre_itvl, sz_id, type, target_class, n_sample, save=True):
    # Load the CSV file
    f_imp = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/tabular/"
                        f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance"
                        f"_sz{sz_id}_{type}_class{target_class}_{n_sample}.csv")
    df = pd.read_csv(f_imp)

    freq_names = np.loadtxt(RST_PATH / 'coherence_frequencies_170.csv', delimiter=',').astype(int)
    # Initialize a 2D array for importance values (109 channels by 16 frequencies)
    imp_2d = np.zeros((109, 16))
    imp_pair_2d = np.zeros((109, 16*15))
    chn_pairs = [f"C{i}-C{j}" for i in range(1,17) for j in range(1,17) if i != j]

    # Process each row and extract features
    for index, row in df.iterrows():
        c1, c2, freq = extract_features(row['Criteria'])
        c1, c2 = c1 - 1, c2 - 1
        importance = row['Importance_Weighted'] #  Counts

        # Add importance to the 2D array at the respective positions
        if c1 is not None and c2 is not None and freq is not None:
            f1_idx = np.where(freq_names == freq)[0]
            imp_2d[f1_idx, c1] += importance
            imp_2d[f1_idx, c2] += importance

            for pair in [f"C{c1+1}-C{c2+1}", f"C{c2+1}-C{c1+1}"]:
                pair_idx = chn_pairs.index(pair)
                imp_pair_2d[f1_idx, pair_idx] += importance

    # # Normalize the importance values for better visualization
    # imp_2d = (imp_2d - imp_2d.min()) / (imp_2d.max() - imp_2d.min())
    # imp_pair_2d = (imp_pair_2d - imp_pair_2d.min()) / (imp_pair_2d.max() - imp_pair_2d.min())

    # Save the data to CSV files
    df_imp_2d = pd.DataFrame(imp_2d, index=freq_names, columns=[f"C{i}" for i in range(1, 17)])
    df_imp_pair_2d = pd.DataFrame(imp_pair_2d, index=freq_names, columns=chn_pairs)

    # Save the 2D arrays to CSV
    if save:
        f_chn = f_imp.parent / f"{f_imp.stem}_channel.csv"
        f_chn_pair = f_imp.parent / f"{f_imp.stem}_channel-pair.csv"
        df_imp_2d.to_csv(f_chn)
        print(f"Importance data saved to {f_chn}")
        df_imp_pair_2d.to_csv(f_chn_pair)
        print(f"Importance data saved to  {f_chn_pair}")

    # plot 1: x-axis: channel, y-axis:frequency
    plt.figure(figsize=(10, 15))
    xticklabels = [f"C{i}" for i in range(1, 17)]
    yticklabels = freq_names
    sns.heatmap(imp_2d, cmap="viridis", xticklabels=xticklabels, yticklabels=yticklabels)
    # Add solid gray horizontal lines to separate frequency bands
    for y in [4, 8, 13, 30]:
        y_idx = np.abs(freq_names - y).argmin()
        plt.axhline(y=y_idx, color='lightgrey', linestyle='solid', linewidth=1)
    # Add solid gray horizontal lines to separate gamma frequency bands
    for y in [53, 75, 103, 128]:
        y_idx = np.abs(freq_names - y).argmin()
        plt.axhline(y=y_idx, color='lightgrey', linestyle='dashed', linewidth=1)
    plt.gca().invert_yaxis()
    plt.title(f'Patient {pat_id}, Importance Heatmap')
    plt.ylabel('Frequencies')
    plt.xlabel('Channels')
    plt.tight_layout()
    if save:
        f_plot = f_chn.with_suffix('.png')
        plt.savefig(f_plot, dpi=300)
        print(f"Heatmap 1 saved to {f_plot}")
    plt.show()

    # plot 2: x-axis: channel pairs, y-axis:frequency
    fig = plt.figure(figsize=(40, 18))
    gs = gridspec.GridSpec(1, 2, width_ratios=[50, 1], wspace=0.04)

    # Create the heatmap and the color bar at the same time
    ax_heatmap = plt.subplot(gs[0])
    ax_cbar = plt.subplot(gs[1])

    # Create the heatmap with the color bar inside
    sns.heatmap(imp_pair_2d, cmap="viridis", xticklabels=chn_pairs, yticklabels=freq_names, ax=ax_heatmap, cbar=True,
                cbar_ax=ax_cbar)
    ax_heatmap.invert_yaxis()

    # Draw vertical lines to separate channel pairs starting with different numbers
    for i in range(1, 16):
        ax_heatmap.axvline(x=i * 15, color='lightgrey', linestyle='solid', lw=1)

    # Add solid gray horizontal lines to separate frequency bands
    for y in [4, 8, 13, 30]:
        y_idx = np.abs(freq_names - y).argmin()
        ax_heatmap.axhline(y=y_idx, color='lightgrey', linestyle='solid', linewidth=1)
    # Add solid gray horizontal lines to separate gamma frequency bands
    for y in [53, 75, 103, 128]:
        y_idx = np.abs(freq_names - y).argmin()
        ax_heatmap.axhline(y=y_idx, color='lightgrey', linestyle='dashed', linewidth=1)

    # Set titles and labels
    ax_heatmap.set_title(f'Patient {pat_id}, Importance Heatmap (Channel Pair)', fontsize=18)
    ax_heatmap.set_ylabel('Frequencies', fontsize=16)
    ax_heatmap.set_xlabel('Channel Pairs', fontsize=16)

    if save:
        f_plot = f_chn_pair.with_suffix('.png')
        plt.savefig(f_plot, dpi=300)
        print(f"Heatmap 1 saved to {f_plot}")

    # plt.tight_layout()
    plt.show()




# Function to load and display a LIME explanation
def load_and_display_explanation(f_exp):
    # Load the saved explanation from the pickle file
    with open(f_exp, 'rb') as f:
        explanation = pickle.load(f)

    # Print the explanation as text
    exp_list = explanation.as_list()
    for i in range(10):
        print(exp_list[i])

    # Plot the explanation
    explanation.as_pyplot_figure()
    plt.show()
    print('done')


def plot_importance_histogram(pat_id, pre_itvl, sz_id, target_class, n_sample, save=True):
    # Load the CSV file
    f_imp = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/tabular/"
                        f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance"
                        f"_sz{sz_id}_{type}_class{target_class}_{n_sample}.csv")
    df = pd.read_csv(f_imp)

    # Calculate the weights for percentages
    weights = (100 / len(df)) * np.ones_like(df['Importance_Weighted'])

    # Plot the histogram for the 'Importance_Weighted' column
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(df['Importance_Weighted'], weights=weights, bins=20, edgecolor='black')

    # Set the title and labels
    plt.title(f'Patient {pat_id},Histogram of Weighted Importance ({len(df):,} criteria)')
    plt.xlabel('Importance Weighted')
    plt.ylabel('Percentage (%)')
    plt.grid(axis='y')

    # Annotate the percentage on top of each bin
    for count, bin_edge in zip(counts, bins):
        plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, f'{count:.1f}',
                 ha='center', va='bottom')
    if save:
        save_path = Path(f_imp).parent / f"Pat{pat_id}_importance_histogram_pre{pre_itvl}_sz{sz_id}_class{target_class}_{n_sample}.png"
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()



def plot_on_xray(pat_id, channel_pairs, values, title, save=False, f_save=None, linestyle='solid',
                 mat_file_path=None, xray_image_path=None):
    # Use explicit paths or the caller's private metadata, never study filenames.
    files = XRAY_FILES.get(str(pat_id), {})
    mat_file_path = mat_file_path or files.get('electrodes')
    xray_image_path = xray_image_path or files.get('image')
    if not mat_file_path or not xray_image_path:
        raise ValueError('Supply electrode/image paths or configure private xray_files')
    electrode_coords = sio.loadmat(mat_file_path)['ELoc']
    xray_image = Image.open(xray_image_path)

    # Plot the X-ray image
    fig, ax = plt.subplots()  # Create a figure and axes
    ax.imshow(xray_image, cmap='gray')

    # Fix coherence values color range to [0, 1]
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = colormaps.get_cmap('jet')  # You can change to 'viridis' or another colormap if desired

    # Plot the electrodes and the top n coherence connections with color indicating connection strength
    for i, ch_pair in enumerate(channel_pairs):
        ch1, ch2 = [int(ch.split('C')[1]) for ch in ch_pair.split('-')]
        coord1, coord2 = electrode_coords[ch1 - 1], electrode_coords[ch2 - 1]

        # Map the coherence value (already in [0, 1]) to a color
        color = cmap(norm(values[i]))

        # Plot connection line with color
        ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], color=color, linewidth=2, linestyle=linestyle)

    # Scatter plot for electrodes
    ax.scatter(electrode_coords[:, 0], electrode_coords[:, 1], c='black', s=50)

    # Create a divider for the colorbar that matches the height of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Add colorbar to indicate the strength of the connection, fixed to range [0, 1]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)  # Colorbar shows range [0, 1],  label='Coherence'
    ax.set_title(title)
    if save:
        f_save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f_save, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {f_save}")
    else:
        plt.show()



def plot_electrode_connection(pat_id, pre_itvl, sz_id, type, target_class, n_sample, mean_by, n_top, save=True):
    f_coh = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/image/"
                        f"Pat{pat_id}_pre{pre_itvl}_lime_avg_coherence_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
    coh = np.load(f_coh)
    coh = np.flipud(coh)
    coh = pd.DataFrame(coh)

    freq_names = np.loadtxt(RST_PATH / 'coherence_frequencies_170.csv', delimiter=',').astype(int)[::-1]
    chn_pairs = [f"C{i}-C{j}" for i in range(1, 17) for j in range(i, 17) if i != j]
    coh.columns = chn_pairs
    coh.index = freq_names

    # Calculate mean coherence based on the selected frequency band
    if mean_by == 'all':
        avg_coh = coh.mean(axis=0)
    elif mean_by in FREQ_BANDS:
        # Get the frequency range for the selected band
        band_min, band_max = FREQ_BANDS[mean_by]
        # Select the rows in coh corresponding to this frequency range
        selected_freqs = coh.loc[(coh.index >= band_min) & (coh.index < band_max)]
        avg_coh = selected_freqs.mean(axis=0)
    else:
        print(f"Unknown mean_by value: {mean_by}")
        return

    # Find the top n highest coherence values and corresponding channel pairs
    top_n = avg_coh.nlargest(n_top)
    top_n_channels = top_n.index
    top_n_values = top_n.values
    itvl = 'preictal' if target_class else 'interictal'
    title = f"Patient {pat_id}: Top {n_top} connections, {mean_by}, {itvl}"
    save_path = RST_PATH / (f"figures/Pat{pat_id}/connectivity/"
                            f"Pat{pat_id}_connectivity_pre{pre_itvl}_sz{sz_id}_{type}"
                            f"_class{target_class}_{n_sample}"
                            f"_freq-{mean_by}_top{n_top}.png")
    plot_on_xray(pat_id, top_n_channels, top_n_values, title, save=save, f_save=save_path)
    print('Done!')


def plot_connection_change(pat_id, pre_itvl, sz_id, type, target_class, n_sample, mean_by, n_top, save=True):
    f_imp = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/tabular/"
                        f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}"
                        f"_{type}_class{target_class}_{n_sample}_channel-pair.csv")
    imp = pd.read_csv(f_imp, index_col=0)
    # normalize
    imp = (imp - imp.min().min()) / (imp.max().max() - imp.min().min())

    if mean_by == 'all':
        avg_imp = imp.mean(axis=0)
    elif mean_by in FREQ_BANDS:
        # Get the frequency range for the selected band
        band_min, band_max = FREQ_BANDS[mean_by]
        # Select the rows in coh corresponding to this frequency range
        imp_freqs = imp.loc[(imp.index >= band_min) & (imp.index < band_max)]
        avg_imp = imp_freqs.mean(axis=0)
    else:
        print(f"Unknown mean_by value: {mean_by}")
        return


    # Remove duplicate channel pairs (e.g., C6-C7 and C7-C6)
    channel_pairs = []
    for pair in avg_imp.index:
        c1, c2 = pair.split('-')
        c1, c2 = sorted([c1, c2])  # Always order channel pairs in ascending order
        sorted_pair = f"{c1}-{c2}"
        channel_pairs.append(sorted_pair)

    # Create a new DataFrame with the sorted pairs
    avg_imp.index = channel_pairs

    # Group by unique channel pairs and take the mean for duplicates
    avg_imp = avg_imp.groupby(avg_imp.index).mean()

    # Find the top n highest coherence values and corresponding channel pairs
    top_n = avg_imp.nlargest(n_top)
    top_n_channels = top_n.index
    top_n_values = top_n.values
    title = f"Patient {pat_id}: Top {n_top} connection changes, freq-{mean_by}, {top_n.mean():.2f}"
    save_path = RST_PATH / (f"figures/Pat{pat_id}/connectivity/"
                            f"Pat{pat_id}_connect-change_pre{pre_itvl}_sz{sz_id}_{type}"
                            f"_class{target_class}_{n_sample}"
                            f"_freq-{mean_by}_top{n_top}.png")
    plot_on_xray(pat_id, top_n_channels, top_n_values, title, save=save, f_save=save_path, linestyle='dotted')
    print('Done!')
