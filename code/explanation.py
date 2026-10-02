import os
import pathlib
import pickle
import random
import time
from collections import defaultdict
import re
from scipy.stats import f_oneway, ttest_ind
import pandas as pd
import torch
from analysis import load_coherence_dict, mycol, plot_coh_heatmap, select_samples
import matplotlib.gridspec as gridspec
from config import N_FREQ, N_CHN_PAIR, WORK_PATH, PAT_LIST_ALL, PAT_LIST, PAT_ID_LIST, RST_PATH, CONFIG, LIME_PATH
from data import compute_coherence, compute_coherence_one_segment
from models import get_model
from util import get_saved_info, get_final_model_id, get_opt_thd, sigmoid, load_mat_to_np, fill_na, load_tst_coh, \
    get_segment_steps, my_save
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image, lime_tabular
from skimage.color import gray2rgb
from skimage.segmentation import mark_boundaries
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from skimage import measure
from scipy.stats import pearsonr





# Function to plot image with LIME explanation
def plot_image_with_explanation(pat_id, sz_id, image, explanation, ax, prob):
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=False,
                                                num_features=5,
                                                hide_rest=False)
    print(f"image.shape={image.shape}, temp.shape={temp.shape}, mask.shape={mask.shape}")

    # Flip both image and temp upside down
    image_flipped = np.flipud(image)
    temp_flipped = np.flipud(temp)
    mask_flipped = np.flipud(mask)

    # Plot the flipped images with better colormap and figure size
    ax.imshow(image_flipped, interpolation='none', cmap='viridis', alpha=1.0)  # plasma inferno magma cividis
    ax.imshow(temp_flipped, alpha=0.5, cmap='coolwarm')

    # # Find contours around the red areas
    # contours = measure.find_contours(mask_flipped, 0.5)
    # for contour in contours:
    #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='yellow')

    ax.set_title(f'Pat{pat_id}, sz{sz_id}, Prob: {prob:.2f}', fontsize=18)
    # ax.set_ylabel('Frequency', fontsize=16)
    ax.axis('off')


def plot_lime_explanation(pat_id, data, coh, n_row, n_col, thd):
    n_top = n_row * n_col
    # true_positives = data[(data['probability'] < thd) & (data['label'] == 0)]
    # top_tps = true_positives.nsmallest(n_top, 'probability').reset_index()
    true_positives = data[(data['probability'] > thd) & (data['label'] == 1)]
    top_tps = true_positives.nlargest(n_top, 'prediction').reset_index()
    if top_tps.empty: return

    # Initialize LIME image explainer
    explainer = lime_image.LimeImageExplainer()

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 5, n_row * 5))
    axes = axes.flatten()
    # Load the corresponding images
    for idx, row in top_tps.iterrows():
        if idx >= n_row * n_col: break
        print(f"{idx + 1}/{len(top_tps)}:")

        epoch = row['seg_epoch']
        sz_id = row['sz_id']
        image_one = coh.get(epoch, None)
        if image_one is None:
            axes[idx].axis('off')
            continue
        prob_one = torch.tensor(row['probability'], dtype=torch.float32).to('cpu')
        explanation = explainer.explain_instance(gray2rgb(image_one),
                                                 predict_proba,
                                                 top_labels=1,
                                                 hide_color=0,
                                                 num_samples=1000)
        plot_image_with_explanation(pat_id, sz_id, image_one, explanation, axes[idx], prob_one)
    for i in range(idx, n_top):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()




def predict_proba(images, model, device):
    # images shape (num_samples, 109, 120, 3)
    if images.ndim == 1:
      images = images.reshape([1, N_FREQ,N_CHN_PAIR,1])
    elif images.ndim == 2:
        # Assume images were flattened from (120, 109) to (13080,), so we reshape accordingly
        num_samples = images.shape[0]
        images = images.reshape(num_samples, N_FREQ, N_CHN_PAIR, 1)

    model.eval()
    images = torch.tensor(images[:, :, :, 0], dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(images).cpu().numpy()
        probabilities = 1 / (1 + np.exp(-outputs))
        return np.concatenate((1 - probabilities, probabilities), axis=1)


def generate_avg_importance_lime(pat_id, type, pre_itvl=45, target_class=1, n_sample=30, sz_id=-1, save=True,
                                 plot=True):
    print(
        f"\nPat{pat_id}: Generating average importance with LIME: patient={pat_id}, type={type}, pre_itvl={pre_itvl}, "
        f"target_class={target_class}, n_sample={n_sample}, sz_id={sz_id}, save={save}, plot={plot}")

    assert pat_id in PAT_ID_LIST, f"ERROR: unknown patient id={pat_id}, available={PAT_ID_LIST}"

    # Load model
    mid = get_final_model_id(pat_id, pre_itvl)
    mid_dict = {'model_id': mid, 'round_id': 1}
    config = get_saved_info(mid_dict, 'cfg')
    model = get_model(model_id=mid_dict, config=config, device='cpu')
    model.eval()

    # Load predictions and filter based on sz_id
    inter_itvl = 1440 if type == 'top-tp' else -1
    file_path = RST_PATH / f'detail/model{mid}_round1_pred_{inter_itvl}_tst.csv'
    pred = pd.read_csv(file_path)
    pred['probability'] = sigmoid(pred['prediction'])

    if sz_id != -1:
        pred = pred[pred['sz_id'] == sz_id].reset_index()

    sel_samples = select_samples(type, n_sample, pred, target_class)
    if sel_samples.empty:
        return None

    # Explain using LIME and accumulate importance masks
    explainer = lime_image.LimeImageExplainer()
    masks = []
    for idx, row in sel_samples.iterrows():
        print(
            f"Pat{pat_id}, sz{row['sz_id']}: generating LIME explanation {idx + 1}/{len(sel_samples)}, type={type}, class={target_class}...")

        f_exp = LIME_PATH / f"Pat{pat_id}/Pat{pat_id}_pre{pre_itvl}_{row['seg_epoch']}_exp_class-{target_class}_lime.npy"
        f_exp.parent.mkdir(parents=True, exist_ok=True)

        if f_exp.exists():
            mask = np.load(f_exp)
            print(f"Pat{pat_id}: Loaded mask from {f_exp}")
        else:
            image_one = compute_coherence_one_segment(pat_id, row['seg_epoch'], row['file_name'])
            if isinstance(image_one, torch.Tensor):
                image_one = image_one.cpu().numpy()
            explanation = explainer.explain_instance(
                gray2rgb(image_one),
                lambda x: predict_proba(x, model, 'cpu'),
                labels=(target_class,),
                hide_color=0,
                num_samples=5000,
                batch_size=500
            )
            _, mask = explanation.get_image_and_mask(
                label=target_class,
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            my_save(mask, f_exp)
            print(f"Pat{pat_id}: Saved mask to {f_exp}")

        masks.append(mask)

    if len(masks) == 0:
        return None

    # Aggregate importance
    avg_imp = np.zeros((N_FREQ, N_CHN_PAIR), dtype=float)
    for mask in masks:
        avg_imp += mask
    avg_imp /= len(masks)
    avg_imp = np.flipud(avg_imp)

    # Save aggregated importance
    f_avg_imp = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                            f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
    if save:
        my_save(avg_imp, f_avg_imp)

    # Optionally plot the result
    if plot:
        title = (
            f"Patient {pat_id}, Seizure {sz_id}, Preictal {pre_itvl}, Average Importance using LIME, \n"
            f"sample type={type}-class{target_class}, #samples={len(sel_samples)}, from {len(sel_samples['sz_id'].unique())} (out of {len(pred['sz_id'].unique())}) seizures\n"
            f"(#preictal:{sel_samples['label'].sum()}, #interictal:{len(sel_samples) - sel_samples['label'].sum()}), "
            f"(Probability mean: {int(100 * sel_samples['probability'].mean())}%, "
            f"min: {int(100 * sel_samples['probability'].min())}%, "
            f"max: {int(100 * sel_samples['probability'].max())}%)"
        )
        plot_avg_imp_lime(avg_imp, title, save=save, f_save=f_avg_imp.with_suffix('.png'))

    return avg_imp





# def generate_avg_imp_lime(pat_id, type, pre_itvl=45, target_class=1,
#                           n_sample=30, sz_id=-1, save=True, plot=True):
#     print(f"\nPat{pat_id}: Generating average importance: patient={pat_id}, type={type}, pre_itvl={pre_itvl}, "
#           f"target_class={target_class}, n_sample={n_sample}, sz_id={sz_id}, save={save}, plot={plot}")
#
#     assert pat_id in PAT_ID_LIST, f"ERROR: unknown patient id={pat_id}, available={PAT_ID_LIST}"
#
#     # Load model and threshold
#     inter_itvl = 1440 if type == 'top-tp' else -1
#     mid = get_final_model_id(pat_id, pre_itvl)
#     # thd = get_opt_thd(pat_id, pre_itvl)
#
#     # Load predictions
#     file_path = RST_PATH / f'detail/model{mid}_round1_pred_{inter_itvl}_tst.csv'
#     pred = pd.read_csv(file_path)
#     pred['probability'] = sigmoid(pred['prediction'])
#
#     # Load coherence matrices
#     mid_dict = {'model_id': mid, 'round_id': 1}
#     config = get_saved_info(mid_dict, 'cfg')
#     model = get_model(model_id=mid_dict, config=config, device='cpu')
#     model.eval()
#
#     # Filter predictions based on sz_id
#     if sz_id != -1:
#         pred = pred[pred['sz_id'] == sz_id].reset_index()
#
#     # Choose samples to be explained according to type
#     sel_samples = select_samples(type, n_sample, pred, target_class)
#
#     if sel_samples.empty:
#         return None
#
#     # Explain using LIME
#     explainer = lime_image.LimeImageExplainer()
#     masks = []
#     avg_coherence = np.zeros((N_FREQ, N_CHN_PAIR), dtype=float)
#     coh = load_tst_coh(pat_id)
#
#     for idx, row in sel_samples.iterrows():
#         print(
#             f"Pat{pat_id}, sz{row['sz_id']}: generating LIME explanation {idx + 1}/{len(sel_samples)}, type={type}, class={target_class}...")
#         f_exp = LIME_PATH / f"Pat{pat_id}/Pat{pat_id}_pre{pre_itvl}_{row['seg_epoch']}_exp_class-{target_class}_lime.npy"
#         f_exp.parent.mkdir(parents=True, exist_ok=True)
#
#         # Generate the mask
#         image_one = coh.get(row['seg_epoch'], None)
#         if image_one is None:
#             # shape (109, 120)
#             image_one = compute_coherence_one_segment(pat_id, row['seg_epoch'], row['file_name'])
#         if image_one is None:
#             continue
#         if isinstance(image_one, torch.Tensor):
#             image_one = image_one.cpu().numpy()
#
#         if f_exp.exists():
#             # Load the saved mask
#             mask = np.load(f_exp)
#             print(f"Pat{pat_id}: Loaded mask from {f_exp}")
#         else:
#             # # Generate the mask
#             # image_one = coh.get(row['seg_epoch'], None)
#             # if image_one is None:
#             #     image_one = compute_coherence_one_segment(pat_id, row['seg_epoch'], row['file_name'])
#             # if image_one is None:
#             #     continue
#             #     if isinstance(image_one, torch.Tensor):
#             #         image_one = image_one.cpu().numpy()
#
#             explanation = explainer.explain_instance(
#                 gray2rgb(image_one),  # Convert to RGB, shape (109, 120, 3)
#                 lambda x: predict_proba(x, model, 'cpu'),  # Function to predict class probabilities.
#                 labels=(target_class,),  # Specify the class to explain.
#                 hide_color=0,  # Color used to hide the rest of the image.
#                 top_labels=None,  # Do not limit to top predicted labels.
#                 num_samples=5000,  # Number of perturbed samples generated for LIME.
#                 batch_size=500  # Batch size for the model predictions.
#             )
#
#             _, mask = explanation.get_image_and_mask(
#                 label=target_class,  # Focus on the explanation for the target class.
#                 positive_only=True,  # Only highlight the positive contributions.
#                 num_features=5,  # Show the top 5 features(superpixels) that contribute to the prediction.
#                 hide_rest=False  # Do not hide the rest of the image.
#             )
#
#             # Save the mask
#             my_save(mask, f_exp)
#             print(f"Pat{pat_id}: Saved mask to {f_exp}")
#         masks.append(mask)
#
#         # Accumulate coherence values
#         avg_coherence += image_one
#
#     if len(masks) == 0: return None
#
#     # Aggregate importance
#     avg_imp = np.zeros((N_FREQ, N_CHN_PAIR), dtype=float)
#     for mask in masks:
#         avg_imp += mask
#     avg_imp /= len(masks)
#     avg_imp = np.flipud(avg_imp)
#     avg_coherence /= len(masks)
#
#     # Save aggregated importance
#     f_avg_imp = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
#                          f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
#     f_avg_coh = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
#                              f"Pat{pat_id}_pre{pre_itvl}_lime_avg_coherence_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
#     if save:
#         my_save(avg_imp, f_avg_imp)
#         my_save(avg_coherence, f_avg_coh)
#
#     if plot:
#         title = (
#             f"Patient {pat_id}, Seizure {sz_id}, Preictal {pre_itvl}, Average Importance using LIME, \n"
#             f"sameple type={type}-class{target_class}, #samples={len(sel_samples)}, from {len(sel_samples['sz_id'].unique())} (out of {len(pred['sz_id'].unique())}) seizures\n"
#             f"(#preictal:{sel_samples['label'].sum()}, #interictal:{len(sel_samples) - sel_samples['label'].sum()}), "
#             f"(Probability mean: {int(100 * (sel_samples['probability'].mean()))}%, "
#             f"min: {int(100 * (sel_samples['probability'].min()))}%, "
#             f"max: {int(100 * (sel_samples['probability'].max()))}%)"
#         )
#         plot_avg_imp_lime(avg_imp, title, save=save, f_save=f_avg_imp.with_suffix('.png'))
#
#     return avg_imp




def plot_avg_imp_lime(avg_imp, title, save=True, f_save=None, colorbar=False, ax=None, ax_fontsize=14, contour=False,
                      range01=True, cmap='cividis', n_contour=3, contour_list=None):
    # Create the plot
    if ax is None:
        full_plot = True
        fig, ax = plt.subplots(figsize=(15, 15))  # Using subplots to get an axis object
    else:
        full_plot = False
    x_tick_labels, x_ticks, y_tick_labels, y_ticks = get_x_y_ticks()

    # Plot the image
    if range01:
        cax = ax.imshow(avg_imp, cmap=cmap, interpolation='none', vmin=0, vmax=1)
    else:
        cax = ax.imshow(avg_imp, cmap=cmap, interpolation='none')
    # Add solid gray horizontal lines to separate frequency bands
    for y in [4, 8, 13, 30]:
        ax.axhline(y=y_ticks[y_tick_labels.index(y)], color='white', linestyle='solid', linewidth=1.5)
    # Add solid gray horizontal lines to separate gamma frequency bands
    for y in [53, 75, 103, 128]:
        ax.axhline(y=y_ticks[y_tick_labels.index(y)], color='lightgrey', linestyle='dashed', linewidth=1.5)
    for x in x_ticks:
        ax.axvline(x=x, color='lightgrey', linewidth=1)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation=45, fontsize=ax_fontsize)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=ax_fontsize)
    ax.set_title(title, fontsize=ax_fontsize+2)

    if colorbar and full_plot:
        # Create a divider for the existing axes and append a new axes for the colorbar
        divider = make_axes_locatable(ax)
        cax_cb = divider.append_axes("right", size="5%", pad=0.05)
        # Add colorbar with fixed range from 0 to 1 and make it the same height as the plot
        cbar = fig.colorbar(cax, cax=cax_cb)
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        # Set the font size of the color bar ticks
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Importance', fontsize=16)

    if contour:
        if n_contour == 3:
            # Add contour lines at specific importance levels
            contour_levels = [0.5, 0.7, 0.9] if contour_list is None else contour_list
            contour_colors = ['cyan', 'magenta', 'red']
            contours = ax.contour(avg_imp, levels=contour_levels, colors=contour_colors, linewidths=1.5)
            ax.clabel(contours, inline=True, fontsize=ax_fontsize - 2, fmt='%0.1f')
        elif n_contour == 2:
            contour_levels = [0.5, 0.9] if contour_list is None else contour_list
            contour_colors = ['cyan', 'red']
            contours = ax.contour(avg_imp, levels=contour_levels, colors=contour_colors, linewidths=1.5)
            ax.clabel(contours, inline=True, fontsize=ax_fontsize - 2, fmt='%0.1f')
        elif n_contour == 1:
            contour_levels = [0.9] if contour_list is None else contour_list
            contour_colors = ['red']
            contours = ax.contour(avg_imp, levels=contour_levels, colors=contour_colors, linewidths=1.5)
            ax.clabel(contours, inline=True, fontsize=ax_fontsize - 2, fmt='%0.1f')

    if full_plot:
        # Save or show the plot
        if save and f_save:
            my_save(fig, f_save)
        else:
            plt.show()






def get_top_contributors(importance, channel_pairs, frequencies, top_n=5):
    print(f"\nTop {top_n} areas contributing most to the prediction:")
    flat_importance = importance.flatten()
    top_indices = np.argsort(flat_importance)[-top_n:]

    for idx in reversed(top_indices):
        row, col = np.unravel_index(idx, importance.shape)
        print(
            f"Channel pair: {channel_pairs[col]}, Frequency: {frequencies[row]:.2f} Hz, Importance: {flat_importance[idx]:.4f}")

# 3. Summarize and list the bottom area contributing least to the prediction
def get_bottom_contributors(importance, channel_pairs, frequencies, bottom_n=5):
    print(f"\nBottom {bottom_n} areas contributing least to the prediction:")
    flat_importance = importance.flatten()
    bottom_indices = np.argsort(flat_importance)[:bottom_n]

    for idx in bottom_indices:
        row, col = np.unravel_index(idx, importance.shape)
        print(
            f"Channel pair: {channel_pairs[col]}, Frequency: {frequencies[row]:.2f} Hz, Importance: {flat_importance[idx]:.4f}")



def get_x_y_ticks():
    # y axis represents frequencies
    coh_freq = np.loadtxt(RST_PATH / f'coherence_frequencies_170.csv', delimiter=',').astype('int')[::-1]
    # (delta (0.5-4Hz), theta (4-8Hz), beta (8-13 Hz), alpha (13-30 Hz),
    # gamma band 1 (30-47 Hz), gamma band 2 (53-75 Hz), gamma band 3 (75-97 Hz), and gamma band 4 (103-128 Hz)),
    y_tick_labels = [4, 8, 13, 30, 53, 75, 103, 128, 170]
    y_ticks = [np.abs(coh_freq - i).argmin() for i in y_tick_labels]

    # x axis represents channel pairs
    channel_pairs = np.loadtxt(RST_PATH / f'channel_pairs.csv', delimiter=',', dtype=str)
    # Generate x-tick labels for the first pair of each channel
    x_tick_labels = [f"{i}-{i + 1}" for i in range(1, 16)]
    x_ticks = [np.where(channel_pairs == i)[0][0] for i in x_tick_labels]
    # x_tick_labels = np.array([f"{i}-{i + 1}" for i in range(1, 16)] + ['1-16', '2-16', '1-7', '2-6'])
    # x_ticks = np.array([np.where(channel_pairs == i)[0][0] for i in x_tick_labels])
    # x_tick_labels[-2:] = '...'
    # # Combine labels and ticks, sort based on ticks, and return the sorted labels and ticks
    # sorted_pairs = sorted(zip(x_ticks, x_tick_labels))
    # x_ticks, x_tick_labels = zip(*sorted_pairs)

    return x_tick_labels, x_ticks, y_tick_labels, y_ticks


# def plot_lime_importance_all_patients(type='top-tp', save=None):
def plot_lime_importance_all_patients(type, pre_itvl=45,
                                      target_class=1, sz_id=-1, n_sample=500, save=True, contour=False):
    x_tick_labels, x_ticks, y_tick_labels, y_ticks = get_x_y_ticks()

    n_cols = 5  # Number of columns in the plot grid
    n_rows = len(PAT_ID_LIST) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, pat_id in enumerate(PAT_ID_LIST):
        ax = axes[i]

        f_save = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                             f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
        if f_save.exists:
            avg_imp = np.load(f_save)
        else:
            print(f"File {f_save.name} not found for patient {pat_id}")
            return

        title = f"Patient {pat_id}"

        plot_avg_imp_lime(avg_imp, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=8,
                          contour=contour)

        # # cax = ax.imshow(importance_matrix, cmap='cividis', interpolation='none')
        # cax = ax.imshow(importance_matrix, cmap='cividis', interpolation='none', vmin=0, vmax=1)
        # # ax.axis('off')
        # ax.set_xticks(x_ticks)
        # ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=6)
        # ax.set_yticks(y_ticks)
        # ax.set_yticklabels(y_tick_labels, fontsize=6)
        # ax.set_title(f"Patient {pat_id}", fontsize=14)
    # Remove unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    # # Adjust subplots to make room for the color bar
    # fig.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.05)
    # # Add a color bar for the entire figure on the right
    # cbar_ax = fig.add_axes([0.81, 0.08, 0.015, 0.81])  # [left, bottom, width, height]
    # cbar = fig.colorbar(cax, cax=cbar_ax, orientation='vertical')
    # cbar.set_label('Importance', fontsize=14)

    # Save the plot if save_path is provided
    if save:
        if contour:
            f_save = RST_PATH/f"figures/lime_importance_all_sz{sz_id}_{type}_class{target_class}_{n_sample}_contour.png"
        else:
            f_save = RST_PATH / f"figures/lime_importance_all_sz{sz_id}_{type}_class{target_class}_{n_sample}.png"
        my_save(fig, f_save)
    plt.show()


def analyze_lime_imp_one_4subplots(pat_id, type, pre_itvl=45, target_class=1, n_sample=100, sz_id=-1, save=True):
    # Load data
    f_avg_imp = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                            f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
    avg_imp = np.load(f_avg_imp)
    f_avg_coh = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                            f"Pat{pat_id}_pre{pre_itvl}_lime_avg_coherence_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
    avg_coh = np.load(f_avg_coh)
    avg_coh = np.flipud(avg_coh)

    # Create a figure with 2 rows and 2 columns (4 subplots)
    fig, axes = plt.subplots(1, 4, figsize=(30, 6))

    # ------------------------------- 0 -----------------------------------
    # First subplot: [0,0]: Average Importance
    ax = axes[0]
    title = "Average Importance"
    plot_avg_imp_lime(avg_imp, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14, contour=True)
    # Plot top 5% values in avg_imp as red dots
    threshold = np.percentile(avg_imp, 95)
    y_coords, x_coords = np.where(avg_imp >= threshold)
    ax.scatter(x_coords, y_coords, color='red', s=10, label='Top 5% values')

    # ---------------------------------- 1 --------------------------------
    # Second subplot:  Distribution of Importance Values
    ax = axes[1]
    bins = np.linspace(0, 1, 11)
    counts, bins, patches = ax.hist(avg_imp.flatten(), bins=bins, color=mycol['blue'], alpha=0.7, density=False)
    percentages = (counts / counts.sum()) * 100
    ax.cla()
    width = 0.09 # 0.9 * (bins[1] - bins[0])  # Slightly less than the bin width to create gaps
    x_positions = np.arange(0.05, 1.0, 0.1)
    bars = ax.bar(x_positions, percentages, width=width, color=mycol['blue'], alpha=0.7) # edgecolor='black'
    # bars = ax.bar(bins[:-1] + (width / 2), percentages, width=width, color=mycol['blue'], alpha=0.7) # edgecolor='black'
    # Add sample numbers on top of each bin
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(count)}', ha='center', va='bottom',
                fontsize=12)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_title(f'Distribution of Importance Values \n (total {np.size(avg_imp)} features)', fontsize=16)
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.grid(axis='y')

    #-------------------------------- 2 ----------------------------------
    # Third subplot: Mean Coherence for Different Importance Ranges
    ax = axes[2]
    importance_ranges = np.arange(0, 1.1, 0.1)  # Ranges from 0 to 1 with a step of 0.1
    mean_coherence = []
    for i in range(len(importance_ranges) - 1):
        lower_bound = importance_ranges[i]
        upper_bound = importance_ranges[i + 1]
        mask = (avg_imp >= lower_bound) & (avg_imp < upper_bound)
        if np.any(mask):  # Ensure there are values in the range
            avg_coh_value = np.mean(avg_coh[mask])
            mean_coherence.append(avg_coh_value)
        else:
            mean_coherence.append(np.nan)
    # Compute x positions as the middle of each range
    x_positions = importance_ranges[:-1] + 0.05  # Middle of each range (e.g., 0.05 for [0, 0.1])
    ax.plot(x_positions, mean_coherence, label='Mean Coherence', color='blue', marker='o')
    ax.set_xticks(np.arange(0, 1.1, 0.1))  # Ticks at 0, 0.1, ..., 1
    ax.set_xticklabels([f'{tick:.1f}' for tick in np.arange(0, 1.1, 0.1)], rotation=0, fontsize=12)
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_ylabel('Mean Coherence', fontsize=14)
    ax.set_title('Mean Coherence \n in Different Importance Ranges', fontsize=16)
    ax.grid(axis='both')

    # ----------------------------------- 3 ------------------------------
    # Fourth subplot: Distribution of Coherence Values for Different Importance Ranges
    ax = axes[3]
    coherence_distributions = []

    for i in range(len(importance_ranges) - 1):
        lower_bound = importance_ranges[i]
        upper_bound = importance_ranges[i + 1]
        mask = (avg_imp >= lower_bound) & (avg_imp < upper_bound)
        if np.any(mask):  # Ensure there are values in the range
            coherence_distributions.append(avg_coh[mask])
        else:
            coherence_distributions.append([])  # Handle empty regions

    # Compute x positions as the middle of each range
    x_positions = np.arange(0.05, 1.05, 0.1)  # Midpoints: 0.05, 0.15, ..., 0.95

    # Draw the boxplot with the x positions in the middle of the range
    ax.boxplot(coherence_distributions, positions=x_positions, widths=0.06, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='darkblue'))

    # Set x-axis ticks and labels
    ax.set_xticks(np.arange(0, 1.1, 0.1))  # Ticks at 0, 0.1, ..., 1
    ax.set_xticklabels([f'{tick:.1f}' for tick in np.arange(0, 1.1, 0.1)], rotation=0, fontsize=12)
    ax.set_xlim(0, 1)
    # Set labels and title
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_ylabel('Coherence', fontsize=14)
    ax.set_title('Distribution of Coherence \n in Different Importance Ranges', fontsize=16)
    ax.grid(axis='both')

    # ----------------------------------- end ------------------------------
    plt.tight_layout()
    if save:
        f_save = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                             f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{type}_class{target_class}_{n_sample}_detail.png")
        my_save(fig, f_save)
    plt.show()


def analyze_lime_imp_one_6subplots_2(pat_id, type, pre_itvl=45, target_class=1, n_sample=100, sz_id=-1, save=True):
    # Load data
    f_avg_imp = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                            f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
    avg_imp = np.load(f_avg_imp)
    f_avg_coh = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                            f"Pat{pat_id}_pre{pre_itvl}_lime_avg_coherence_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
    avg_coh = np.load(f_avg_coh)
    avg_coh = np.flipud(avg_coh)

    # Create a figure with 2 rows and 2 columns (4 subplots)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ------------------------------- 0-0 -----------------------------------
    # First subplot: [0,0]: Average Importance
    ax = axes[0, 0]
    title = f"(a) Patient {pat_id} Average Importance (type:{type})"
    plot_avg_imp_lime(avg_imp, title, save=False, f_save=None,
                      colorbar=False, ax=ax, ax_fontsize=14, contour=True)
    # Plot top 5% values in avg_imp as red dots
    threshold = np.percentile(avg_imp, 95)
    y_coords, x_coords = np.where(avg_imp >= threshold)
    ax.scatter(x_coords, y_coords, color='red', s=10, label='Top 5% values', alpha=0.3)

    # ---------------------------------- 0-1 --------------------------------
    # Second subplot:  Distribution of Importance Values
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 11)
    counts, bins, patches = ax.hist(avg_imp.flatten(), bins=bins, color=mycol['blue'], alpha=0.7, density=False)
    percentages = (counts / counts.sum()) * 100
    ax.cla()
    width = 0.09 # 0.9 * (bins[1] - bins[0])  # Slightly less than the bin width to create gaps
    x_positions = np.arange(0.05, 1.0, 0.1)
    bars = ax.bar(x_positions, percentages, width=width, color=mycol['blue'], alpha=0.7) # edgecolor='black'
    # bars = ax.bar(bins[:-1] + (width / 2), percentages, width=width, color=mycol['blue'], alpha=0.7) # edgecolor='black'
    # Add sample numbers on top of each bin
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(count)}', ha='center', va='bottom',
                fontsize=12)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_title(f'(b) Distribution of Importance Values \n (total {np.size(avg_imp)} features)', fontsize=16)
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.grid(axis='y')

    #-------------------------------- 0-2 ----------------------------------
    #  Mean Coherence for Different Importance Ranges
    ax = axes[0, 2]
    importance_ranges = np.arange(0, 1.1, 0.1)  # Ranges from 0 to 1 with a step of 0.1
    mean_coherence = []
    mean_importance = []

    for i in range(len(importance_ranges) - 1):
        lower_bound = importance_ranges[i]
        upper_bound = importance_ranges[i + 1]
        mask = (avg_imp >= lower_bound) & (avg_imp < upper_bound)
        if np.sum(mask) >= 10:  # Ensure there are at least 10 values in the range
            avg_coh_value = np.mean(avg_coh[mask])
            avg_imp_value = np.mean(avg_imp[mask])
            mean_coherence.append(avg_coh_value)
            mean_importance.append(avg_imp_value)
        else:
            mean_coherence.append(np.nan)
            mean_importance.append(np.nan)

    # Compute Pearson correlation between mean_importance and mean_coherence
    mask_valid = ~np.isnan(mean_importance) & ~np.isnan(mean_coherence)
    pearson_corr_mean = pearsonr(np.array(mean_importance)[mask_valid], np.array(mean_coherence)[mask_valid])[0]

    # Compute Pearson correlation between avg_imp and avg_coh
    avg_imp_1d = np.asarray(avg_imp).flatten()
    avg_coh_1d = np.asarray(avg_coh).flatten()
    pearson_corr_avg = pearsonr(avg_imp_1d, avg_coh_1d)[0]

    # Compute x positions as the middle of each range
    x_positions = importance_ranges[:-1] + 0.05  # Middle of each range (e.g., 0.05 for [0, 0.1])
    ax.plot(x_positions, mean_coherence, label='Mean Coherence', color=mycol['blue'], marker='o')
    ax.set_xticks(np.arange(0, 1.1, 0.1))  # Ticks at 0, 0.1, ..., 1
    ax.set_xticklabels([f'{tick:.1f}' for tick in np.arange(0, 1.1, 0.1)], rotation=0, fontsize=12)
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_ylabel('Mean Coherence', fontsize=14)
    ax.set_title('(c) Mean Coherence \n in Different Importance Ranges', fontsize=16)
    ax.grid(axis='both')

    # Display the Pearson correlation values on the plot
    textstr = '\n'.join((
        f'Pearson Corr (Mean): {pearson_corr_mean:.2f}',
        f'Pearson Corr (All): {pearson_corr_avg:.2f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    # ----------------------------------- 1-0 ------------------------------
    #  Mean Coherence Plot
    ax = axes[1, 0]
    #  apply a scaling factor to enhance contrast
    scaling_factor = 5  # Increase contrast
    avg_coh_scaled = np.clip(avg_coh * scaling_factor, 0, 1)
    title = f"(d) Average Coherence (type:{type})"
    plot_avg_imp_lime(avg_coh_scaled, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14,
                      contour=False, cmap='viridis')

    # ----------------------------------- 1-1 ------------------------------
    #  Distribution of Coherence
    ax = axes[1, 1]
    bins = np.linspace(0, 1, 11)
    counts, bins, patches = ax.hist(avg_coh.flatten(), bins=bins, color=mycol['blue'], alpha=0.7, density=False)
    percentages = (counts / counts.sum()) * 100
    ax.cla()
    width = 0.09  # 0.9 * (bins[1] - bins[0])  # Slightly less than the bin width to create gaps
    x_positions = np.arange(0.05, 1.0, 0.1)
    bars = ax.bar(x_positions, percentages, width=width, color=mycol['orange'], alpha=0.7)  # edgecolor='black'
    # bars = ax.bar(bins[:-1] + (width / 2), percentages, width=width, color=mycol['blue'], alpha=0.7) # edgecolor='black'
    # Add sample numbers on top of each bin
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(count)}', ha='center', va='bottom',
                fontsize=12)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_title(f'(e) Distribution of Coherence \n (total {np.size(avg_coh)} features)', fontsize=16)
    ax.set_xlabel('Coherence', fontsize=14)
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.grid(axis='y')

    # --------------------------------- 1-2 ---------------------------------
    #  Mean Importance for Different Coherence Ranges
    ax = axes[1, 2]
    coherence_ranges = np.arange(0, 1.1, 0.1)
    mean_importance = []
    mean_coherence = []

    for i in range(len(coherence_ranges) - 1):
        lower_bound = coherence_ranges[i]
        upper_bound = coherence_ranges[i + 1]
        mask = (avg_coh >= lower_bound) & (avg_coh < upper_bound)
        if np.sum(mask) >= 10:  # Ensure there are at least 10 values in the range
            avg_imp_value = np.mean(avg_imp[mask])
            avg_coh_value = np.mean(avg_coh[mask])
            mean_importance.append(avg_imp_value)
            mean_coherence.append(avg_coh_value)
        else:
            mean_importance.append(np.nan)
            mean_coherence.append(np.nan)

    # Compute Pearson correlation between mean_importance and mean_coherence
    mask_valid = ~np.isnan(mean_importance) & ~np.isnan(mean_coherence)
    pearson_corr_mean = pearsonr(np.array(mean_importance)[mask_valid], np.array(mean_coherence)[mask_valid])[0]

    # Compute Pearson correlation between avg_imp and avg_coh
    pearson_corr_avg = pearsonr(avg_coh_1d, avg_imp_1d)[0]

    # Compute x positions as the middle of each range
    x_positions = coherence_ranges[:-1] + 0.05  # Midpoints: 0.05, 0.15, ..., 0.95

    # Plotting the data with the x positions in the middle of the range
    ax.plot(x_positions, mean_importance, label='Mean Importance', color=mycol['orange'], marker='o')

    # Set x-axis ticks and labels
    ax.set_xticks(np.arange(0, 1.1, 0.1))  # Ticks at 0, 0.1, ..., 1
    ax.set_xticklabels([f'{tick:.1f}' for tick in np.arange(0, 1.1, 0.1)], rotation=0, fontsize=12)

    # Set labels and title
    ax.set_xlabel('Coherence', fontsize=14)
    ax.set_ylabel('Mean Importance', fontsize=14)
    ax.set_title('(f) Mean Importance \n in Different Coherence Ranges', fontsize=16)
    ax.grid(axis='both')

    # Display the Pearson correlation values on the plot
    textstr = '\n'.join((
        f'Pearson Corr (Mean): {pearson_corr_mean:.2f}',
        f'Pearson Corr (All): {pearson_corr_avg:.2f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)


    # --------------------------------- end ---------------------------------

    plt.tight_layout()
    if save:
        f_save = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                             f"Pat{pat_id}_pre{pre_itvl}_lime_analysis_sz{sz_id}_{type}_class{target_class}_{n_sample}.png")
        my_save(fig, f_save)
    plt.show()


def analyze_lime_imp_one_3subplots(pat_id, type, pre_itvl=45, target_class=1, n_sample=100, sz_id=-1, save=True):
    # Load data
    f_avg_imp = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                            f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
    avg_imp = np.load(f_avg_imp)
    f_avg_coh = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                            f"Pat{pat_id}_pre{pre_itvl}_lime_avg_coherence_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
    avg_coh = np.load(f_avg_coh)
    avg_coh = np.flipud(avg_coh)

    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(25, 10))

    # First subplot: [0,0]: Average Importance
    ax = axes[0]
    title = "Average Importance"
    plot_avg_imp_lime(avg_imp, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14, contour=True)
    # Plot top 5% values in avg_imp as red dots
    threshold = np.percentile(avg_imp, 95)
    y_coords, x_coords = np.where(avg_imp >= threshold)
    ax.scatter(x_coords, y_coords, color='red', s=10, label='Top 5% values')

    # Second subplot: [0,1]: Distribution of Importance Values
    axes[1].hist(avg_imp.flatten(), bins=50, color=mycol['blue'], alpha=0.7)
    axes[1].set_title('Distribution of Importance Values', fontsize=16)
    axes[1].set_xlabel('Importance', fontsize=14)
    axes[1].set_ylabel('Frequency', fontsize=14)
    axes[1].grid(axis='y')

    # Third subplot: [0,2]: Mean Coherence
    ax = axes[2]
    importance_ranges = np.arange(0, 1.1, 0.1)  # Ranges from 0 to 1 with a step of 0.1
    mean_coherence = []
    coherence_distributions = []

    for i in range(len(importance_ranges) - 1):
        lower_bound = importance_ranges[i]
        upper_bound = importance_ranges[i + 1]
        mask = (avg_imp >= lower_bound) & (avg_imp < upper_bound)
        if np.any(mask):  # Ensure there are values in the range
            avg_coh_value = np.median(avg_coh[mask])
            mean_coherence.append(avg_coh_value)
            coherence_distributions.append(avg_coh[mask])
        else:
            mean_coherence.append(np.nan)
            coherence_distributions.append([])  # Handle empty regions

    # Plot the mean coherence on the primary y-axis
    ax.plot(range(len(mean_coherence)), mean_coherence, label='Mean Coherence', color='blue', marker='o',
            linestyle='--', linewidth=2)
    ax.set_xlabel('Importance Range', fontsize=14)
    ax.set_ylabel('Mean Coherence', fontsize=14, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')

    # Create a secondary y-axis for the box plots
    ax2 = ax.twinx()
    ax2.boxplot(coherence_distributions, positions=range(len(coherence_distributions)), widths=0.6, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'), medianprops=dict(color='darkblue'))
    ax2.set_ylabel('Coherence Distribution', fontsize=14, color='darkblue')
    ax2.tick_params(axis='y', labelcolor='darkblue')

    # Set x-axis labels
    ax.set_xticks(range(len(mean_coherence)))
    ax.set_xticklabels(
        [f'[{importance_ranges[i]:.1f}, {importance_ranges[i + 1]:.1f}]' for i in range(len(importance_ranges) - 1)],
        rotation=45, fontsize=12)
    ax.set_title('Mean Coherence and Distribution for Different Importance Ranges', fontsize=16)
    ax.grid(axis='both')

    plt.tight_layout()
    if save:
        f_save = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                             f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{type}_class{target_class}_{n_sample}_detail.png")
        my_save(fig, f_save)
    plt.show()


def analyze_lime_imp_one_6subplots(pat_id, type, pre_itvl=45, target_class=1, n_sample=100, sz_id=-1, save=True):
    x_tick_labels, x_ticks, y_tick_labels, y_ticks = get_x_y_ticks()

    # Load data
    f_avg_imp = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                         f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
    avg_imp = np.load(f_avg_imp)
    f_avg_coh = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                             f"Pat{pat_id}_pre{pre_itvl}_lime_avg_coherence_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
    avg_coh = np.load(f_avg_coh)
    avg_coh = np.flipud(avg_coh)

    # plot start -----------
    fig, axes = plt.subplots(2, 3, figsize=(25, 15))

    # [0,0]: average importance
    ax = axes[0, 0]
    title = "Average Importance"
    plot_avg_imp_lime(avg_imp, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14, contour=True)
    # Plot top 5% values in avg_imp as red dots
    threshold = np.percentile(avg_imp, 95)
    y_coords, x_coords = np.where(avg_imp >= threshold)
    ax.scatter(x_coords, y_coords, color='red', s=10, label='Top 5% values')

    # [0,1] clusters of the importance matrix
    ax = axes[0, 1]
    reshaped_matrix = avg_imp.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(reshaped_matrix)
    clusters = kmeans.labels_.reshape(avg_imp.shape)

    # Calculate mean importance for each cluster
    cluster_imp_means = np.array([avg_imp[clusters == i].mean() for i in range(3)])
    cluster_coh_means = np.array([avg_coh[clusters == i].mean() for i in range(3)])
    sorted_indices = np.argsort(-cluster_imp_means)
    colors = [mycol['red'], mycol['blue'], 'black']
    rgb_colors = [mcolors.to_rgb(color) for color in colors]  # Convert to RGB
    # Create an RGB image for the clusters
    colored_clusters = np.zeros((*clusters.shape, 3))
    for idx, color in zip(sorted_indices, rgb_colors):
        colored_clusters[clusters == idx] = color
    im = ax.imshow(colored_clusters, interpolation='none')
    ax.set_title('Importance Clusters', fontsize=16)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation=45, fontsize=14)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=14)
    for y in [4, 8, 13, 30]:
        ax.axhline(y=y_ticks[y_tick_labels.index(y)], color='white', linestyle='solid', linewidth=1.5)
    # Add solid gray horizontal lines to separate gamma frequency bands
    for y in [53, 75, 103, 128]:
        ax.axhline(y=y_ticks[y_tick_labels.index(y)], color='lightgrey', linestyle='dashed', linewidth=1.5)
    for x in x_ticks:
        ax.axvline(x=x, color='lightgrey', linewidth=1)

    # Add legend for clustering results
    leg_txt = ['High', 'Medium', 'Low']
    legend_labels = [f'{leg_txt[i]}: Mean={cluster_imp_means[sorted_indices[i]]:.2f}, '
                     f'{int(100* (np.sum(clusters == sorted_indices[i])/avg_imp.size))}%, '
                     f'coh_mean={cluster_coh_means[sorted_indices[i]]:.2f}' for i in range(3)]
    legend_patches = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(3)]
    ax.legend(legend_patches, legend_labels, loc='upper left', fontsize=12)


    # [0, 2]: importance Histogram
    axes[0, 2].hist(avg_imp.flatten(), bins=50, color=mycol['blue'], alpha=0.7)
    axes[0, 2].set_title('Distribution of Importance Values', fontsize=16)
    axes[0, 2].set_xlabel('Importance', fontsize=14)
    axes[0, 2].set_ylabel('Frequency', fontsize=14)
    axes[0, 2].grid(axis='y')


    # [1, 0]: average coherence
    ax = axes[1, 0]
    title = "Average Coherence"
    plot_avg_imp_lime(avg_coh, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14, contour=False)
    # min_val = np.min(avg_coh)
    # max_val = np.max(avg_coh)
    # if max_val > min_val:
    #     normalized_coherence = (avg_coh - min_val) / (max_val - min_val)
    # else:
    #     normalized_coherence = avg_coh  # If all values are the same, keep them unchanged
    #
    # # Optionally, apply a scaling factor to enhance contrast
    # scaling_factor = 5  # Increase contrast
    # normalized_coherence = np.clip(normalized_coherence * scaling_factor, 0, 1)
    #
    # plot_avg_imp_lime(normalized_coherence, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14, contour=False)


    # [1, 1]: Plot the trends of mean coherence and mean importance
    # Calculate averages for coherence and importance for different thresholds
    ax = axes[1, 1]
    thresholds = np.array(list(range(10, 100, 10)) + [95, 99])[::-1]
    legend_info = []
    mean_coherence = []
    mean_importance = []
    for thresh in thresholds:
        imp_threshold = np.percentile(avg_imp, thresh)
        mask = avg_imp >= imp_threshold
        avg_coh_value = np.mean(avg_coh[mask])
        avg_imp_value = np.mean(avg_imp[mask])
        mean_coherence.append(avg_coh_value)
        mean_importance.append(avg_imp_value)

    ax2 = ax.twinx()  # Create a twin Axes sharing the x-axis

    ax.plot(range(len(thresholds)), mean_importance, label='Mean Importance', color='red', marker='x')
    ax2.plot(range(len(thresholds)), mean_coherence, label='Mean Coherence', color='blue', marker='o')

    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'Top {100 - t}%' for t in thresholds], rotation=45, fontsize=12)
    ax.set_ylabel('Mean Importance', fontsize=14)
    ax2.set_ylabel('Mean Coherence', fontsize=14)

    ax.set_title('Mean Coherence and Importance Trends', fontsize=16)

    # Combine legends from both y-axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper center', fontsize=14)
    ax.grid(axis='both')

    # [1, 2]: avg-coh with only top 5% importance regions in color
    ax = axes[1, 2]
    top_5_percent_mask = avg_imp >= np.percentile(avg_imp, 95)
    gray_coherence = np.ones_like(avg_coh) * 0  # gray value (0.5)
    # Apply the mask to keep the top 5% regions in their original values
    highlighted_coherence = np.where(top_5_percent_mask, avg_coh, gray_coherence)
    # Normalize the highlighted coherence to the range [0, 1] for better contrast
    min_val = np.min(highlighted_coherence[top_5_percent_mask])
    max_val = np.max(highlighted_coherence[top_5_percent_mask])
    if max_val > min_val:
        normalized_coherence = (highlighted_coherence - min_val) / (max_val - min_val)
    else:
        normalized_coherence = highlighted_coherence  # If all values are the same, keep them unchanged

    # Optionally, apply a scaling factor to enhance contrast
    scaling_factor = 10  # Increase contrast
    normalized_coherence = np.clip(normalized_coherence * scaling_factor, 0, 1)

    # Plot the enhanced coherence map
    title = 'Coherence (Top 5% Importance Regions)'
    plot_avg_imp_lime(normalized_coherence, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14,
                      contour=False, range01=False)
    plt.tight_layout()
    if save:
        f_save = RST_PATH / (f"figures/Pat{pat_id}/explain_lime/"
                         f"Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{type}_class{target_class}_{n_sample}_detail.png")
        my_save(fig, f_save)
    plt.show()


def compare_noon_midnight(pre_itvl=45, target_class=0, n_sample=500, sz_id=-1, save=True):
    n_row, n_col = 4, 10
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*5))
    noon = 'time-11-14-inter' if target_class==0 else 'time-11-14-pre'
    night = 'time-2-5-inter' if target_class==0 else 'time-2-5-pre'
    axes_list = axes.flatten()
    idx = 0
    for pat_id in PAT_ID_LIST:
        f_all = RST_PATH / f"figures/Pat{pat_id}/explain_lime/Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_top_class{target_class}_{n_sample}.npy"
        imp_noon = np.load(f_all)
        ax = axes_list[idx]
        idx += 1
        title = f"P{pat_id} Importance for Nonseizure"
        plot_avg_imp_lime(imp_noon, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14, contour=True)
    for pat_id in PAT_ID_LIST:
        f_noon = RST_PATH / f"figures/Pat{pat_id}/explain_lime/Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{noon}_class{target_class}_{n_sample}.npy"
        imp_noon = np.load(f_noon)
        ax = axes_list[idx]
        idx += 1
        title = f"P{pat_id} Noon"
        plot_avg_imp_lime(imp_noon, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14, contour=True)
    for pat_id in PAT_ID_LIST:
        f_night = RST_PATH / f"figures/Pat{pat_id}/explain_lime/Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{night}_class{target_class}_{n_sample}.npy"
        imp_night = np.load(f_night)
        ax = axes_list[idx]
        idx += 1
        title = f"P{pat_id} Night"
        plot_avg_imp_lime(imp_night, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14,
                          contour=True)
    for pat_id in PAT_ID_LIST:
        f_noon = RST_PATH / f"figures/Pat{pat_id}/explain_lime/Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{noon}_class{target_class}_{n_sample}.npy"
        imp_noon = np.load(f_noon)
        f_night = RST_PATH / f"figures/Pat{pat_id}/explain_lime/Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_{night}_class{target_class}_{n_sample}.npy"
        imp_night = np.load(f_night)

        diff = imp_noon - imp_night
        # Normalize the difference to 0-1 range
        diff_min = np.min(diff)
        diff_max = np.max(diff)
        normalized_diff = (diff - diff_min) / (diff_max - diff_min) if diff_max != diff_min else diff

        ax = axes_list[idx]
        idx += 1
        title = f"P{pat_id} Noon - Night"
        plot_avg_imp_lime(normalized_diff, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14,
                          contour=True, n_contour=1, contour_list=[0.8])
    # Tight layout after plotting all figures
    plt.tight_layout()
    if save:
        f_save = RST_PATH / (f"figures/"
                             f"lime_importance_noon_night_interical_pre{pre_itvl}_sz{sz_id}_class{target_class}_{n_sample}.png")
        my_save(fig, f_save)
    plt.show()


def compare_imp_pre_inter(pre_itvl=45, target_class=1, n_sample=500, sz_id=-1, save=True):
    n_row, n_col = 4, 10
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*5))
    type_pre = 'top-pre'

    axes_list = axes.flatten()
    idx = 0

    for pat_id in PAT_ID_LIST:
        f_pre = RST_PATH / f"figures/Pat{pat_id}/explain_lime/Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_top_class{target_class}_{n_sample}.npy"
        imp_pre = np.load(f_pre)
        ax = axes_list[idx]
        idx += 1
        title = f"P{pat_id} in both"
        plot_avg_imp_lime(imp_pre, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14, contour=True)
    for pat_id in PAT_ID_LIST:
        f_pre = RST_PATH / f"figures/Pat{pat_id}/explain_lime/Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_top-pre_class{target_class}_{n_sample}.npy"
        imp_pre = np.load(f_pre)
        ax = axes_list[idx]
        idx += 1
        title = f"P{pat_id} in Preictal"
        plot_avg_imp_lime(imp_pre, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14, contour=True)
    for pat_id in PAT_ID_LIST:
        f_inter = RST_PATH / f"figures/Pat{pat_id}/explain_lime/Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_top-inter_class{target_class}_{n_sample}.npy"
        imp_inter = np.load(f_inter)
        ax = axes_list[idx]
        idx += 1
        title = f"P{pat_id} in Interictal"
        plot_avg_imp_lime(imp_inter, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14,
                          contour=True)
    for pat_id in PAT_ID_LIST:
        f_pre = RST_PATH / f"figures/Pat{pat_id}/explain_lime/Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_top-pre_class{target_class}_{n_sample}.npy"
        imp_pre = np.load(f_pre)
        f_inter = RST_PATH / f"figures/Pat{pat_id}/explain_lime/Pat{pat_id}_pre{pre_itvl}_lime_avg_importance_sz{sz_id}_top-inter_class{target_class}_{n_sample}.npy"
        imp_inter = np.load(f_inter)

        diff = imp_pre - imp_inter
        # Normalize the difference to 0-1 range
        diff_min = np.min(diff)
        diff_max = np.max(diff)
        normalized_diff = (diff - diff_min) / (diff_max - diff_min) if diff_max != diff_min else diff

        ax = axes_list[idx]
        idx += 1
        title = f"P{pat_id} Pre - Inter"
        plot_avg_imp_lime(normalized_diff, title, save=False, f_save=None, colorbar=False, ax=ax, ax_fontsize=14,
                          contour=True, n_contour=1, contour_list=[0.8])
    # Tight layout after plotting all figures
    plt.tight_layout()
    if save:
        f_save = RST_PATH / (f"figures/"
                             f"lime_importance_top_pre_inter_pre{pre_itvl}_sz{sz_id}_class{target_class}_{n_sample}.png")
        my_save(fig, f_save)
    plt.show()





    pass
