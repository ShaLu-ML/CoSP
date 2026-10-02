from config import XRAY_FILES
import math
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
    INFO_PATH, FREQ_BANDS, PID_XRAY
from data import compute_coherence, compute_coherence_one_segment
from models import get_model
from util import get_saved_info, get_final_model_id, sigmoid, load_tst_coh, my_save, my_print
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image, lime_tabular

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
from networkx.algorithms import efficiency_measures
import community  # For modularity calculation, install with `pip install python-louvain`

def compute_connectivity_metrics(connectivity_matrix):
    """
    Compute graph-theoretic metrics for a given connectivity matrix.

    Parameters:
    connectivity_matrix (numpy.ndarray): Symmetric matrix representing connectivity strengths.

    Returns:
    dict: Calculated metrics including:
          (1) Degree Distribution: Number of connections per node.
          (2) Clustering Coefficient: How densely connected the neighbors of a node are.
          (3) Characteristic Path Length: Average shortest path length between all node pairs.
          (4) Global Efficiency: Efficiency of information transfer across the whole network.
          (5) Modularity: Measures the division of the network into communities.
    """

    # Create a graph from the connectivity matrix
    G = nx.Graph()
    n_nodes = connectivity_matrix.shape[0]

    # Add nodes
    G.add_nodes_from(range(n_nodes))

    # Add weighted edges from the connectivity matrix (assuming undirected and symmetric)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Only upper triangle for efficiency
            weight = connectivity_matrix[i, j]
            if weight > 0:  # Only add edges with positive weight
                G.add_edge(i, j, weight=weight)

    # Compute Degree Distribution
    degree_distribution = [degree for node, degree in G.degree(weight='weight')]

    # Compute Clustering Coefficient
    clustering_coeff = nx.clustering(G, weight='weight')
    avg_clustering_coeff = np.mean(list(clustering_coeff.values()))

    # Compute Characteristic Path Length
    if nx.is_connected(G):
        char_path_length = nx.average_shortest_path_length(G, weight='weight')
    else:
        char_path_length = np.nan  # Use NaN if the graph is disconnected

    # Compute Global Efficiency
    global_efficiency = efficiency_measures.global_efficiency(G)

    # Compute Modularity
    partition = community.best_partition(G)  # Returns a dictionary of node partitions
    modularity = community.modularity(partition, G)

    # Aggregate results
    metrics = {
        "degree_distribution": degree_distribution,
        "avg_clustering_coefficient": avg_clustering_coeff,
        "characteristic_path_length": char_path_length,
        "global_efficiency": global_efficiency,
        "modularity": modularity
    }

    return metrics

def plot_on_xray(pat_id, channel_pairs, values, title=None, save=False, f_save=None, linestyle='solid', colorbar=True,
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
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Add colorbar to indicate the strength of the connection, fixed to range [0, 1]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cax)  # Colorbar shows range [0, 1],  label='Coherence'
    if title: ax.set_title(title)
    ax.axis('off')
    if save:
        f_save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f_save, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Plot saved to {f_save}")
        plt.close()
    else:
        plt.show()



def plot_electrode_connection(pat_id, pre_itvl, sz_id, type, target_class, n_sample, mean_by, n_top, save=True,
                              colorbar=True):
    f_coh = RST_PATH / (f"figures/Pat{pat_id}/coh_average/"
                        f"Pat{pat_id}_pre{pre_itvl}_avg_coherence_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
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
    # title = f"P{pat_id}, top{n_top}, {mean_by}, {type}"
    title = None
    save_path = RST_PATH / (f"figures/Pat{pat_id}/connect/"
                            f"Pat{pat_id}_connectivity_pre{pre_itvl}_sz{sz_id}_{type}"
                            f"_class{target_class}_{n_sample}"
                            f"_freq-{mean_by}_top{n_top}.png")
    plot_on_xray(pat_id, top_n_channels, top_n_values, title=title, save=save, f_save=save_path, colorbar=colorbar)
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
    save_path = RST_PATH / (f"figures/Pat{pat_id}/connect/detail"
                            f"Pat{pat_id}_connect-change_pre{pre_itvl}_sz{sz_id}_{type}"
                            f"_class{target_class}_{n_sample}"
                            f"_freq-{mean_by}_top{n_top}.png")
    plot_on_xray(pat_id, top_n_channels, top_n_values, title, save=save, f_save=save_path, linestyle='dotted')
    print('Done!')


def plot_connect_comparison(pid, n_sample, types, save=False):
    band_list = list(FREQ_BANDS.keys())[::-1]
    phase_count = len(types)
    top_list = [5, 5, 10, 10, 20, 20, 30, 30] if phase_count == 2 else [5, 5, 5, 10, 10, 10, 20, 20, 20, 30, 30, 30]

    n_row = len(band_list)
    n_col = len(top_list)

    fig, axes = plt.subplots(
        n_row, n_col, figsize=(30, 12 if phase_count == 3 else 15),
        gridspec_kw={'hspace': 0, 'wspace': 0}  # Reduce gaps
    )

    # Add titles for columns (top_list)
    for i, (ax, col_name) in enumerate(zip(axes[0], top_list)):
        phase_idx = i % phase_count
        phase = types[phase_idx]
        ax.set_title(f'P{pid}, Top{col_name}, {phase}', fontsize=12)

    folder_path = RST_PATH / f"figures/Pat{pid}/connect/detail/"
    for idx, ax in enumerate(axes.flat):
        idx_band = idx // n_col
        idx_top = idx % n_col
        n_top = top_list[idx_top]
        freq = band_list[idx_band]
        phase_idx = idx_top % phase_count
        type = ['top-inter', 'top-pre', 'ictal'][phase_idx] if phase_count == 3 else ['top-inter', 'top-pre'][phase_idx]

        f_img = folder_path / (f"Pat{pid}_connectivity_pre{pre_itvl}_sz-1_{type}"
                               f"_classNone_{n_sample}_freq-{freq}_top{n_top}.png")
        assert f_img.exists(), f"ERROR: {f_img} doesn't exist"
        img = Image.open(f_img)
        ax.imshow(img)
        ax.axis('off')  # Turn off axis for each image

    # Add row titles (band_list) as text outside the grid
    for i, band_name in enumerate(band_list):
        fig.text(
            0.04,  # x-coordinate (outside the grid)
            0.9 - (i / n_row) * 0.96,  # Adjust y-coordinate for alignment
            band_name, va='center', ha='center', fontsize=16, rotation=90
        )

    # Add vertical lines to separate phase columns
    separation_lines = [0.286, 0.523, 0.761]
    for line_x in separation_lines:
        line = mlines.Line2D([line_x, line_x], [0, 1], color='orange', linewidth=6 if phase_count == 3 else 8,
                             transform=fig.transFigure, linestyle='-')
        fig.add_artist(line)

    plt.tight_layout(pad=0, rect=[0.05, 0, 1, 1])
    if save:
        save_folder = RST_PATH / f"figures/Pat{pid}/connect/"
        f_save = save_folder / (f"Pat{pid}_connectivity_compare_pre45_sz-1_phases{phase_count}_{n_sample}.png")
        plt.savefig(f_save)
        print(f"{f_save} has been saved.")
    else:
        plt.show()
