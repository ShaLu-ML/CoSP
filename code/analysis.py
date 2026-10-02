import pickle
import sys

import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy.ma as ma
import torch
from scipy.stats import wilcoxon
from config import INFO_PATH, WORK_PATH, RST_PATH, STAT_PATH, CONFIG, PAT_LIST, FINAL_INFO, PAT_LIST_ALL, PAT_ALL_DAYS, \
    PAT_ID_LIST, THD_APPROACHES, PRE_ITVL_TESTED, N_FREQ, N_CHN_PAIR
from data import gen_seg_list_pat, load_segment_eeg_2d, split_segment_list, load_segment_eeg_3d, compute_coherence, \
    path_to_time, compute_coherence_one_segment
from util import load_mat_to_np, get_sz_info_from_epochs, sigmoid, smooth_logit_per_sz, get_final_model_id, \
    get_opt_thd, get_saved_info, load_tst_coh, my_save
import numpy as np
from scipy import signal
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.gridspec import GridSpec
# pd.set_option('display.float_format', lambda x: '%.f' % x)
import datetime
import matplotlib.dates as mdates

col = {'inter':'green',
       'pre': 'orange',
       'diff': 'blue',
       'ictal': 'red'}

# Define frequency regions and their colors
frequency_regions = [(0, 4), (4, 8), (8, 12), (12, 40), (40, 170)]
region_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lavender']

mycol = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf'
}


def weight_compare(pre_path, inter_path, ictal_path, output_prefix="channel_weights"):
    pre_weights = pd.read_csv(pre_path, index_col=0)
    inter_weights = pd.read_csv(inter_path, index_col=0)
    ictal_weights = pd.read_csv(ictal_path, index_col=0)

    # Choose a colormap and extract colors
    colormap = cm.get_cmap('viridis', 3)  # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    colors = [colormap(i) for i in range(3)]
    for i, idx in enumerate(pre_weights.index):
        pre_one, inter_one, ictal_one = pre_weights.loc[idx], inter_weights.loc[idx], ictal_weights.loc[idx]
        # Setting the positions and width for the bars
        pos = np.arange(len(pre_one))
        bar_width = 0.25
        plt.figure(figsize=(14, 6))
        plt.bar(pos, pre_one, bar_width, label='preictal', color=colors[0])
        plt.bar(pos + bar_width, inter_one, bar_width, label='interictal', color=colors[1])
        plt.bar(pos + 2 * bar_width, ictal_one, bar_width, label='ictal', color=colors[2])

        plt.xlabel('Channels')
        plt.ylabel('Contribution (Weights)')
        plt.title(f'Comparison of contributions for predicting Channel {i+1}')
        plt.xticks(pos + bar_width, [f'C{j}' for j in list(range(1,i+1)) + list(range(i+2,17)) ])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(INFO_PATH/f'examples/channel_weights/{output_prefix}_{idx}.png')
    print('channel weight bar plots have been saved to INFO_PATH/examples/channel_weights/.')




def draw_coherence_2channel():
    pat = CONFIG['patient']
    fs = 400  # Example value in Hz
    ch1 = 'Ch8'
    ch2 = 'Ch9'
    # preictal
    CONFIG['trn_label'] = 'pre'
    gen_seg_list_pat(pat)
    trn_list, _, _ = split_segment_list(pat, CONFIG['trn_label'])
    X_train, _ = load_segment_eeg_2d(trn_list, CONFIG, 'trn', sz_id=None)
    ts1 = X_train[ch1].to_numpy()
    ts2 = X_train[ch2].to_numpy()
    frequencies1, coherence1 = signal.coherence(ts1, ts2, fs=fs)
    # interictal
    CONFIG['trn_label'] = 'inter'
    gen_seg_list_pat(pat)
    trn_list, _, _ = split_segment_list(pat, CONFIG['trn_label'])
    X_train, _ = load_segment_eeg_2d(trn_list, CONFIG, 'trn', sz_id=None)
    ts1 = X_train[ch1].to_numpy()
    ts2 = X_train[ch2].to_numpy()
    frequencies2, coherence2 = signal.coherence(ts1, ts2, fs=fs)
    # Plot the coherence
    plt.figure()
    # plt.semilogy(frequencies1, coherence1, label='preictal')
    # plt.semilogy(frequencies2, coherence2, label='interictal')
    plt.plot(frequencies1, coherence1, label='preictal')
    plt.plot(frequencies2, coherence2, label='interictal')
    plt.title(f'Coherence between Channels {ch1} and {ch2}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Coherence')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.savefig(STAT_PATH / f"{pat}/{pat}_coherence_{ch1}_{ch2}.png")
    plt.show()


def draw_coherence_average(freq_range='0-170', log=False):
    pat = CONFIG['patient']
    preprocess = 'coh-0-170-stack'
    n_channels = CONFIG['n_chn']

    coherence_matrix_pre = get_coherence_matrix('trn', 'pre')
    coherence_matrix_inter = get_coherence_matrix('trn', 'inter')
    coherence_matrix_ictal = get_coherence_matrix('trn', 'ictal')
    f = np.loadtxt(STAT_PATH / f"{pat}/coherence_{preprocess}_none_frequencies.csv")

    print('ploting...')
    if not freq_range.startswith('full'):
        low_thd = int(freq_range.split('-')[0])
        high_thd = int(freq_range.split('-')[1])
        high_idx = np.where((f >= low_thd) & (f < high_thd))
        f = f[high_idx]
    else:
        high_idx = np.arange(len(f))

    # Plot average coh-map of preictal
    plt.figure(figsize=(12, 6))
    for region, color in zip(frequency_regions, region_colors):
        plt.axvspan(region[0], region[1], color=color, alpha=0.3)
    for i in range(n_channels):
        one = np.mean(coherence_matrix_pre[i, :, :], axis=0)[high_idx]
        if i ==0:
            plt.plot(f,one, color=col['pre'], label='preictal')
        else:
            plt.plot(f, one, color=col['pre'])

    # Plot average coh-map of interictal
    for i in range(n_channels):
        one = np.mean(coherence_matrix_inter[i, :, :], axis=0)[high_idx]
        if i == 0:
            plt.plot(f,one, color=col['inter'], alpha=0.5, label='interictal')
        else:
            plt.plot(f, one, color=col['inter'], alpha=0.5)
    plt.legend()
    plt.savefig(STAT_PATH / f"{pat}/{pat}_avg_coherence_{preprocess}_none_{freq_range}_pre-inter.png")
    plt.show()

    # Plot triple (interictal, preictal, ictal)
    plt.figure(figsize=(12, 6))
    fig, axes = plt.subplots(4, 4, figsize=(24, 15))  # Adjust the size as needed
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        # Add shaded areas for each frequency region
        for region, color in zip(frequency_regions, region_colors):
            ax.axvspan(region[0], region[1], color=color, alpha=0.3)
        # Plot coherence for each channel
        if i < n_channels:
            pre_one = np.mean(coherence_matrix_pre[i, :, :], axis=0)[high_idx]
            if log:
                ax.semilogy(f, pre_one, label='preictal (p)', color=col['pre'])
            else:
                ax.plot(f, pre_one, label='preictal (p)', color=col['pre'])

            inter_one = np.mean(coherence_matrix_inter[i, :, :], axis=0)[high_idx]
            if log:
                ax.semilogy(f, inter_one, label='interictal (n)', color=col['inter'])
            else:
                ax.plot(f, inter_one, label='interictal (n)', color=col['inter'])


            ictal_one = np.mean(coherence_matrix_ictal[i, :, :], axis=0)[high_idx]
            if log:
                ax.semilogy(f, ictal_one, label='ictal (c)', color=col['ictal'])
            else:
                ax.plot(f, ictal_one, label='ictal (c)', color=col['ictal'])

            pn_corr = np.corrcoef(inter_one, pre_one)[0,1]
            cp_corr = np.corrcoef(ictal_one, pre_one)[0,1]
            cn_corr = np.corrcoef(ictal_one, inter_one)[0,1]
            ax.set_title(f'Channel {i + 1} (corr: pn={pn_corr:.2f} cp={cp_corr:.2f} cn={cn_corr:.2f})')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Average Coherence')
            ax.legend()
        else:
            ax.axis('off')  # Turn off axis for any unused subplots
    plt.tight_layout()
    if log:
        plt.savefig(STAT_PATH / f"{pat}/{pat}_avg_coherence_{preprocess}_none_triple_{freq_range}_log.png")
    else:
        plt.savefig(STAT_PATH / f"{pat}/{pat}_avg_coherence_{preprocess}_none_triple_{freq_range}.png")
    # plt.show()

    # Plot diff-ictal
    fig, axes = plt.subplots(4, 4, figsize=(24, 15))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        # Add shaded areas for each frequency region
        for region, color in zip(frequency_regions, region_colors):
            ax.axvspan(region[0], region[1], color=color, alpha=0.3)

        # Plot coherence for each channel
        if i < n_channels:
            # diff = np.mean(np.abs(coherence_matrix_inter[i, :, :] - coherence_matrix_pre[i, :, :]),
            #                axis=0)[high_idx]
            diff = np.mean((coherence_matrix_inter[i, :, :] - coherence_matrix_pre[i, :, :]),
                           axis=0)[high_idx]
            # normalize to (0, 1)
            diff = (diff - np.min(diff))/(np.max(diff) - np.min(diff))
            ax.plot(f, diff, label='abs-diff', color=col['diff'])
            # ax.semilogy(f, diff, label='abs-diff')
            ictal_one = np.mean(coherence_matrix_ictal[i, :, :], axis=0)[high_idx]
            ictal_one = (ictal_one - np.min(ictal_one))/(np.max(ictal_one) - np.min(ictal_one))
            ax.plot(f, ictal_one, label='ictal', color=col['ictal'])
            # ax.semilogy(f, ictal_one, label='ictal')

            id_corr = np.corrcoef(ictal_one, diff)[0, 1]
            ax.set_title(f'Channel {i + 1} (corr:{id_corr:.2f})')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Normalized Coherence Difference and Coherence')
            ax.legend()
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(STAT_PATH / f"{pat}/{pat}_avg_coherence_{preprocess}_none_diff_{freq_range}_normalized.png")
    # plt.show()
    print('Done!')


def get_coherence_matrix(data_type, label):
    pat = CONFIG['patient']
    fs = CONFIG['freq']  # Sampling frequency
    preprocess = 'coh-0-170-stack'
    n_channels = CONFIG['n_chn']

    f_name = STAT_PATH / f"{pat}/coherence_{preprocess}_none_matrix_{label}.csv"
    if not f_name.exists():
        print(f'\nComputing coherecnce for {label}...')
        coh = compute_coherence_average(pat, data_type, label, fs, n_channels, f_name)
        coherence_matrix = coh.to_numpy().reshape(n_channels, n_channels, -1)
    else:
        coherence_matrix = pd.read_csv(f_name).to_numpy().reshape(n_channels, n_channels, -1)
    return coherence_matrix


def compute_coherence_average(pat, data_type, label, fs, n_channels, f_name):
    preprocess = 'coh-0-170-stack'
    gen_seg_list_pat(pat)
    trn_list, val_list, tst_list = split_segment_list(pat, label)
    list_dict = {'trn': trn_list, 'val': val_list, 'tst': tst_list}
    eeg, _ = load_segment_eeg_2d(list_dict[data_type], CONFIG, data_type, sz_id=None)
    # Convert DataFrame to numpy array if necessary
    data = eeg.to_numpy() if isinstance(eeg, pd.DataFrame) else eeg
    coherence_matrix, f = compute_coherence(data, fs, n_channels, CONFIG['detrend'])

    # Save the coherence matrix
    np.savetxt(STAT_PATH / f"{pat}/coherence_{preprocess}_none_frequencies.csv", np.array(f))
    # Flattening the 3D matrix to 2D for saving in CSV, with frequency information lost
    df = pd.DataFrame(coherence_matrix.reshape(n_channels, -1))
    df.to_csv(f_name, index=False)
    print(f"File {f_name.name} has been saved to {f_name.parent}.")

    return df


def compute_frequency_psd(data, fs, n_channels):
    """
    Compute the frequency for each channel in the data.

    Parameters:
    - data: 2D numpy array of shape (n_timesteps, n_channels)
    - fs: Sampling frequency
    - n_channels: Number of channels

    Returns:
    - frequency_matrix: 2D numpy array of shape (n_channels, n_frequencies)
    """
    frequency_matrix = np.zeros((n_channels, int(np.floor(data.shape[0] / 2) + 1)))

    for i in range(n_channels):
        f, Pxx = signal.periodogram(data[:, i], fs)
        frequency_matrix[i, :] = Pxx

    return f, frequency_matrix


def draw_frequency_psd():
    pat = CONFIG['patient']
    fs = CONFIG['freq']  # Sampling frequency
    n_channels = CONFIG['n_chn']

    pre_name = STAT_PATH / f"{pat}/frequency_psd_pre.csv"
    if not pre_name.exists():
        CONFIG['trn_label'] = 'pre'
        gen_seg_list_pat(pat)
        trn_list, _, _ = split_segment_list(pat, CONFIG['trn_label'])
        eeg_trn, _ = load_segment_eeg_2d(trn_list, CONFIG, 'trn', sz_id=None)
        # Convert DataFrame to numpy array if necessary
        eeg_pre = eeg_trn.to_numpy() if isinstance(eeg_trn, pd.DataFrame) else eeg_trn
        print("Computing PSD for preictals...")
        f_pre, freq_matrix_pre = compute_frequency_psd(eeg_pre, fs, n_channels)
        # Save frequency data to CSV
        np.savetxt(pre_name, freq_matrix_pre, delimiter=",")
        np.savetxt(STAT_PATH / f"{pat}/frequency_pre.csv", f_pre, delimiter=",")
    else:
        print(f"load {pre_name.name}")
        freq_matrix_pre = np.loadtxt(pre_name, delimiter=",")
        f_pre = np.loadtxt(STAT_PATH / f"{pat}/frequency_pre.csv", delimiter=",")

    inter_name = STAT_PATH / f"{pat}/frequency_psd_inter.csv"
    if not inter_name.exists():
        CONFIG['trn_label'] = 'inter'
        gen_seg_list_pat(pat)
        trn_list, _, _ = split_segment_list(pat, CONFIG['trn_label'])
        eeg_trn, _ = load_segment_eeg_2d(trn_list, CONFIG, 'trn', sz_id=None)
        # Convert DataFrame to numpy array if necessary
        eeg_inter = eeg_trn.to_numpy() if isinstance(eeg_trn, pd.DataFrame) else eeg_trn

        # Load or compute frequency data for preictal and interictal
        print("Computing PSD for interictals...")
        f_inter, freq_matrix_inter = compute_frequency_psd(eeg_inter, fs, n_channels)
        np.savetxt(inter_name, freq_matrix_inter, delimiter = ",")
        np.savetxt(STAT_PATH / f"{pat}/frequency_inter.csv", f_inter, delimiter=",")
    else:
        print(f"load {inter_name.name}")
        freq_matrix_inter = np.loadtxt(inter_name, delimiter=",")
        f_inter = np.loadtxt(STAT_PATH / f"{pat}/frequency_inter.csv", delimiter=",")

    # Plotting
    print('Ploting...')
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < n_channels:
            ax.plot(f_pre, freq_matrix_pre[i, :], label='preictal', alpha=0.5)
            ax.plot(f_inter, freq_matrix_inter[i, :], label='interictal', alpha=0.5)
            ax.set_title(f'Channel {i + 1}')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Power Spectral Density')
            ax.legend()
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(STAT_PATH / f"{pat}/{pat}_frequency_psd_comparison.png")
    plt.show()


def compute_fft(data, fs):
    """
    Compute the FFT of the data.

    Parameters:
    - data: 2D numpy array of shape (n_timesteps, n_channels)
    - fs: Sampling frequency

    Returns:
    - freqs: Frequencies corresponding to the FFT
    - fft_magnitudes: Magnitudes of the FFT for each channel
    """
    # n_timesteps = data.shape[0]
    n_timesteps = 40000
    data = data[:n_timesteps]
    freqs = np.fft.rfftfreq(n_timesteps, 1/fs)
    fft_magnitudes = np.fft.rfft(data, axis=0)
    # fft_magnitudes = np.abs(np.fft.rfft(data, axis=0))
    return freqs, fft_magnitudes


def draw_frequency_fft():
    pat = CONFIG['patient']
    fs = CONFIG['freq']  # Sampling frequency
    pre_name = STAT_PATH / f"{pat}/preictal_fft.csv"
    pre_ft = STAT_PATH / f"{pat}/frequency_fft_pre.csv"
    if not (pre_name.exists() and pre_ft.exists()):
        CONFIG['trn_label'] = 'pre'
        gen_seg_list_pat(pat)
        trn_list, _, _ = split_segment_list(pat, CONFIG['trn_label'])
        eeg_trn, _ = load_segment_eeg_2d(trn_list, CONFIG, 'trn', sz_id=None)
        # Convert DataFrame to numpy array if necessary
        eeg_pre = eeg_trn.to_numpy() if isinstance(eeg_trn, pd.DataFrame) else eeg_trn
        # Load or compute frequency data for preictal and interictal
        print("Computing FFT for preictals...")
        freqs_pre, preictal_fft = compute_fft(eeg_pre, fs)
        # Save frequency data to CSV
        np.savetxt(pre_name, preictal_fft, delimiter=",")
        np.savetxt(pre_ft, freqs_pre, delimiter=",")
    else:
        print(f"Loading {pre_name.name}... ")
        preictal_fft = np.genfromtxt(pre_name, delimiter=",", dtype='complex')
        # preictal_fft = np.loadtxt(pre_name, delimiter=",")
        print('shape of preictal_fft:', preictal_fft.shape)
        freqs_pre = np.loadtxt(pre_ft, delimiter=",")

    inter_name = STAT_PATH / f"{pat}/interictal_fft.csv"
    inter_ft = STAT_PATH / f"{pat}/frequency_fft_inter.csv"
    if not (inter_name.exists() and inter_ft.exists()):
        CONFIG['trn_label'] = 'inter'
        gen_seg_list_pat(pat)
        trn_list, _, _ = split_segment_list(pat, CONFIG['trn_label'])
        eeg_trn, _ = load_segment_eeg_2d(trn_list, CONFIG, 'trn', sz_id=None)
        # Convert DataFrame to numpy array if necessary
        eeg_inter = eeg_trn.to_numpy() if isinstance(eeg_trn, pd.DataFrame) else eeg_trn

        # Load or compute frequency data for preictal and interictal
        print("Computing FFT for interictals...")
        freqs_inter, interictal_fft = compute_fft(eeg_inter, fs)
        np.savetxt(inter_name, interictal_fft, delimiter=",")
        np.savetxt(inter_ft, freqs_inter, delimiter=",")
    else:
        print(f"Loading {inter_name.name}... ")
        # interictal_fft = np.loadtxt(inter_name, delimiter=",")
        interictal_fft = np.genfromtxt(inter_name, delimiter=",", dtype='complex')

        print('shape of interictal_fft:', interictal_fft.shape)
        freqs_inter = np.loadtxt(inter_ft, delimiter=",")

    # Plot the frequency and magnitude for comparison
    print('Ploting...')
    n_channels = preictal_fft.shape[1]
    # fig, axes = plt.subplots(n_channels, 1, figsize=(10, 2 * n_channels))
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    axes = axes.flatten()
    for i in range(n_channels):
        axes[i].plot(freqs_pre, preictal_fft[:, i], label='Preictal', alpha=0.5)
        axes[i].plot(freqs_inter, interictal_fft[:, i], label='Interictal', alpha=0.5)
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('Magnitude')
        axes[i].set_title(f'Channel {i + 1}')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(STAT_PATH / f"{pat}/{pat}_FFT_comparison.png")
    plt.show()


def draw_coherence_segments():
    pat = CONFIG['patient']
    fs = CONFIG['freq']
    preprocess = 'coh-0-170-stack'
    n_channels = CONFIG['n_chn']
    # load eeg_pre
    seg_list, _, _ = split_segment_list(pat, 'pre')
    CONFIG['batch_size'] = 1
    eeg_pre, _ = load_segment_eeg_3d(seg_list[:20], CONFIG, 'trn', sz_id=None)
    eeg_pre = np.array(eeg_pre)
    # load eeg_inter
    seg_list, _, _ = split_segment_list(pat, 'inter')
    eeg_inter, _ = load_segment_eeg_3d(seg_list[:20], CONFIG, 'trn', sz_id=None)
    eeg_inter = np.array(eeg_inter)
    # plot
    fig, axes = plt.subplots(4, 4, figsize=(24, 15))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        # Add shaded areas for each frequency region
        for region, color in zip(frequency_regions, region_colors):
            ax.axvspan(region[0], region[1], color=color, alpha=0.3)

        if i < 8:
            eeg_one = eeg_pre[i].reshape(-1, 16)
            coh_one, f = compute_coherence(eeg_one, fs, n_channels, CONFIG['detrend'])

            for j in range(n_channels):
                one = np.mean(coh_one[j, :, :], axis=0)[f < 170]
                if j == 0:
                    ax.plot(f[f < 170], one, color=col['pre'], label="preinctal")
                else:
                    ax.plot(f[f < 170], one, color=col['pre'])

        if i >= 8:
            eeg_one = eeg_inter[i].reshape(-1, 16)
            coh_one, f = compute_coherence(eeg_one, fs, n_channels, CONFIG['detrend'])
            for j in range(n_channels):
                one = np.mean(coh_one[j, :, :], axis=0)[f < 170]

                if j == 0:
                    ax.plot(f[f < 170], one, color=col['inter'], label="interictal", alpha=0.5)
                else:
                    ax.plot(f[f < 170], one, color=col['inter'], alpha=0.5)
        ax.set_ylim([0, 0.5])
        ax.legend()
    # plt.savefig(STAT_PATH / f"{pat}/{pat}_coherence_segments.png")
    plt.show()
    # print(f"{pat}_coherence_segments.png has been saved.")


def draw_coh_heatmap_combined(pat):
    f_coh = STAT_PATH / f"{pat}/coherence_raw_none_matrix_inter.csv"
    coh_inter = pd.read_csv(f_coh).to_numpy().reshape(16, 16, -1)
    f_coh = STAT_PATH / f"{pat}/coherence_raw_none_matrix_pre.csv"
    coh_pre = pd.read_csv(f_coh).to_numpy().reshape(16, 16, -1)
    freqs = np.loadtxt(STAT_PATH / f"{pat}/coherence_raw_none_frequencies.csv")
    sel_idx = np.array(list(range(1, 100, 2))[:5] + list(range(20, 100, 5))[:5] + list(range(40, 100, 10))[:6])
    fig, axes = plt.subplots(4, 4, figsize=(25, 25), constrained_layout=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        idx = sel_idx[i]
        # Mask for upper and lower parts
        mask_upper = np.tri(16, k=-1, dtype=bool)
        mask_lower = ~mask_upper

        one_inter = coh_inter[:, :, idx].reshape(16, 16)
        one_pre = coh_pre[:, :, idx].reshape(16, 16)
        for j in range(16):
            one_inter[j, j] = 0
            one_pre[j, j] = 0

        one_inter = ma.masked_array(one_inter, mask=mask_lower)
        one_pre = ma.masked_array(one_pre, mask=mask_upper)

        ax.imshow(one_inter, cmap='Greens', extent=[0, 16, 0, 16])
        ax.imshow(one_pre, cmap='Reds', extent=[0, 16, 0, 16])

        # ax.plot([0, 15], [0, 15], color='gray', linestyle='dash', linewidth=3)
        # im = ax.imshow(coh_one, cmap='viridis')
        ax.set_title(f'Heatmap of Frequency {freqs[idx]:.1f} Hz')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
    plt.savefig(STAT_PATH / f"{pat}/{pat}_coherence_heatmap.png")
    plt.show()



def load_coherence_dict(pat, data_type):
    if data_type == 'tst':
        f_key = f'10_185_5_1_non-overlap_non-overlap_10_coh-0-170-stack_{data_type}.pkl'
    else:
        f_key = f'_non-overlap_non-overlap_10_coh-0-170-stack_{data_type}.pkl'
    f_folder = WORK_PATH / f'data/{pat}'
    f_coh = [f for f in f_folder.iterdir() if f_key in f.name]
    if len(f_coh) < 1:
        print(f"ERROR: coherence file does not exist: data_type={data_type}.")
        sys.exit(1)
    else:
        f_coh = f_coh[0]
    # load coherence
    with open(f_coh, 'rb') as f:
        print(f"loading file {f_coh.name}...")
        coh = pickle.load(f)

    return coh



def draw_coh_stack_plots(pat, n_min, channel_average=False):
    print(f"\nPatient {pat}: drawing stacked channel coherence plots with {n_min} minutes before seizure onset.")
    n_freq = 109

    # load coherence of test set
    data_type = 'tst'
    coh_tst = load_coherence_dict(pat, data_type)
    epochs_tst = np.array(list(coh_tst.keys())).astype(int)
    ids_tst, types_tst = get_sz_info_from_epochs(pat, epochs_tst, data_type)

    # load coherence of ictal
    data_type = 'ictal'
    coh_ictal = load_coherence_dict(pat, data_type)
    epochs_ictal = np.array(list(coh_ictal.keys()))
    ids_ictal, _ = get_sz_info_from_epochs(pat, epochs_ictal, data_type)

    if channel_average:
        for key, val in coh_tst.items():
            coh_tst[key] = val.mean(dim=1).reshape(-1, 1)
        for key, val in coh_ictal.items():
            coh_ictal[key] = val.mean(dim=1).reshape(-1, 1)

    # only plot seizure in test set
    for id in np.unique(ids_tst):
        # id = np.unique(ids_tst)[0]

        sz_type = types_tst[np.where(ids_tst==id)[0]][0]
        x_time = n_min  # Total time

        # get the coherence before a seizure onset, item shape: [109, 120] (n_freq, n_channel_pair)
        epochs_tst_one = np.sort(epochs_tst[np.where(ids_tst == id)])
        coh_b4_sz = [coh_tst[key].cpu().numpy() for key in epochs_tst_one][-(5 * n_min):]
        # get the coherence in seizure
        epochs_ictal_one = np.sort(epochs_ictal[np.where(ids_ictal == id)])
        coh_in_sz = [coh_ictal[key].cpu().numpy() for key in epochs_ictal_one]

        # concatenate for plotting
        coh_plt = np.concatenate(coh_b4_sz + coh_in_sz, axis=1)

        # Plotting the concatenated array
        plt.figure(figsize=(15, 4))  # You can adjust the figure size as needed
        plt.imshow(coh_plt, aspect='auto', interpolation='none', cmap='viridis')
        plt.gca().invert_yaxis()  # To reverse the y-axis
        # plt.yticks([0, n_freq/4, n_freq/2, n_freq * 0.75, n_freq],
        #            [0, int(170/4), int(170/2), int(170 * 0.75), 170])
        # Assuming n_freq is defined
        n_freq_positions = [0, n_freq / 4, n_freq / 2, n_freq * 0.75, n_freq]
        n_freq_labels = [0, 40, 80, 120, 170]
        plt.yticks(n_freq_positions, n_freq_labels)

        for pos in n_freq_positions:
            plt.axhline(y=pos, color='gray', linestyle='--', linewidth=0.5)
        plt.colorbar()

        # Plot a red vertical line to show seizure onset
        sz_onset = len(coh_b4_sz) if channel_average else 120 * len(coh_b4_sz) # Each item is 120 columns wide
        plt.axvline(x=sz_onset, color='r', linestyle='-', linewidth=0.6)
        plt.plot(sz_onset, plt.gca().get_ylim()[1]+2, marker='v', color='r', markersize=10)

        ticks_every_10_min = np.linspace(0, sz_onset - 1, x_time // 10 + 1,
                                         dtype=int)  # Calculate positions for every 10 minutes
        x_tick_text = [f'-{x_time - i * 10}m' for i in range(len(ticks_every_10_min))]
        x_tick_text[-1] = 'onset'
        plt.xticks(ticks_every_10_min, x_tick_text)
        plt.title(f"Coherece of {n_min} minutes before Seizure {id} (type: {sz_type}, channel_average:{channel_average})")
        plt.ylabel("Frequency")
        plt.xlabel("Minutes to Seizure Onset")

        plt.tight_layout()
        f_fig = RST_PATH/f"figures/{pat}/coh_stack/{pat}_coh_stack_minutes-{n_min}_sz-{id}_avg-{channel_average}_type-{sz_type}.png"
        f_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f_fig, bbox_inches='tight')
        plt.close()
        # plt.show()
        print(f"Plots {f_fig.name} have been saved to {f_fig.parent}.")


def convert_coh_to_3d(coh_2d, n_chn=16, n_freq=109):
    if coh_2d.shape[0] == n_freq:
        coh_2d = coh_2d.T

    coh_3d = np.zeros((n_chn, n_chn, n_freq))
    index = 0  # Index for the current row in coh_2d
    for i in range(1, n_chn):
        for j in range(i):
            coh_3d[j, i, :] = coh_2d[index, :]
            coh_3d[i, j, :] = coh_2d[index, :]
            index += 1

    # Set the diagonal elements to 1 for each frequency slice
    # for f in range(n_freq):
    #     np.fill_diagonal(coh_3d[:, :, f], 1)
    return coh_3d

# def plot_coh_heatmap(coh, title, show=True, vmax=1, vmin=0):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(coh, cmap='viridis', annot=False,
#                 xticklabels=[channel_names[i] for i in order],
#                 yticklabels=[channel_names[i] for i in order],
#                 vmin=vmin, vmax=vmax)
#     plt.title(title)
#     plt.xlabel('Channel (Ordered)')
#     plt.ylabel('Channel (Ordered)')
#     if show:
#             plt.show()

def plot_coh_heatmap(coh, title, ax, vmax=1, vmin=0, order=None, cbar_ax=None):
    if order is None: order = range(16)
    channel_names = [f'c{i + 1}' for i in range(16)]
    sns.heatmap(coh, cmap='viridis', annot=False,
                xticklabels=[channel_names[i] for i in order],
                yticklabels=[channel_names[i] for i in order],
                vmin=vmin, vmax=vmax, ax=ax, cbar=False, square=True)
    ax.set_title(title)
    # ax.set_xlabel('Channel (Ordered)')
    # ax.set_ylabel('Channel (Ordered)')

    # Only draw colorbar for the last subplot to avoid redundancy
    if cbar_ax is not None:
        # Create a mappable object to manually draw the colorbar
        mappable = ax.collections[0]
        plt.colorbar(mappable, cax=cbar_ax)


def wilcox_test(cosp_scores, compare_scores):
    """Compare aligned, caller-supplied private scores with a one-sided test."""
    cosp_revised = np.asarray(cosp_scores, dtype=float)
    compare_scores = dict(compare_scores)
    if cosp_revised.ndim != 1 or not len(cosp_revised):
        raise ValueError('cosp_scores must be a nonempty one-dimensional array')
    if any(len(scores) != len(cosp_revised) for scores in compare_scores.values()):
        raise ValueError('All comparison scores must align with cosp_scores')

    # Convert None to np.nan and ensure all values are floats for np.isnan to work
    for method in compare_scores:
        compare_scores[method] = np.array([np.nan if x is None else x for x in compare_scores[method]], dtype=float)

    p_values_pp = {}
    for method, scores in compare_scores.items():
        # Filter out np.nan values to align the compared scores
        valid_indices = ~np.isnan(scores)
        filtered_cosp_score = cosp_revised[valid_indices]
        filtered_compare_score = scores[valid_indices]

        # Perform Wilcoxon test if arrays are not empty
        if filtered_cosp_score.size > 0 and filtered_compare_score.size > 0:
            stat, p = wilcoxon(filtered_cosp_score, filtered_compare_score, alternative="greater")
            p_values_pp[method] = p
        else:
            p_values_pp[method] = np.nan  # Indicates the test was not performed due to lack of data

    print(p_values_pp)
    return p_values_pp




def draw_coh_snapshots(pat, plot_type, mean_type, clustered, freq_band=None):
    freq_band = None if plot_type == 'coh-stack' else freq_band
    clustered = False if plot_type == 'coh-stack' else clustered
    assert(plot_type in ['heatmap', 'coh-stack'])

    print(f"\nPatient {pat}: drawing coherence snapshots: "
          f"plot_type={plot_type}  clustered={clustered}  mean_type={mean_type}  freq_band={freq_band}")
    # load coherence and predictions
    coh_ictal, epochs_ictal, ids_ictal, pred_ictal = load_coh_and_pred(pat, 'ictal')
    coh_tst, epochs_tst, ids_tst, pred_tst = load_coh_and_pred(pat, 'tst')

    # only plot seizure in test set
    for id in np.unique(ids_tst):
        # id = np.unique(ids_tst)[5]
        # get plot data: seizure
        epochs_ictal_id = np.sort(epochs_ictal[np.where(ids_ictal == id)])
        coh_in_sz = [coh_ictal[key].cpu().numpy() for key in epochs_ictal_id]
        if len(coh_in_sz) == 0: continue
        plot_ictal, prob_ictal_one = get_plot_data('ictal', coh_in_sz, epochs_ictal_id, mean_type, plot_type, pred_ictal,
                                                   freq_band=freq_band)
        plot_ictal, order, vmax, vmin  = cluster_heatmap(plot_ictal, plot_type, clustered)

        # get plot data: before seizure
        min_list = [60, 30, 10, 2]
        tst_plot = {}
        tst_prob = {}
        epochs_tst_id = np.sort(epochs_tst[np.where(ids_tst == id)])
        coh_b4_sz = [coh_tst[key].cpu().numpy() for key in epochs_tst_id]
        if len(coh_b4_sz) < (5 * max(min_list)): continue
        for idx, n_min in enumerate(min_list):
            plot_tst_one, prob_tst_one = get_plot_data('tst', coh_b4_sz, epochs_tst_id, mean_type, plot_type, pred_tst,
                                                       order=order, n_min=n_min, freq_band=freq_band)
            tst_plot[n_min] = plot_tst_one
            tst_prob[n_min] = prob_tst_one

        if plot_type == 'heatmap':
            plot_snapshot_heatmap(clustered, id, mean_type, order, pat, plot_ictal, plot_type, prob_ictal_one, tst_plot,
                                  tst_prob, vmax, vmin, freq_band)
        if plot_type == 'coh-stack':
            plot_snapshot_stack(clustered, coh_in_sz, id, mean_type, pat, prob_ictal_one, tst_plot, tst_prob)
    return


def plot_snapshot_stack(clustered, coh_in_sz, id, mean_type, pat, prob_ictal_one, tst_plot, tst_prob):
    fig = plt.figure(figsize=(30, 5))  # Adjust the figure size as needed
    gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 1], figure=fig)  # Allocate space for 5 plots
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    n_freq = 109
    n_freq_positions = [0, n_freq / 4, n_freq / 2, n_freq * 0.75, n_freq - 1]
    n_freq_labels = [0, 40, 80, 120, 170]
    x_ticks = np.arange(1, 121, 10)
    all_pair = []
    for i in range(1, 17):
        for j in range(i + 1, 17):
            all_pair.append(f"c{i}-c{j}")
    x_ticklabels = np.array(all_pair)[x_ticks - 1]
    for index, (key, value) in enumerate(tst_plot.items()):
        if index < len(axes):  # Ensure we don't go out of bounds
            im = axes[index].imshow(value, aspect='auto')  # Use value directly
            axes[index].invert_yaxis()  # Apply directly on the axes object
            axes[index].set_title(f'{key} minutes before seizure({tst_prob[key]:.2f})')

            # Set axis labels
            axes[index].set_ylabel('Frequency (Hz)')
            axes[index].set_xlabel('Channel Pairs')

            # Set x-ticks for each subplot
            # axes[index].set_xticks(x_ticks)
            # axes[index].set_xticklabels(x_ticklabels, rotation=45)
            axes[index].set_xticks([])
            axes[index].set_xticklabels([])

            # Set y-ticks for each subplot
            axes[index].set_yticks(n_freq_positions)
            axes[index].set_yticklabels(n_freq_labels)
            for pos in n_freq_positions:
                axes[index].axhline(y=pos, color='white', linestyle='--', linewidth=0.8)
        else:
            print(f"Warning: Skipping {key} due to more items than allocated subplots.")
    # Plot the seizure data in the last subplot if there's space
    if len(tst_plot) < 5:
        axes[-1].imshow(coh_in_sz[0], aspect='auto')
        axes[-1].invert_yaxis()
        axes[-1].set_title(f'Seizure({prob_ictal_one:.2f})')
        # Set axis labels
        axes[-1].set_ylabel('Frequency (Hz)')
        axes[-1].set_xlabel('Channel Pairs')
        # Set x-ticks for each subplot
        # axes[-1].set_xticks(x_ticks)
        # axes[-1].set_xticklabels(x_ticklabels, rotation=45)
        axes[-1].set_xticks([])
        axes[-1].set_xticklabels([])
        # Set y-ticks for each subplot
        axes[-1].set_yticks(n_freq_positions)
        axes[-1].set_yticklabels(n_freq_labels)
        # Add horizontal lines for each subplot
        for pos in n_freq_positions:
            axes[-1].axhline(y=pos, color='white', linestyle='--', linewidth=0.8)
    plt.subplots_adjust(wspace=0, hspace=0)  # Remove spaces between subplots
    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    f_fig = RST_PATH / (f"figures/{pat}/coh_snapshot_stack/"
                        f"{pat}_coh_snapshot_stack_mean-{mean_type}_clustered-{clustered}_sz-{id}.png")
    f_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f_fig)
    plt.close()
    # plt.show()


def plot_snapshot_heatmap(clustered, id, mean_type, order, pat, plot_ictal, plot_type, prob_ictal_one, tst_plot,
                          tst_prob, vmax, vmin, freq_band):
    # Setup for the figure and GridSpec
    fig = plt.figure(figsize=(25, 5))  # Adjust the figure size as needed
    gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.05],
                  figure=fig)  # Allocate space for 5 plots and 1 colorbar
    # Define axes for the heatmaps and the colorbar
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]  # 5 subplots for the coherence data
    cbar_ax = fig.add_subplot(gs[0, -1])  # Shared colorbar axis

    freqs = ['1-40', '41-80', '81-120', '121-170']
    freq_str = freqs[freq_band] if freq_band is not None else 'all'
    for index, (key, value) in enumerate(tst_plot.items()):
        plot_coh_heatmap(value, f'{key} minutes before seizure ({mean_type}, {freq_str},{tst_prob[key]:.2f})', axes[index],
                         vmax=vmax, vmin=vmin, order=order)
    plot_coh_heatmap(plot_ictal, f'Seizure ({mean_type},{freq_str},{prob_ictal_one:.2f})', axes[4], vmax=vmax, vmin=vmin,
                     order=order, cbar_ax=cbar_ax)
    plt.subplots_adjust(wspace=0, hspace=0)  # Remove spaces between subplots
    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    f_fig = RST_PATH / (f"figures/{pat}/coh_snapshot_heatmap/"
                        f"{pat}_coh_snapshot_{plot_type}_mean-{mean_type}_clustered-{clustered}_sz-{id}_freq-{freq_str}.png")
    f_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f_fig)
    plt.close()
    # plt.show()


def cluster_heatmap(plot_data, plot_type, clustered):
    if plot_type == 'heatmap' and clustered:
        # Z = linkage(avg_coh_ictal, method='average')
        distance_matrix = 1 - plot_data  # Assuming similarity scores are normalized
        np.fill_diagonal(distance_matrix, 1)  # to exclude the diagonal
        # condensed_distance_matrix = squareform(distance_matrix, checks=False)
        # Z = linkage(condensed_distance_matrix, method='average')
        Z = linkage(distance_matrix, method='average')
        order = leaves_list(Z)
    else:
        order = range(16)

    plot_data = plot_data[order, :][:, order]
    vmax, vmin = np.max(plot_data), np.min(plot_data)
    return plot_data, order, vmax, vmin


def get_plot_data(data_type, coh, epochs, mean_type, plot_type, pred, order=None, n_min=None, freq_band=None):
    assert(data_type in ['tst', 'ictal'])

    if mean_type == '10-second':
        start = len(coh) - 5 * n_min if data_type=='tst' else 0
        coh_one = coh[start]
        prob_one = pred.loc[epochs[start], 'probability']
    elif 'minute' in mean_type:
        mean_min = int(mean_type.split('-')[0])
        start = len(coh) - 5 * n_min if data_type == 'tst' else 0
        if data_type == 'tst':
            end = len(coh) - 5 * (n_min - mean_min - 1) if n_min > (mean_min - 1) else len(coh)
        else:
            end = 5 * mean_min if len(coh) > 5 * mean_min else len(coh)
        coh_one = np.mean(np.stack(coh[start:end]), axis=0)
        eps = [epochs[i] for i in range(start, end)]
        prob_one = np.mean([pred.loc[ep, 'probability'] for ep in eps])
    else:
        raise ValueError(f"ERROR: unknown mean_type={mean_type}.")

    if plot_type == 'heatmap':
        coh_3d = convert_coh_to_3d(coh_one)
        if freq_band is not None:
            assert (freq_band < 4)  # most 4 bands
            freq_splits = np.array_split(np.arange(coh_3d.shape[2]), 4)
            coh_one = np.mean(coh_3d[:,:, freq_splits[freq_band]], axis=2)
        else:
            coh_one = np.mean(coh_3d, axis = 2)
        if order is not None:
            coh_one = coh_one[order][:, order]
        return coh_one, prob_one


def load_coh_and_pred(pat, data_type):
    coh_dic = load_coherence_dict(pat, data_type)
    epochs = np.array(list(coh_dic.keys()))
    ids, _ = get_sz_info_from_epochs(pat, epochs, data_type)
    # load prediction
    mid = FINAL_INFO.loc[FINAL_INFO['pat'] == pat, 'P10s-4m'].iloc[0]
    if data_type == 'ictal':
        pred = pd.read_csv(RST_PATH / f"detail/model{mid}_round1_pred_ictal.csv")
    elif data_type == 'tst':
        pred = pd.read_csv(RST_PATH / f"detail/model{mid}_round1_pred_180_tst.csv")
    else:
        raise ValueError(f"ERROR: unknown data_type={data_type}.")
    pred['probability'] = sigmoid(pred['prediction'])
    pred.set_index('seg_epoch', inplace=True)
    return coh_dic, epochs, ids, pred


def generate_patient_info_table():
    pat_info_list = []  # Use a list to collect dictionaries for each patient
    for idx, pat in enumerate(PAT_LIST_ALL):
        anno = pd.read_csv(INFO_PATH / f"annotation/{pat}_Annots.csv")
        n_sz_12 = np.sum((anno['sz_type'] == 1) | (anno['sz_type'] == 2))
        n_sz_lead12 = np.sum(anno['lead_12'])
        used = pat in PAT_LIST
        n_days = PAT_ALL_DAYS[idx]

        if used:
            sz_used = np.loadtxt(WORK_PATH / f"data/{pat}/{pat}_sz_id_used.csv", delimiter=',').astype(int)
            n_sz_trn = int(len(sz_used) * CONFIG['trn_sz_pct'])
            n_sz_tst = len(sz_used) - int(len(sz_used) * (CONFIG['trn_sz_pct'] + CONFIG['val_sz_pct']))
            n_sz_val = len(sz_used) - n_sz_trn - n_sz_tst
        else:
            n_sz_trn, n_sz_val, n_sz_tst = '-', '-', '-'

        # Append a new dictionary for each patient
        pat_info_list.append({
            'Patient Index': idx + 1,
            'Patient': pat,
            'Total Seizures': n_sz_12,
            'Days': n_days,
            'Seizure per Day': round(n_sz_12 / n_days, 2),
            'Lead Seizures': n_sz_lead12,
            'Included': used,
            'Seizures in Training': n_sz_trn,
            'Seizures in Validation': n_sz_val,
            'Seizures in Test': n_sz_tst
        })
    # Convert the list of dictionaries into a DataFrame
    pat_info = pd.DataFrame(pat_info_list)
    pat_info.columns = ['Patient Index', 'Patient', 'Total Seizures', 'Days', 'Seizures per Day', 'Lead Seizures',
                        'Included', 'Seizures in Training', 'Seizures in Validation', 'Seizures in Test']
    print(pat_info)
    pat_info.to_csv(RST_PATH / "patient_info.csv", index=False)
    return pat_info


def draw_raw_eeg(eeg_data, save_name=None):
    num_channels = eeg_data.shape[0]
    sample_rate = 400  # Hz
    time = np.linspace(0, eeg_data.shape[1] / sample_rate, eeg_data.shape[1])
    channel_offsets = np.arange(num_channels) * eeg_data.std() * 4  # Increased gap
    plt.figure(figsize=(18, 10))
    for i in range(num_channels):
        plt.plot(time, eeg_data[i, :] + channel_offsets[i], color='#6093AC')
        plt.text(-0.1, channel_offsets[i], f'ch{num_channels - i}', verticalalignment='center',
                 horizontalalignment='right', color='black', fontsize=12)
    plt.axis('off')
    if save_name is None:
        plt.show()
        print("Plot has been shown.")
    else:
        plt.savefig(save_name)
        print(f"{save_name.name} has been saved to {save_name.parent}")
        plt.close()


def draw_eeg_snapshot(pat, save=True):
    id_used = np.sort(np.loadtxt(WORK_PATH / f"data/{pat}/{pat}_sz_id_used.csv", delimiter=',').astype(int))
    tst_ids = id_used[int(len(id_used) * 0.8):]
    ictal_list = pd.read_csv(INFO_PATH / f"segment_list/{pat}/{pat}_segment_list_ictal_10_4000.csv")
    szb4_list = pd.read_csv(
        INFO_PATH / f"segment_list/{pat}/{pat}_segment_list_10_185_5_1_non-overlap_non-overlap_10.csv")
    for id in tst_ids:
        if id in ictal_list['sz_id'].to_numpy():
            seg_one = ictal_list[ictal_list['sz_id'] == id].reset_index(drop=True)
            seg_one = seg_one.sort_values(by='seg_epoch').reset_index(drop=True)
            file_name, seg_epoch = seg_one.iloc[0]['file_name'], seg_one.iloc[0]['seg_epoch']
            start, end = seg_one.iloc[0]['seg_start_step'], seg_one.iloc[0]['seg_end_step']
            rec_np, _ = load_mat_to_np(file_name)
            eeg = rec_np[start:end, :].T
            if save:
                f_name = RST_PATH / f"figures/{pat}/eeg_raw/{pat}_eeg_snapshot_10s_sz-{id}_ictal-1_{seg_epoch}.png"
                f_name.parent.mkdir(parents=True, exist_ok=True)
                draw_raw_eeg(eeg, f_name)
            else:
                draw_raw_eeg(eeg, save_name=None)

        if id in szb4_list['sz_id'].to_numpy():
            seg_one = szb4_list[szb4_list['sz_id'] == id].reset_index(drop=True)
            seg_one = seg_one.sort_values(by='seg_epoch').reset_index(drop=True)
            for n_min in [2, 10, 30, 60]:
                idx = len(seg_one) - n_min * 5  # 5 segments per minutes
                if idx > 0:
                    file_name, seg_epoch = seg_one.iloc[idx]['file_name'], seg_one.iloc[idx]['seg_epoch']
                    start, end = seg_one.iloc[idx]['seg_start_step'], seg_one.iloc[idx]['seg_end_step']
                    rec_np, _ = load_mat_to_np(file_name)
                    eeg = rec_np[start:end, :].T
                    if save:
                        f_name = RST_PATH / f"figures/{pat}/eeg_raw/{pat}_eeg_snapshot_10s_sz-{id}_tst-{n_min}m_{seg_epoch}.png"
                        f_name.parent.mkdir(parents=True, exist_ok=True)
                        draw_raw_eeg(eeg, f_name)
                    else:
                        draw_raw_eeg(eeg, save_name=None)



def draw_coh_average_plots(pat, n_min, with_ictal=True, split_freq=True, all_sz=False):
    print(f"\nPatient {pat}: drawing average coherence plots with {n_min} minutes before seizure onset.")
    freqs = ['1-40', '41-80', '81-120', '121-170']

    # freq_col = ['blue', 'orange', 'green', 'purple']
    freq_splits = np.array_split(np.arange(109), 4)

    id_used = np.sort(np.loadtxt(WORK_PATH / f"data/{pat}/{pat}_sz_id_used.csv", delimiter=',').astype(int))
    tst_used = id_used[int(len(id_used) * 0.8):]

    # load coherence of test set
    data_type = 'tst'
    coh_tst = load_coherence_dict(pat, data_type)
    epochs_tst = np.array(list(coh_tst.keys())).astype(int)
    ids_tst, types_tst = get_sz_info_from_epochs(pat, epochs_tst, data_type)

    # load coherence of ictal
    data_type = 'ictal'
    coh_ictal = load_coherence_dict(pat, data_type)
    epochs_ictal = np.array(list(coh_ictal.keys()))
    ids_ictal, _ = get_sz_info_from_epochs(pat, epochs_ictal, data_type)

    plot_all_sz = {}
    for id in tst_used:
        coh_plot_dict = {}
        sz_type = types_tst[np.where(ids_tst==id)[0]][0]

        # get the coherence before a seizure onset, item shape: [109, 120] (n_freq, n_channel_pair)
        epochs_tst_one = np.sort(epochs_tst[np.where(ids_tst == id)])[-(5 * n_min):]
        coh_b4_sz = [coh_tst[key] for key in epochs_tst_one]
        coh_b4_avg = [coh.mean() for coh in coh_b4_sz]
        if len(coh_b4_avg) < 5 * n_min:
            # padding to same length
            coh_b4_avg = (5 * n_min - len(coh_b4_avg)) * [np.nan] + coh_b4_avg
        if with_ictal:
            # get the coherence in seizure
            epochs_ictal_one = np.sort(epochs_ictal[np.where(ids_ictal == id)])
            coh_in_sz = [coh_ictal[key] for key in epochs_ictal_one]
            coh_sz_avg = [coh.mean() for coh in coh_in_sz]
            coh_plt = np.array(coh_b4_avg + coh_sz_avg)
        else:
            coh_plt = np.array(coh_b4_avg)

        if not split_freq:
            coh_plot_dict['all'] = coh_plt
        else:
            for idx, freq in enumerate(freq_splits):
                coh_b4_freq = [np.mean(convert_coh_to_3d(coh_2d)[:, :, freq]) for coh_2d in coh_b4_sz]
                if with_ictal:
                    coh_sz_freq = [np.mean(convert_coh_to_3d(coh_2d)[:, :, freq]) for coh_2d in coh_in_sz]
                    coh_plt = np.array(coh_b4_freq + coh_sz_freq)
                else:
                    coh_plt = np.array(coh_b4_freq)
                coh_plot_dict[freqs[idx]] = coh_plt

        if all_sz:
            plot_all_sz[id] = coh_plot_dict
        else:
            plot_coh_average_one(pat, id, coh_plot_dict, n_min, split_freq, sz_type, with_ictal)

    if all_sz:
        coh_plot_dict = {}
        for key in next(iter(plot_all_sz.values())).keys():
            lists = [item[key] for item in plot_all_sz.values()]
            max_length = max(len(l) for l in lists)
            # Pad lists with NaN and create DataFrame
            padded_lists = [list(l) + [np.nan] * (max_length - len(l)) for l in lists]
            df = pd.DataFrame(padded_lists, index=plot_all_sz.keys())
            if with_ictal and (n_min == 120):
                df.to_csv(RST_PATH / f"figures/{pat}/coh_average/{pat}_coh_average_freq-{key}.csv")
            coh_plot_dict[key] = df.mean(axis=0, skipna=True).to_numpy()
        plot_coh_average_one(pat, 'all', coh_plot_dict, n_min, split_freq, 'NA', with_ictal)


def plot_coh_average_one(pat, id, coh_plot_dict, n_min, split_freq, sz_type, with_ictal):
    plt.figure(figsize=(15, 4))  # You can adjust the figure size as needed
    line_styles = [':', '-.', '-', '--'] if split_freq else ['-']
    for i, (key, value) in enumerate(coh_plot_dict.items()):
        plt.plot(value, label=key, linestyle=line_styles[i % len(line_styles)])  # , linestyle=line_styles[i % len(line_styles)]
    # Plot a red vertical line to show seizure onset
    sz_onset = n_min * 5
    plt.axvline(x=sz_onset - 1, color='r', linestyle='-', linewidth=0.6)
    plt.plot(sz_onset - 1, plt.gca().get_ylim()[1], marker='v', color='r', markersize=10)
    ticks_every_10_min = np.linspace(0, sz_onset - 1, n_min // 10 + 1,
                                     dtype=int)  # Calculate positions for every 10 minutes
    x_tick_text = [f'-{n_min - i * 10}m' for i in range(len(ticks_every_10_min))]
    x_tick_text[-1] = 'onset'
    plt.xticks(ticks_every_10_min, x_tick_text)
    plt.title(f"{pat} Average coherence of {n_min} minutes before Seizure {id} (type: {sz_type})")
    plt.ylabel("Average Correlation")
    plt.xlabel("Minutes to Seizure Onset")
    plt.grid(axis='y')
    if split_freq: plt.legend(loc='upper left')
    f_folder = RST_PATH / f"figures/{pat}/coh_average/"
    if id != 'all': f_folder = f_folder / "seizures"
    if split_freq:
        f_fig = f_folder / f"{pat}_coh_average_minutes-{n_min}_sz-{id}_type-{sz_type}_freq-split_wictal-{with_ictal}.png"
    else:
        f_fig = f_folder / f"{pat}_coh_average_minutes-{n_min}_sz-{id}_type-{sz_type}_freq-all_wictal-{with_ictal}.png"
    f_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f_fig, bbox_inches='tight')
    plt.close()
    print(f"Plots {f_fig.name} have been saved to {f_fig.parent}.")
    # plt.show()


def get_summary_one(coh_freq, mode, start, end):
    if mode == 'mean':
        sum_one = np.mean(coh_freq.iloc[:, start:end].mean(axis=1, skipna=True))
    elif mode == 'var':
        sum_one = np.mean(coh_freq.iloc[:, start:end].var(axis=1, skipna=True))
    elif mode == 'ratio':
        base = np.mean(coh_freq.iloc[:, 0:(120 * 5)].mean(axis=1, skipna=True))
        sum_one = np.mean(coh_freq.iloc[:, start:end].mean(axis=1, skipna=True))
        sum_one = sum_one / base
    else:
        raise ValueError(f"ERROR: unknown mode={mode}")
    return sum_one


def gen_coh_summary_one(pat, mode, itvl_min):
    print(f"generating coherence average information: pat={pat}, mode={mode}")
    freq_str = ['all', '1-40', '41-80', '81-120', '121-170']
    coh_summary = {}
    for freq in freq_str:
        f_name = RST_PATH / f"figures/{pat}/coh_average/{pat}_coh_average_freq-{freq}.csv"
        coh_freq = pd.read_csv(f_name, index_col=0)
        sum_one = [pat, mode]

        for idx, itvl in enumerate(itvl_min):
            start_point = (120 - sum(itvl_min)) * 5
            start = start_point if idx == 0 else start_point + sum(itvl_min[:idx]) * 5
            end = start + itvl * 5 if (start + itvl * 5) < (120 * 5) else 120 * 5
            sum_one_idx = get_summary_one(coh_freq, mode, start, end)
            sum_one.append(sum_one_idx)

        # -120 ~ -1
        sum_one_idx = get_summary_one(coh_freq, mode, 0, 120 * 5) if mode != 'ratio' else 1
        sum_one.append(sum_one_idx)
        # ictal
        sum_one_idx = get_summary_one(coh_freq, mode, 120 * 5, coh_freq.shape[1])
        sum_one.append(sum_one_idx)
        coh_summary[freq] = sum_one

    coh_summary = pd.DataFrame(coh_summary).T
    col_one = [-sum(itvl_min[i:]) for i in range(len(itvl_min))] + [-1]
    col_names = (['pat', 'mode'] + [f"{col_one[i]}~{col_one[i + 1]}m" for i in range(len(col_one) - 1)] +
                 [f'-120~-1m', f'ictal'])
    coh_summary.columns = col_names
    # print(coh_summary.round(4))
    return coh_summary


def generate_coh_avg_summary():
    itvl_min = [60, 30, 20, 5, 5]
    coh_sum = []
    for pat in PAT_LIST:
        for mode in ['mean', 'ratio', 'var' ]:
            sum_one = gen_coh_summary_one(pat, mode, itvl_min)
            coh_sum.append(sum_one)
    coh_sum = pd.concat(coh_sum, axis=0)
    coh_sum.reset_index(inplace=True)
    coh_sum.rename(columns={'index': 'frequency'}, inplace=True)
    columns_order = ['pat'] + [col for col in coh_sum.columns if col != 'pat']
    coh_sum = coh_sum[columns_order]

    print(coh_sum.round(4))
    f_save = RST_PATH / f"coherence_average_summary.csv"
    coh_sum.to_csv(f_save, index=False)
    print(f"{f_save.name} has been save to {f_save.parent}")


def plot_predict_prob(pat):
    # prediction from Predictor-10s
    mid_10s = FINAL_INFO.loc[FINAL_INFO['pat'] == pat, 'P10s-4m'].iloc[0]
    pred_10s = pd.read_csv(RST_PATH / f"detail/model{mid_10s}_round1_pred_180_tst.csv", index_col=False).sort_values(
        by='seg_epoch')
    ids_10s, _ = get_sz_info_from_epochs(pat, pred_10s['seg_epoch'], 'tst')
    pred_10s['sz_id'] = ids_10s
    pred_10s['probility'] = sigmoid(pred_10s['prediction'].to_numpy())
    # prediction from Predictor-1m
    mid_1m = FINAL_INFO.loc[FINAL_INFO['pat'] == pat, 'P1m_4m'].iloc[0]
    pred_1m = pd.read_csv(RST_PATH / f"detail/model{mid_1m}_round1_pred_1440_tst.csv")
    seg_list = pd.read_csv(
        INFO_PATH / f"segment_list/{pat}/{pat}_segment_list_10_1445_5_1_non-overlap_non-overlap_10.csv")
    seg_list = seg_list[seg_list['sz_id'].isin(np.unique(ids_10s))].sort_values(by='seg_epoch')
    rec_epoch = seg_list['rec_start_epoch'].unique()
    ids_1m, _ = get_sz_info_from_epochs(pat, rec_epoch, 'tst')
    pred_1m['rec_epoch'] = rec_epoch[:len(pred_1m)]  ##########!
    pred_1m['sz_id'] = ids_1m[:len(pred_1m)]  ##########!
    pred_1m['probability'] = sigmoid(pred_1m['prediction'].to_numpy())
    # threshold
    thd_pd = pd.read_csv(RST_PATH / f"detail/model{mid_1m}_round1_performance_product_1440_tst.csv")
    thd = thd_pd[thd_pd['perf_prod'] == thd_pd['perf_prod'].max()]['threshold'].iloc[0]
    for id in np.unique(ids_10s):
        pred_10s_one = pred_10s[pred_10s['sz_id'] == id]
        prob_raw = pred_10s_one['probility'].to_numpy()
        prob_smooth = sigmoid(pred_10s_one['prediction'].rolling(window=60 * 5, min_periods=1).mean().to_numpy())

        pred_1m_one = pred_1m[pred_1m['sz_id'] == id]
        prob_1m = pred_1m_one['probability'].to_numpy()
        prob_1m_10s = [i for i in prob_1m for _ in range(5)]

        n_max = 2 * 60 * 5 if len(prob_raw) > 600 else len(prob_raw)
        prob_raw = prob_raw[(-n_max):]
        prob_smooth = prob_smooth[(-n_max):]
        offset_1m = 1  # 5 * 10
        prob_1m_10s = prob_1m_10s[(-n_max - offset_1m):-offset_1m]

        fig = plt.figure(figsize=(15, 4))
        plt.plot(prob_raw, label='Predictor-10s', color='darkgray', linestyle='--', linewidth=1.5)
        # plt.plot(prob_smooth, label='Smoothed Predictor-10s', color='green', linestyle=':', linewidth=1.5)
        plt.plot(prob_1m_10s, label='Predictor-1m', color='orange', linestyle='-', linewidth=1.5)

        plt.axhline(y=thd, color='black', linestyle='-', linewidth=0.8, label='threshold')

        sz_onset = n_max + 5
        plt.axvline(x=sz_onset, color='r', linestyle='-', linewidth=0.6)
        # plt.axvline(x=sz_onset-5, color='blue', linestyle='-', linewidth=0.6)
        plt.plot(sz_onset, 1, marker='v', color='r', markersize=10)
        x_time = int(n_max / 5) - 1
        ticks_every_10_min = np.linspace(0, sz_onset, x_time // 10,
                                         dtype=int)  # Calculate positions for every 10 minutes
        x_tick_text = [f'-{x_time - i * 10}m' for i in range(len(ticks_every_10_min))]
        x_tick_text[-1] = 'seizure onset'
        plt.xticks(ticks_every_10_min, x_tick_text)
        # plt.title( f"{pat} predictions of {x_time} minutes before Seizure {id}")
        plt.ylabel("Probability")
        plt.xlabel("Minutes to Seizure Onset")

        plt.legend(loc='upper left')
        # plt.grid(axis='y')
        plt.show()

def plot_results_histogram(data, single_fig=True):
        """Plot caller-supplied metrics; no participant results are bundled.

        data uses Patient and <metric>_<interval> columns, with metrics
        Sen/TiW/PP and intervals 4m/15m/30m/45m.
        """
        df = pd.DataFrame(data)

        # Plot settings
        # metrics = ['AUC', 'Sen', 'TiW', 'PP']
        metrics = ['Sen', 'TiW', 'PP']
        preictal_intervals = ['4m', '15m', '30m', '45m']
        n_patients = df.shape[0]
        # colormap = cm.get_cmap('viridis', 3)  # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        # colors = [colormap(i) for i in range(4)]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        # colors = ['#ff7f0e', '#2ca02c', '#d62728']

        if single_fig:
            # Creating the subplot grid: n_patients rows and len(metrics) columns
            n_rows = max(1, math.ceil(n_patients / 2))
            n_columns = 7  # 3 metrics per patient + 1 column as a gap, for 2 patients
            fig, axs = plt.subplots(n_rows, n_columns, figsize=(20, 2 * n_rows), sharey=False, squeeze=False)
            for i in range(n_patients):
                patient_data = df.iloc[i]
                patient_id = patient_data['Patient']

                # Determine the row for the current patient
                row_idx = i // 2
                # Determine the starting column for the current patient (0 for the first patient, 5 for the second to add a gap)
                col_start = (3 + 1) * (i % 2)
                for j, metric in enumerate(metrics):
                    col_idx = col_start + j
                    ax = axs[row_idx, col_idx]
                    values = [patient_data[f'{metric}_{interval}'] for interval in preictal_intervals]
                    ax.bar(preictal_intervals, values, color=colors, width=0.95, edgecolor='none')
                    ax.set_title(f'{metric} (Patient {int(patient_id)})', fontsize=14)
                    ax.set_ylim(0, 1)
                    # Simplify the plot by removing unnecessary spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    # Make adjustments for the gap column (5th column in each row)
                    if (i % 2 == 0) and (j == 2):  # After plotting the last metric for the first patient
                        gap_col = col_start + 3
                        axs[row_idx, gap_col].axis('off')  # Turn off the gap column
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(RST_PATH / f"figures/result_histogram_3.png")
            plt.show()
        else:
            # Creating a plot for each patient
            for i in range(n_patients):
                patient_data = df.iloc[i]
                patient_id = int(patient_data['Patient'])

                fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
                fig.suptitle(f'Performance of Patient {patient_id}', fontsize=20)

                # Adjust the width for no gaps
                width = 0.95  # Width of bars

                for j, metric in enumerate(metrics):
                    values = [patient_data[f'{metric}_{interval}'] for interval in preictal_intervals]
                    axs[j].bar(preictal_intervals, values, color=colors, width=width, edgecolor='none')
                    axs[j].set_title(metric, fontsize=18)
                    axs[j].set_ylim(0, 1)  # Assuming the metrics values are between 0 and 1
                    # Remove the top and right spines (borders)
                    # axs[j].spines['right'].set_visible(False)
                    axs[j].spines['top'].set_visible(False)
                    # axs[j].spines['left'].set_visible(False)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                # plt.show()
                plt.savefig(RST_PATH / f"figures/result_histogram_patient-{patient_id}.png")


def probability_to_logit(p):
    if p <= 0 or p >= 1:
        raise ValueError("Probability must be between 0 and 1, exclusive.")
    logit = np.log(p / (1 - p))
    return np.array(logit)

def plot_result_comparison():
    pat_list = FINAL_INFO['pat_id']
    pat_names = FINAL_INFO['pat']
    thd45 = FINAL_INFO['thd45_opt']
    thd45_logit = [probability_to_logit(p) for p in thd45]
    model_type = 'model45'

    for idx, mid in enumerate(FINAL_INFO[model_type]):
        pat = pat_list[idx]
        logit = thd45_logit[idx]
        pred_trn = pd.read_csv(RST_PATH / f"detail/model{mid}_round1_pred_trn_sampled.csv")
        pre_trn = pred_trn.loc[pred_trn['label']==1, 'prediction']
        inter_trn = pred_trn.loc[pred_trn['label']==0, 'prediction']

        pred_tst = pd.read_csv(RST_PATH / f"detail/model{mid}_round1_pred_1440_tst.csv")
        pre_tst = pred_tst.loc[pred_tst['label'] == 1, 'prediction']
        inter_tst = pred_tst.loc[pred_tst['label'] == 0, 'prediction']

        # Setting up the plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # Comparing preictal and interictal for the training set
        axs[0, 0].hist(pre_trn, bins=50, color='red', alpha=0.5, label='Preictal', density=True)
        axs[0, 0].hist(inter_trn, bins=50, color='green', alpha=0.4, label='Interictal', density=True)
        axs[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='threshold 0.5', alpha=0.5)
        # axs[0, 0].axvline(x=logit, color='purple', linestyle='-', linewidth=1.5)
        axs[0, 0].set_title('Predictions Distribution (Training Set)')
        axs[0, 0].set_xlabel('Prediction Value')
        axs[0, 0].set_ylabel('Distribution Density')
        axs[0, 0].legend()

        # Comparing preictal and interictal for the test set
        axs[0, 1].hist(pre_tst, bins=50, color='red', alpha=0.5, label='Preictal', density=True)
        axs[0, 1].hist(inter_tst, bins=50, color='green', alpha=0.4, label='Interictal', density=True)
        axs[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='threshold 0.5', alpha=0.5)
        axs[0, 1].axvline(x=logit, color='purple', linestyle='-', linewidth=1.5, label='optimal threshold')
        axs[0, 1].set_title('Predictions Distribution (Test Set)')
        axs[0, 1].set_xlabel('Prediction Value')
        axs[0, 1].set_ylabel('Distribution Density')
        axs[0, 1].legend()

        # Distribution of interictal in test vs training sets
        axs[1, 0].hist(inter_tst, bins=50, color='purple', alpha=0.5, label='Test', density=True)
        axs[1, 0].hist(inter_trn, bins=50, color='blue', alpha=0.4, label='Train', density=True)
        axs[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='threshold 0.5', alpha=0.5)
        axs[1, 0].set_title('Predictions Distribution (Interictal)')
        axs[1, 0].set_xlabel('Prediction Value')
        axs[1, 0].set_ylabel('Distribution Density')
        axs[1, 0].legend()

        # Distribution of preictal in test vs training sets
        axs[1, 1].hist(pre_tst, bins=50, color='purple', alpha=0.5, label='Test', density=True)
        axs[1, 1].hist(pre_trn, bins=50, color='blue', alpha=0.4, label='Train', density=True)
        axs[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='threshold 0.5', alpha=0.5)
        axs[1, 1].set_title('Predictions Distribution (Preictal)')
        axs[1, 1].set_xlabel('Prediction Value')
        axs[1, 1].set_ylabel('Distribution Density')
        axs[1, 1].legend()

        fig.suptitle(f'Comparative Analysis of Predictions for Patient {pat}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
        # plt.show()
        plt.savefig(RST_PATH / f"figures/{pat_names[idx]}/{pat_names[idx]}_result_comparison_{model_type}.png")
        print(pat, ': done.')


def load_predict_logit(pat_id, pre_itvl=45, data_type='tst', pred_len=1440):
    # obtain the prediction of a patient
    final_info = FINAL_INFO.copy().set_index('pat_id')
    assert pat_id in final_info.index, f"ERROR: unknown patient id {pat_id}"
    assert pre_itvl in PRE_ITVL_TESTED, f"ERROR: unknown preictal interval={pre_itvl}"
    assert data_type in ['trn', 'val','tst'], f"ERROR: unknown data_type={data_type}"
    pred_len = 1440 if data_type in ['trn', 'val'] else pred_len

    print(f"Loading predictions: pat_id={pat_id}, pre_itvl={pre_itvl}, "
          f"data_type={data_type}, pred_len={pred_len}")

    mid = final_info.loc[pat_id, f"model{pre_itvl}"]
    f_name = f"detail/model{mid}_round1_pred_{pred_len}_{data_type}.csv"
    print(f"Prediction file name is {f_name}")
    prediction = pd.read_csv(RST_PATH / f_name )

    return prediction


import matplotlib.lines as mlines

def draw_pp_pre_itvl(thd_from='tst', save=False):
    assert thd_from in THD_APPROACHES, f"ERROR: unknown thresholding approach={thd_from}, options={THD_APPROACHES}"
    data = pd.read_csv(RST_PATH / "result_summary.csv")
    # Apply the filter conditions
    data = data[data['thd_from'] == thd_from]
    data = data[data['interictal_itvl'] == 1440]
    data = data[data['model_id'] != 'random']
    data = data.sort_values(by='preictal_itvl')

    # Create a figure with 10 subplots, one for each patient
    # fig, axs = plt.subplots(10, 1, figsize=(8, 10), sharex=True)
    fig, axs = plt.subplots(5, 2, figsize=(10, 10), sharey=True, sharex=True) #, sharey=True, sharex=True

    for idx, patient in enumerate(PAT_ID_LIST[:10]):
        # ax = axs[idx]
        ax = axs[idx // 2, idx % 2]
        patient_data = data[data['pat_idx'] == patient]
        ax.plot(patient_data['preictal_itvl'], patient_data['pp'], marker='o', label=f'Patient {patient}',linewidth=2)

        ax.set_ylim(0.6, 1)
        ax.grid(axis='both', alpha=0.3)

        if idx % 2 == 0:
            ax.set_ylabel('PP', fontsize=14)
        ax.set_xticks(patient_data['preictal_itvl'])
        ax.set_xticklabels(patient_data['preictal_itvl'], fontsize=12)
        # Set y-axis tick labels to size 12
        ax.tick_params(axis='y', labelsize=12)


        # Create a proxy artist for the legend
        proxy = mlines.Line2D([], [], linestyle='none', label=f'Patient {patient}')
        legend = ax.legend(handles=[proxy], loc='lower center', fontsize=16)
        legend.get_frame().set_linewidth(0)  # Remove legend border

        # Add a horizontal line at the value of 0.9
        ax.axhline(y=0.9, color=mycol['orange'], linestyle='--', linewidth=1.5)

        # Remove y-axis ticks for the second column
        if idx % 2 == 1:
            ax.yaxis.set_tick_params(length=0)

        # Remove x-axis ticks for all but the bottom row
        if idx // 2 != 4:
            ax.xaxis.set_tick_params(length=0)

    axs[-1, -1].set_xlabel('Preictal Interval', fontsize=14)
    axs[4, 0].set_xlabel('Preictal Interval', fontsize=14)
    plt.subplots_adjust(hspace=0.1, wspace=0)  # Adjust space between subplots
    plt.tight_layout()
    if save:
        f_save = RST_PATH / "figures/pp_preictal_itvls.png"
        plt.savefig(f_save)
        print(f"figure has been saved to {f_save}")
    else:
        plt.show()


def draw_pp_pre_itvl_revised(save=False):
    # for the revised paper from 22 Jan 2025 afterwards
    data = pd.read_csv(RST_PATH / "rst_warning.csv")
    # Apply the filter conditions
    data = data[data['method'] == 'ma_pi_prob']
    data = data[data['retrigger'] == True]
    data = data[data['thd_type'] != 'val']
    data = data.sort_values(by='pre_itval')

    # Create a figure with 10 subplots, one for each patient
    # fig, axs = plt.subplots(10, 1, figsize=(8, 10), sharex=True)
    fig, axs = plt.subplots(5, 2, figsize=(10, 10), sharey=True, sharex=True)  # , sharey=True, sharex=True

    for idx, patient in enumerate(PAT_ID_LIST[:10]):
        # ax = axs[idx]
        ax = axs[idx // 2, idx % 2]
        patient_data = data[data['pid'] == patient]
        ax.plot(patient_data['pre_itval'], patient_data['Seizure_PP'], marker='o', label=f'Patient {patient}', linewidth=2)

        ax.set_ylim(0, 1)
        ax.grid(axis='both', alpha=0.3)

        if idx % 2 == 0:
            ax.set_ylabel('PP', fontsize=14)
        ax.set_xticks(patient_data['pre_itval'])
        ax.set_xticklabels(patient_data['pre_itval'], fontsize=12)
        # Set y-axis tick labels to size 12
        ax.tick_params(axis='y', labelsize=12)

        # Create a proxy artist for the legend
        proxy = mlines.Line2D([], [], linestyle='none', label=f'Patient {patient}')
        legend = ax.legend(handles=[proxy], loc='upper center', fontsize=16)
        legend.get_frame().set_linewidth(0)  # Remove legend border

        # # Add a horizontal line at the value of 0.9
        # ax.axhline(y=0.9, color=mycol['orange'], linestyle='--', linewidth=1.5)

        # Remove y-axis ticks for the second column
        if idx % 2 == 1:
            ax.yaxis.set_tick_params(length=0)

        # Remove x-axis ticks for all but the bottom row
        if idx // 2 != 4:
            ax.xaxis.set_tick_params(length=0)

    axs[-1, -1].set_xlabel('Preictal Interval', fontsize=14)
    axs[4, 0].set_xlabel('Preictal Interval', fontsize=14)
    plt.subplots_adjust(hspace=0.1, wspace=0)  # Adjust space between subplots
    plt.tight_layout()
    if save:
        f_save = RST_PATH / "figures/pp_preictal_itvls.png"
        plt.savefig(f_save)
        print(f"figure has been saved to {f_save}")
    else:
        plt.show()



def plot_curves_up4(data1, label1, ycol1='pp',
                    data2=None, label2=None, ycol2='pp',
                    data3=None, label3=None, ycol3='pp',
                    data4=None, label4=None, ycol4='pp',
                    ylable='PP'):
    # Create a figure with 10 subplots (4 rows and 3 columns)
    fig, axs = plt.subplots(4, 3, figsize=(15, 12))

    # Define font size for axis labels and titles
    axis_label_font_size = 12
    title_font_size = 14

    # Iterate through each patient and plot curves of PP over different preictal intervals
    for idx, patient in enumerate(PAT_ID_LIST):
        pat_data1 = data1[data1['pat_idx'] == patient]
        pat_data2 = data2[data2['pat_idx'] == patient] if data2 is not None else None
        pat_data3 = data3[data3['pat_idx'] == patient] if data3 is not None else None
        pat_data4 = data4[data4['pat_idx'] == patient] if data4 is not None else None
        row = (idx + 2) // 3
        col = (idx + 2) % 3

        if idx == 0:
            # For the first patient, make the subplot occupy the entire first row
            ax = fig.add_subplot(4, 1, 1)
        else:
            ax = axs[row, col]

        if not pat_data1.empty:
            ax.plot(pat_data1['preictal_itvl'], pat_data1[ycol1], marker='o', label=label1, linestyle='-', color=mycol['orange'])
        if data2 is not None and not pat_data2.empty:
            color = mycol['green'] if data3 is None and data4 is None else mycol['blue']
            lt = '-.' if data3 is None and data4 is None else '--'
            ax.plot(pat_data2['preictal_itvl'], pat_data2[ycol2], marker='x', label=label2, linestyle=lt, color=color)
        if data3 is not None and not pat_data3.empty:
            color = mycol['green'] if data4 is None else mycol['purple']
            lt = '-.' if data4 is None else ':'
            ax.plot(pat_data3['preictal_itvl'], pat_data3[ycol3], marker='d', label=label3, linestyle=lt, color=color)
        if data4 is not None and not pat_data4.empty:
            ax.plot(pat_data4['preictal_itvl'], pat_data4[ycol4], marker='d', label=label4, linestyle='-.', color=mycol['green'])

        ax.set_title(f'Patient {patient}', fontsize=title_font_size)
        ax.set_xticks(pat_data1['preictal_itvl'])
        ax.set_xticklabels(pat_data1['preictal_itvl'])

        if idx == 0:
            ax.legend()


        if (ylable is not None) and  (idx == 0 or col == 0):
                ax.set_ylabel(ylable, fontsize=axis_label_font_size)
        # else:
        #     ax.set_yticklabels([])

        if idx == 0 or row == 3:
            ax.set_xlabel('Preictal Interval', fontsize=axis_label_font_size)
        # else:
        #     ax.set_xticklabels([])

        ax.grid(axis='both', alpha=0.3)
        # ax.set_ylim([0.4, 1])

    # Delete the first three subplots to account for the merged first row
    for i in range(3):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    plt.show()


def draw_pp_thresholds():
    data_full = pd.read_csv(RST_PATH / "result_summary.csv")
    data = data_full[data_full['interictal_itvl'] == 1440]
    data = data[data['model_id'] != 'random']
    data = data[~data['preictal_itvl'].isin([150, 180])]
    data = data.sort_values(by='preictal_itvl')

    # CoSP + Dynamic
    cosp = data[data['thd_from'] == 'dynamic'].copy()
    cosp_val = data[data['thd_from'] == 'val'].copy()
    cosp_opt = data[data['thd_from'] == 'tst'].copy()
    cosp_fixed = data[data['thd_from'] == 'fixed'].copy()

    plot_curves_up4(cosp, 'CoSP',
                    cosp_val, 'CoSP + Validation',
                    cosp_opt, 'CoSP + Optimal',
                    cosp_fixed, 'CoSP + Fixed')

def draw_any_four():
    data_full = pd.read_csv(RST_PATH / "result_summary.csv")
    data_full = data_full[~data_full['preictal_itvl'].isin([150, 180])]
    data_full = data_full[data_full['interictal_itvl'] == 1440]

    # CoSP (Dynamic)
    data = data_full[data_full['model_id'] != 'random'].copy()
    data = data.sort_values(by='preictal_itvl')
    cosp_dny = data[data['thd_from'] == 'dynamic'].copy()
    cosp_val = data[data['thd_from'] == 'val'].copy()
    cosp_fix = data[data['thd_from'] == 'fixed'].copy()
    cosp_opt = data[data['thd_from'] == 'tst'].copy()

    # CP
    data_cp = data_full[data_full['model_id'] == 'random'].copy()
    data_cp = data_cp.sort_values(by='preictal_itvl')
    cp_dyn = data_cp[data_cp['thd_from'] == 'dynamic'].copy()
    cp_fix = data_cp[data_cp['thd_from'] == 'fixed'].copy()
    cp_opt = data_cp[data_cp['thd_from'] == 'tst'].copy()
    cp_val = data_cp[data_cp['thd_from'] == 'val'].copy()

    plot_curves_up4(
        cosp_dny, 'CoSP + Dynamic',
        cosp_val, 'CoSP + Validation',
        cosp_opt, 'CoSP + Optimal',
        cosp_fix, 'CoSP + Fixed',
    )


def plot_predictions_multidays_one(pid, pred_itvl, data, sz_id, n_days=3, save=False):
    sz_data = data[data['sz_id'] == sz_id]
    sz_data = sz_data.sort_values(by='seg_epoch')

    start_time = sz_data['datetime'].min().replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = sz_data['datetime'].max().replace(hour=23, minute=59, second=59, microsecond=999999)
    total_days = (end_time - start_time).days + 1

    # Filter out days with no data
    days_with_data = []
    for i in range(total_days):
        day_start = start_time + pd.Timedelta(days=i)
        day_end = day_start + pd.Timedelta(days=1)
        day_data = sz_data[(sz_data['datetime'] >= day_start) & (sz_data['datetime'] < day_end)]
        if not day_data.empty:
            days_with_data.append((day_start, day_end, day_data))

    if (n_days > 0) and (len(days_with_data) > n_days): days_with_data = days_with_data[-n_days:]

    fig, axes = plt.subplots(len(days_with_data), 1, figsize=(30, 5 * len(days_with_data)), sharex=False)
    if len(days_with_data) == 1: axes = [axes]

    for i, (day_start, day_end, day_data) in enumerate(days_with_data):
        # print(f"Day {i + 1}: day_start={day_start}, day_end={day_end}, num_samples={len(day_data)}")
        axes[i].plot(day_data['datetime'], day_data['sigmoid_prediction'], color='gray', alpha=0.4, label='raw predictions')
        axes[i].plot(day_data['datetime'], sigmoid(smooth_logit_per_sz(day_data, 45)),
                     color=mycol['blue'], linestyle='-', linewidth=2, label=f'45-minute smoothed predictions')
        thd = get_opt_thd(pid, pred_itvl)
        axes[i].axhline(y=thd, color=mycol['red'], linestyle='-.', label=f'threshold({thd})') # plot threshold

        preictal_data = day_data[day_data['label'] == 1]
        if not preictal_data.empty:
            axes[i].axvline(x=preictal_data['datetime'].min(), color='orange', linestyle='--')
            axes[i].fill_between(preictal_data['datetime'], 0, 1, color='orange', alpha=0.4, label='Preictal Interval')
            # Add vertical red line and upside down triangle to indicate seizure onset
            max_time = day_data['datetime'].max()
            seizure_onset_time = max_time + pd.Timedelta(minutes=1)
            axes[i].axvline(x=seizure_onset_time, color='red', linewidth=2, linestyle='-', label='Seizure Onset')
            # axes[i].scatter(seizure_onset_time, 1, color='red', marker='v', s=100)  # upside-down triangle

        axes[i].set_title(
            f'Patient: {pid}, Seizure ID: {sz_id}, Day:{day_start.year}-{day_start.month}-{day_start.day}',
            fontsize=16)
        axes[i].set_ylabel('Probability', fontsize=14)
        if i == len(days_with_data) - 1: axes[i].legend(loc='lower right', fontsize=14)

        # Set fixed x-axis from 00:00 to 24:00
        axes[i].set_xlim(pd.Timestamp(day_start.date()), pd.Timestamp(day_end.date()))
        axes[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[i].tick_params(axis='x', rotation=45)

    axes[-1].set_xlabel('Time', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        f_save = RST_PATH / f"figures/{PAT_LIST_ALL[pid - 1]}/predictions_plots/p{pid}_sz{sz_id}_pre{pred_itvl}_pdays{n_days}.png"
        f_save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f_save)
        print(f"{f_save.name} has been saved to {f_save.parent}.")
    else:
        plt.show()


def plot_predictions_multidays(pid=-1, pred_itvl=45, inter_itvl=-1, sz_id=-1, n_sz=-1, n_days=3, save=True):
    assert pred_itvl in PRE_ITVL_TESTED
    assert inter_itvl in [1440, -1]
    if pid <= 0:
        pids = PAT_ID_LIST
    else:
        assert pid in PAT_ID_LIST
        pids = [pid]

    for pid_one in pids:
        print(f"\nPlotting prediction plots of multiday: pid={pid_one}, "
              f"pred_itvl={pred_itvl}, inter_itvl={inter_itvl}, n_days={n_days}, save={save}")
        mid = get_final_model_id(pid_one, pred_itvl)
        file_path = RST_PATH / f'detail/model{mid}_round1_pred_{inter_itvl}_tst.csv'
        data = pd.read_csv(file_path)

        # Prepare the data
        data['sigmoid_prediction'] = sigmoid(data['prediction'])
        data['datetime'] = pd.to_datetime(data['seg_epoch'], unit='s')
        unique_sz_ids = data['sz_id'].unique()

        if sz_id <= 0:
            if (n_sz > 0) and (len(unique_sz_ids) > n_sz): unique_sz_ids = unique_sz_ids[:n_sz]
            for sz_one in unique_sz_ids:
                plot_predictions_multidays_one(pid_one, pred_itvl, data, sz_one, n_days=n_days, save=save)
        else:
            plot_predictions_multidays_one(pid_one, pred_itvl, data, sz_id, n_days=n_days, save=save)

def plot_predictions_1440(pid, pred_itvl, inter_itvl=1440, align_midnight=False, save=False):
    assert pred_itvl in PRE_ITVL_TESTED
    assert inter_itvl in [1440, -1]
    if pid <= 0:
        pids = PAT_ID_LIST
    else:
        assert pid in PAT_ID_LIST
        pids = [pid]

    for id_one in pids:
        mid = get_final_model_id(id_one, pred_itvl)
        file_path = RST_PATH / f'detail/model{mid}_round1_pred_{inter_itvl}_tst.csv'
        data = pd.read_csv(file_path)

        data['sigmoid_prediction'] = sigmoid(data['prediction'])
        data['datetime'] = pd.to_datetime(data['seg_epoch'], unit='s')
        sz_ids = data['sz_id'].unique()

        n_row, n_col = 5, 4
        n_plot = n_row * n_col
        for i in range(0, len(sz_ids), n_plot):
            sz_subset = sz_ids[i:i + n_plot]
            fig, axes = plt.subplots(n_row, n_col, figsize=(50, 25), sharex=False, sharey=True)
            axes = axes.flatten()

            for ax, sz_id in zip(axes, sz_subset):
                sz_data = data[data['sz_id'] == sz_id]
                seizure_onset = sz_data['datetime'].max()
                start_time = seizure_onset - pd.Timedelta(hours=24)

                day_data = sz_data[(sz_data['datetime'] >= start_time) & (sz_data['datetime'] <= seizure_onset)]

                ax.plot(day_data['datetime'], day_data['sigmoid_prediction'], color='gray', alpha=0.4,
                        label='raw predictions')
                ax.plot(day_data['datetime'], sigmoid(smooth_logit_per_sz(day_data, 45)),
                        color=mycol['blue'], linestyle='-', linewidth=3, label='45-minute smoothed predictions')
                thd = get_opt_thd(id_one, pred_itvl)
                ax.axhline(y=thd, color='red', linestyle='-.', label=f'threshold({thd})')

                preictal_data = day_data[day_data['label'] == 1]
                if not preictal_data.empty:
                    ax.axvline(x=preictal_data['datetime'].min(), color='orange', linestyle='--')
                    ax.fill_between(preictal_data['datetime'], 0, 1, color='orange', alpha=0.4, label='Preictal Interval')
                    seizure_onset_time = seizure_onset + pd.Timedelta(minutes=1)
                    ax.axvline(x=seizure_onset_time, color='red', linewidth=2, linestyle='-', label='Seizure Onset')
                    ax.scatter(seizure_onset_time, 1, color='red', marker='v', s=100)

                ax.set_title(f'Patient: {id_one}, Seizure ID: {sz_id}', fontsize=16)
                ax.set_ylabel('Probability', fontsize=14)

                if align_midnight:
                    day_start = seizure_onset.replace(hour=0, minute=0, second=0, microsecond=0)
                    day_end = day_start + pd.Timedelta(hours=24)
                    ax.set_xlim(day_start, day_end)
                else:
                    ax.set_xlim(start_time, seizure_onset)

                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.tick_params(axis='x', rotation=45)

            axes[-1].set_xlabel('Time', fontsize=14)
            plt.tight_layout()
            if save:
                f_save = RST_PATH / (f"figures/{PAT_LIST_ALL[id_one - 1]}"
                                     f"/predictions_plots/"
                                     f"p{id_one}_pre{pred_itvl}_pday1-algin{align_midnight}_sz-all-{i // 10 + 1}.png")
                f_save.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(f_save)
                print(f"{f_save.name} has been saved to {f_save.parent}.")
            else:
                plt.show()


def select_samples(type, n_sample, pred, target_class=None):
    if target_class is not None:
        assert target_class in [0, 1], f"ERROR: target_class should be eithor 0 or 1, target_class={target_class}"

    if (type == 'top') and (target_class == 1):  # select the top predictions for class 1
        sel_samples = pred.nlargest(n_sample, 'prediction').reset_index(drop=True)

    elif (type == 'top') and (target_class == 0):  # select the "top" predictions for class 0
        sel_samples = pred.nsmallest(n_sample, 'prediction').reset_index(drop=True)

    elif type == 'top-pre':  # Top samples in preictal
        pred_sel = pred[(pred['label'] == 1)].reset_index(drop=True)
        n_sample = len(pred_sel) if len(pred_sel) < n_sample else n_sample
        sel_samples = pred_sel.nlargest(n_sample, 'prediction').reset_index(drop=True)

    elif type == 'top-inter':  # Top samples in interictal
        pred_sel = pred[(pred['label'] == 0)].reset_index(drop=True)
        n_sample = len(pred_sel) if len(pred_sel) < n_sample else n_sample
        sel_samples = pred_sel.nsmallest(n_sample, 'prediction').reset_index(drop=True)

    elif type == 'random-pre':  # Randomly select preictal samples
        pred_sel = pred[pred['label'] == 1].reset_index(drop=True)
        n_sample = min(len(pred_sel), n_sample)
        sel_samples = pred_sel.sample(n=n_sample, random_state=42).reset_index(drop=True)

    elif type == 'random-inter':  # Randomly select interictal samples
        pred_sel = pred[pred['label'] == 0].reset_index(drop=True)
        n_sample = min(len(pred_sel), n_sample)
        sel_samples = pred_sel.sample(n=n_sample, random_state=42).reset_index(drop=True)

    elif type.startswith('time-'):
        hour_start, hour_end, period = int(type.split('-')[1]), int(type.split('-')[2]), type.split('-')[3]
        assert period in ['inter', 'pre'], f"ERROR: unknown seizure period ({period}), available: inter, pre."
        sel_label = 0 if period == 'inter' else 1
        sel_pred = pred[(pred['label'] == sel_label)].reset_index(drop=True)

        # choose samples within the time period
        sel_pred['datetime'] = pd.to_datetime(sel_pred['seg_epoch'], unit='s')
        # Extract hour from datetime
        sel_pred['hour'] = sel_pred['datetime'].dt.hour
        # Filter rows where hour is within [hour_start, hour_end]
        sel_pred = sel_pred[(sel_pred['hour'] >= hour_start) & (sel_pred['hour'] <= hour_end)]
        # Drop the auxiliary columns if not needed
        sel_pred = sel_pred.drop(columns=['datetime', 'hour']).reset_index(drop=True)
        if target_class == 1:
            sel_samples = sel_pred.nlargest(n_sample, 'prediction').reset_index(drop=True)
        else:
            sel_samples = sel_pred.nsmallest(n_sample, 'prediction').reset_index(drop=True)

    else:
        raise ValueError(f"ERROR: unknown sample select type ({type})")
    return sel_samples


def generate_avg_coherence(pat_id, type, pre_itvl=45, target_class=None, n_sample=30, sz_id=-1, save=True):
    print(f"\nPat{pat_id}: Generating average coherence: patient={pat_id}, type={type}, pre_itvl={pre_itvl}, "
          f"target_class={target_class}, n_sample={n_sample}, sz_id={sz_id}, save={save}")

    assert pat_id in PAT_ID_LIST, f"ERROR: unknown patient id={pat_id}, available={PAT_ID_LIST}"
    if type in ['top-pre', 'top-inter', 'ictal', 'random-pre', 'random-inter']:
        target_class = None

    start_step, end_step = None, None
    if type == 'ictal':
        pat = PAT_LIST[PAT_ID_LIST==pat_id]
        f_seglist = INFO_PATH / f"segment_list/{pat}/{pat}_segment_list_ictal_10_4000.csv"
        seg_list = pd.read_csv(f_seglist)
        if sz_id != -1:
            seg_list = seg_list[seg_list['sz_id'] == sz_id].reset_index()
        if len(seg_list) > n_sample:
            sel_samples = seg_list.sample(n=n_sample, random_state=42)
        else:
            sel_samples = seg_list

        if sel_samples.empty:
            return None
        print(f"P{pat_id}, selected {len(sel_samples)} samples.")
    else:
        # Load predictions
        mid = get_final_model_id(pat_id, pre_itvl)
        # inter_itvl = 1440 if type == 'top-tp' else -1
        inter_itvl = 1440
        file_path = RST_PATH / f'detail/model{mid}_round1_pred_{inter_itvl}_tst.csv'
        print(f"P{pat_id}, loading predictions from CoSP, {file_path}...")
        pred = pd.read_csv(file_path)
        pred['probability'] = sigmoid(pred['prediction'])
        if sz_id != -1:
            pred = pred[pred['sz_id'] == sz_id].reset_index()
        # Choose samples to be explained according to type
        sel_samples = select_samples(type, n_sample, pred, target_class)
        if sel_samples.empty:
            return None
        print(f"P{pat_id}, selected {len(sel_samples)} samples, average_probability={sel_samples['probability'].mean():.2f}.")

    # Load coherence matrices
    coh = load_tst_coh(pat_id)
    avg_coherence = np.zeros((N_FREQ, N_CHN_PAIR), dtype=float)

    for idx, row in sel_samples.iterrows():
        # print(f"P{pat_id}, {idx + 1}/{len(sel_samples)}: processing {row['seg_epoch']}...")
        image_one = coh.get(row['seg_epoch'], None)
        if image_one is None and type == 'ictal':
            start_step, end_step = row['seg_start_step'], row['seg_end_step']
            image_one = compute_coherence_one_segment(pat_id, row['seg_epoch'], row['file_name'], start_step, end_step)
        # image_one = compute_coherence_one_segment(pat_id, row['seg_epoch'], row['file_name'], start_step, end_step)
        if image_one is None:
            continue
        if isinstance(image_one, torch.Tensor):
            image_one = image_one.cpu().numpy()
        avg_coherence += image_one
    avg_coherence /= len(sel_samples)

    # Save aggregated coherence
    if save:
        f_avg_coh = RST_PATH / (f"figures/Pat{pat_id}/coh_average/"
                                f"Pat{pat_id}_pre{pre_itvl}_avg_coherence_sz{sz_id}_{type}_class{target_class}_{n_sample}.npy")
        my_save(avg_coherence, f_avg_coh)

    return avg_coherence
