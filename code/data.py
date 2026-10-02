import os
import pickle
import random
import shutil
from pathlib import Path
import torch
from joblib import Parallel, delayed
from scipy import signal
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from config import CONFIG, INFO_PATH, DATA_PATH, PAT_LIST, ANNO_PATH, PAT_LIST_ALL, REC_OFFSET, WORK_PATH, rank, \
    PAT_DICT, EEG_CHN, RST_PATH, PAT_ID_LIST
import time, re, calendar, math
from util import my_print, load_mat_to_np, get_rec_seg_list_name, fill_na, get_saved_coh_filename, get_segment_steps, \
    print_dict
from datetime import datetime, timedelta
n_jobs = -1  # all cpu cores
from concurrent.futures import ProcessPoolExecutor
import numpy as np

"""Private recording paths are supplied through COSP_DATA_PATH.
Supported filename layouts use Patient_<participant>/Data_<YYYY>_<MM>_<DD>/
Hour_<HH>/UTC_<HH>_<MM>_<SS>.mat (or CUTC_ for the alternate record format).
"""

# <editor-fold desc="------ EEG related ------">
class EEGDataset(Dataset):
    def __init__(self, list_df, data_type, config, phase, rank=None):
        assert phase in ['train', 'test', 'analysis']
        self.list_df = list_df
        self.config = config
        self.device = f'cuda:{rank}' if isinstance(rank,int) else 'cpu'
        self.data_type = data_type
        self.record_dic = {}
        self.segments_dic = {}
        self.phase = phase
        pat = PAT_DICT[config['pat_id']]
        self.patient = pat
        self.reorder_idx = reordered_channel_pairs()

        f_coh_expect = get_coh_name(pat, data_type, config=config)
        if not f_coh_expect.exists():
            my_print(1, f"{f_coh_expect.name} does not exist.")
            f_coh = get_saved_coh_filename(f_coh_expect)

            if f_coh != f_coh_expect:
                my_print(1, f"load coherence data from {f_coh.name}")
            else:
                my_print(1, f"generating {f_coh.name}")
        else:
            f_coh = f_coh_expect
        if f_coh.exists():
            my_print(1, f"load coherence data from {f_coh.name}")
            with open(f_coh, 'rb') as f:
                self.segments_dic = pickle.load(f)
                self.segments_dic = {k: v.to(self.device) for k, v in self.segments_dic.items()}

    def save_segments_dic(self, filename):
        """Saves the segments_dic dictionary to a file."""
        segments_dic_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in self.segments_dic.items()}
        with open(filename, 'wb') as file:
            pickle.dump(segments_dic_cpu, file)
        my_print(1, f"{filename.name} has been saved to {filename.parent}: n_segs={len(self.segments_dic)}")
        if self.phase == 'test':
            # to release the gpu memory
            self.segments_dic = {}
            torch.cuda.empty_cache()

    def __len__(self):
        return len(self.list_df)

    def __getitem__(self, index):
        # row = self.list_df.iloc[index]
        # label = torch.tensor(row['label'], dtype=torch.int64).to(self.device)
        # seg_epoch = row['seg_epoch']
        # if seg_epoch in self.segments_dic.keys():
        #     # print(f"{seg_epoch} is in saved coh_file.")
        #     seg_coh = self.segments_dic[seg_epoch]
        # else:
        #     # print(f"{seg_epoch} is not in saved coh_file.")
        #     file_name = row['file_name']
        #     if file_name not in self.record_dic.keys():
        #         rec_np, _ = load_mat_to_np(file_name)
        #         # cache one record
        #         self.record_dic = {file_name: rec_np}
        #     else:
        #         rec_np = self.record_dic[file_name]
        #     start, end = int(row['seg_start_step']), int(row['seg_end_step'])
        #
        #     seg_one_org = rec_np[start:end, :]
        #     seg_one_org = fill_na(seg_one_org)
        #
        #     # compute coherence
        #     seg_coh, _ = compute_coherence(seg_one_org, CONFIG['freq'], CONFIG['n_chn'], CONFIG['detrend'], freq_range=[0, 170])
        #     seg_coh = seg_coh.T
        #     seg_coh = torch.tensor(seg_coh, dtype=torch.float32, device=self.device)
        #
        #     # cache the segment
        #     if self.phase == 'train':
        #         self.segments_dic[seg_epoch] = seg_coh
        #
        # return seg_coh, label, seg_epoch

        # Loop forward until a valid cached segment is found
        while index < len(self.list_df):
            row = self.list_df.iloc[index]
            seg_epoch = row['seg_epoch']

            if seg_epoch in self.segments_dic:
                seg_coh = self.segments_dic[seg_epoch]
                label = torch.tensor(row['label'], dtype=torch.int64).to(self.device)

                # Reorder the x-axis (channel pairs)
                seg_coh = seg_coh[:, self.reorder_idx]  # shape: [F, 120] → reordered

                return seg_coh, label, seg_epoch, int(row['sz_id'])
            else:
                # Skip this index and check the next
                index += 1

        # If no valid sample is found (only possible at end of dataset), raise IndexError
        raise IndexError("No valid cached segment found.")


def load_segment_eeg_2d(seg_list, config, data_type=None, sz_id=None):
    my_print(1, f"Loading EEG data: data_type={data_type}, sz_id={sz_id}  ")
    data_list, label_list = load_segment_eeg_3d(seg_list, config, data_type, sz_id)

    # Concatenate the lists of data and labels
    eeg_seg = torch.cat(data_list, dim=0)
    eeg_seg = eeg_seg.view(-1, eeg_seg.shape[2])
    eeg_seg = eeg_seg.cpu().numpy()
    segs_df = pd.DataFrame(eeg_seg, columns=[f'Ch{i + 1}' for i in range(config['n_chn'])])  # [batch_size * n_steps, 16]

    # Concatenating labels
    eeg_labels = torch.cat(label_list, dim=0)
    eeg_labels = eeg_labels.cpu().numpy()
    # extend labels to the same length with data
    n_steps = data_list[0].shape[1]  # n_steps vary after pre-process
    eeg_labels = np.repeat(eeg_labels, n_steps)
    return segs_df, eeg_labels


def load_segment_eeg_3d(seg_list, config, data_type, sz_id):
    if sz_id is not None:
        seg_list = seg_list[seg_list['sz_id'] == sz_id]

    dset = EEGDataset(seg_list, data_type=data_type, config=config, phase='analysis', rank=None)
    batch_size = CONFIG['batch_size']
    data_loader = DataLoader(dset, batch_size=batch_size, shuffle=True)
    data_list, label_list = [], []
    for data, label in data_loader:
        data_list.append(data)
        label_list.append(label)
    return data_list, label_list


def load_eeg_rec_from_list(seg_list):
    rec_names = np.unique(seg_list['file_name'].to_numpy())
    my_print(1, f"\nLoad EEG records from seg_list: n_records={len(rec_names)}")
    start = time.time()
    rec_dict = {}
    results = Parallel(n_jobs=-1)(
        delayed(lambda f: (f, load_mat_to_np(f)[0]))(f) for f in rec_names
    )
    for f, eeg in results:
        rec_dict[f] = eeg
    end = time.time()
    my_print(1, f"Load EEG records end, taking {end-start:.1f} secondes.")
    return rec_dict
# </editor-fold>



# <editor-fold desc="------ basic functions ------">
def pad_to_target_size(x, target_size, pad_value=0):
    """
    Pad a 2D matrix to (target_size, target_size), where target_size is from config['vit_image_size'].
    Padding is applied to the bottom and right sides.
    """
    h, w = x.shape
    assert h <= target_size and w <= target_size, "Input dimensions must be smaller than or equal to target size"

    pad_height = target_size - h
    pad_width = target_size - w

    padded = np.pad(
        x,
        pad_width=((0, pad_height), (0, pad_width)),  # (top, bottom), (left, right)
        mode='constant',
        constant_values=pad_value
    )
    return padded


def generate_records_info():
    pat_idx = list(range(1, len(PAT_LIST)+1))
    rec_start = pd.read_csv(ANNO_PATH / 'record_start.csv')['record_start'].tolist()
    rec_len = pd.read_csv(ANNO_PATH / 'record_lengths.csv').iloc[0].tolist()

    n_sz_all = []
    n_sz_12 = []
    for p in PAT_LIST:
        p_ano = pd.read_csv(ANNO_PATH / f'{p}_Annots.csv')
        n_sz_all.append(p_ano.shape[0])
        sz_type = np.array(p_ano['sz_type'])
        n_sz_12.append(np.sum((sz_type == 1) |(sz_type == 2)))

    info = pd.DataFrame({'patient': PAT_LIST,
                         'patient_id': pat_idx,
                         'rec_start_epoch': rec_start,
                         'rec_len_seconds': rec_len,
                         'n_sz_all': n_sz_all,
                         'n_sz_12': n_sz_12})
    info.to_csv(ANNO_PATH /'pat_info.csv', index=False)
    return

def epoch_to_path(epoch_time, pat):
    struct_time = time.gmtime(epoch_time)
    year, month, day, hour, minute, second = struct_time[:6]
    path = DATA_PATH / f"Patient_{pat}/Data_{year}_{month:02d}_{day:02d}/Hour_{hour:02d}/UTC_{hour:02d}_{minute:02d}_{second:02d}.mat"
    return path

def path_to_time(path):
    path_str = str(path)
    # Regular expression to capture the date and time parts
    # (?:UTC|CUTC): supported recording filename prefixes
    regex = re.compile(r"Data_(\d{4})_(\d{2})_(\d{2})/Hour_(\d{2})/(?:UTC|CUTC)_(\d{2})_(\d{2})_(\d{2})\.mat")
    match = regex.search(path_str)
    if match:
        year, month, day, hour, _, minute, second = map(int, match.groups())
        day_of_week = datetime(year, month, day).weekday()
        struct_time = time.struct_time((year, month, day, hour, minute, second, day_of_week, 0, 0))
        # calendar.timegm to convert the structured time to epoch time. This function considers the time as UTC
        # time.mktime to convert the structured time to epoch time. This function considers the local time,
        # so it might be affected by the time zone of the system where it's run.
        epoch_time = calendar.timegm(struct_time)
        return epoch_time, struct_time
    else:
        raise ValueError(f'Error in getting time from path: {Path(path_str).absolute()}')


def draw_eeg(f=None, label=None, same_range=True):
    # f = f_inter
    # label = 'interictal'

    eeg_np, _ = load_mat_to_np(f)
    n_channels = eeg_np.shape[1]
    fig, axes = plt.subplots(8, 2, figsize=(15, 10), sharey=same_range, sharex=True)
    axes = axes.ravel()  # Flatten the 4x4 grid to make indexing easier
    # Loop through each channel to plot the curves
    for i in range(n_channels):
        axes[i].plot(eeg_np[:, i], linestyle='-', linewidth=0.2)
        axes[i].set_ylabel(f'C{i+1}', rotation=0)
        axes[i].yaxis.set_label_coords(-0.08, 0.5)
    plt.tight_layout()
    plt.savefig(INFO_PATH / f'{label}2_eeg_1m_range-{same_range}.png')
    my_print(1, 'EEG plots are saved to /datainfo/')

def generate_annotations_csv():
    """
    Convert the expected .mat annotation format to .csv.
    """
    f_all = [f for f in (DATA_PATH / 'Annotations').iterdir() if f.suffix == '.mat']
    for f in f_all:
        df, _ = load_mat_to_np(f)
        f_name = ANNO_PATH / f'{f.stem}.csv'
        df.to_csv(f_name, index=False)
        my_print(1, f'{f_name.name} has been saved to {f_name.parent}')
    return


def get_inter_pre_zones(pat, pred_itvl_min=CONFIG['pred_itvl_min'], pre_itvl_min=CONFIG['pre_itvl_min'],
                        inter_itvl_min=CONFIG['inter_itvl_min']):
    anno = pd.read_csv(ANNO_PATH / f'{pat}_Annots.csv')
    anno = anno.sort_values(by='sz_epoch')
    pat_start_epoch = anno['pat_start_epoch'].iloc[0]
    day100_epoch = pat_start_epoch + 100 * 24 * 60 * 60

    # only consider lead seizure with type 1 or 2
    sz_id = anno['sz_id'].to_numpy()
    select_idx = anno['lead_12'].to_numpy()

    select_idx[anno['sz_epoch'] < day100_epoch] = False


    pre_ends = np.array(anno['sz_epoch'] - pred_itvl_min * 60)
    pre_starts = np.array(pre_ends - pre_itvl_min * 60)
    inter_ends = pre_starts

    if inter_itvl_min > 0:
        tmp = np.array(inter_ends - inter_itvl_min * 60)
        # exclude the first 100 days
        inter_starts = np.concatenate([
            np.array([max(day100_epoch, tmp[0])]),
            np.maximum(tmp, np.roll(anno['sz_4h'], shift=1))[1:]
        ])
    else:
        # no time limit for interictal
        inter_starts = np.concatenate([
            np.array([day100_epoch]),
            np.roll(anno['sz_4h'], shift=1)[1:]])

    zones_df = pd.DataFrame({'sz_id': sz_id[select_idx],
                             'inter_starts': inter_starts[select_idx],
                             'inter_ends': inter_ends[select_idx],
                             'pre_starts': pre_starts[select_idx],
                             'pre_ends': pre_ends[select_idx]})
    return zones_df


def update_anno_all():
    pat_info = pd.read_csv(ANNO_PATH / 'pat_info.csv', index_col='patient')
    col_names = ['sz_id', 'idx(unknown)', 'pat_start_epoch', 'sz_duration', 'us_from_pat_start',
                 'sz_type', 'lead', 'lead_12', 'sz_epoch', 'sz_4h']

    for pat in PAT_LIST_ALL:
        pat_start_epoch = int(pat_info.loc[pat, 'rec_start_epoch'])
        anno = pd.read_csv(ANNO_PATH / f'{pat}_Annots.csv')
        anno = anno.sort_values(by='sz_epoch')
        anno_col = anno.columns

        # rename columns
        anno.rename({'lead_12_index': 'lead_12_id', 'lead_sz': 'lead'}, inplace=True)

        # drop unwanted columns
        rmv_col = set(anno_col) - set(col_names)
        anno.drop(list(rmv_col), axis=1, inplace=True)

        # add columns
        if 'sz_id' not in anno_col:
            anno['sz_id'] = np.arange(1, anno.shape[0] + 1).astype(int)
        if 'pat_start_epoch' not in anno_col:
            anno['pat_start_epoch'] = np.repeat(pat_start_epoch, anno.shape[0])
        if 'sz_4h' not in anno_col:
            anno['sz_4h'] = anno['sz_epoch'] + 4 * 60 * 60
        if 'lead' not in anno_col:
            flag = np.array(anno['sz_epoch'] > np.roll(anno['sz_4h'], shift=1))[1:]
            anno['lead'] = [True] + list(flag)
        if 'lead_12' not in anno_col:
            anno['lead_12'] = anno['sz_type'].isin([1, 2]) & anno['lead_sz']

        anno = anno[col_names]
        anno.to_csv(ANNO_PATH / f'{pat}_Annots.csv', index=False)
        my_print(1, 'annotations have been updated.')


# </editor-fold>



# <editor-fold desc="------ segment related ------">
def get_slide_strides(rec_list, n_segs_expect):
    """
    Calculate the strides for each record based on its time span to make the number of generated segments is close to
    n_segs_expect, and the segments are evenly distrubted (but possibly overlapped) and minimize the duplicates.
    """

    seg_len_step = CONFIG['seg_len_sec'] * CONFIG['freq']
    n_records = len(rec_list)
    strides = np.zeros(n_records, dtype=int)
    total_time_span = np.sum(rec_list['seg_end_step'] - rec_list['seg_start_step'])
    if total_time_span == 0:
        my_print(1, "Error: Total time span is zero, cannot calculate strides.")
        return strides

    time_spans = rec_list['seg_end_step'] - rec_list['seg_start_step']
    # 1.4: give extra room for dropout segments to meet n_segs_expect
    seg_counts = np.round(n_segs_expect * 1.4 * (time_spans / total_time_span)).astype(int)
    # deal with all values in seg_counts are 0
    if np.all(seg_counts == 0):
        seg_counts[seg_counts == 0] = 1
        excess_segments = np.sum(seg_counts) - n_segs_expect
        if np.sum(seg_counts) > n_segs_expect:
            # Randomly reduce segments to match n_segs_expect
            idx = np.random.choice(range(n_records), n_segs_expect, replace=False)
            seg_counts = seg_counts[idx]

    seg_counts = np.array(seg_counts)
    for idx in range(len(seg_counts)):
        n_seg_one = seg_counts[idx]
        if n_seg_one == 0:
            continue
        time_span = time_spans.iloc[idx]

        total_seg_length = n_seg_one * seg_len_step
        if total_seg_length > time_span:
            # Calculate the overlap per segment
            overlap_per_seg = (total_seg_length - time_span) / (n_seg_one - 1)
            one_stride = seg_len_step - math.ceil(overlap_per_seg)
        else:
            # If there's no need for overlap
            one_stride = seg_len_step
        strides[idx] = one_stride
    return strides


def split_rec_to_segs(row_np, seg_len_step, stride, keep_ending=True):
    # names of row_np:
    # ['file_name', 'rec_start_epoch','rec_steps', 'sz_id', 'seg_start_step', 'seg_end_step']
    start = row_np[-2]  # seg_start_step
    end =  min(row_np[2], row_np[-1])  # seg_end_step
    if (stride <= 0) or ((end - start) < seg_len_step): return []

    rec_np, _ = load_mat_to_np(row_np[0])  # row_np[0] is file name
    end = rec_np.shape[0]
    if keep_ending:
        # keep the ending of rec_np by overlapping the last segment
        last_seg_start = end - seg_len_step
        seg_start_steps = np.concatenate([np.arange(start, last_seg_start, stride),
                                          np.array([last_seg_start])])
        seg_end_steps = seg_start_steps + seg_len_step
        num_segments = len(seg_start_steps)
    else:
        # discard ending if it is not long enough for a segment
        seg_start_steps = np.arange(start, end, stride)
        seg_end_steps = seg_start_steps + seg_len_step
        valid_indices = seg_end_steps <= end
        seg_start_steps = seg_start_steps[valid_indices]
        seg_end_steps = seg_end_steps[valid_indices]
        num_segments = len(seg_start_steps)

    # Create a NumPy array with the same shape as the resulting list of lists
    seg_array = np.tile(row_np, (num_segments, 1))
    # Modify the columns that change
    seg_array[:, -2] = seg_start_steps  # Update seg_start_step
    seg_array[:, -1] = seg_end_steps  # Update seg_end_step

    # check dropout
    sel_idx = []
    for index in range(seg_array.shape[0]):
        start, end = seg_start_steps[index], seg_end_steps[index]
        seg_one = rec_np[start:end, :]

        # Discard the data if more than CONFIG['dropout_thd'] (default:10%) of rows are all NaN
        total_rows = seg_one.shape[0]
        nan_rows = np.isnan(seg_one).all(axis=1).sum()
        nan_percentage = (nan_rows / total_rows) * 100
        if nan_percentage <= CONFIG['dropout_thd']:
            sel_idx.append(index)

    if len(sel_idx) == 0:
        return []
    seg_array = seg_array[sel_idx, :]
    return seg_array


def generate_seg_list_inter(pat):
    f_rec = get_rec_seg_list_name(pat, 'rec', 'inter')
    assert f_rec.exists(), f"{f_rec.name} does not exist."
    rec_list_df = pd.read_csv(f_rec)
    my_print(1, f"{pat}: Generating interictal segment list, n_inter_records={len(rec_list_df):,} ...")

    seg_len_step = CONFIG['seg_len_sec'] * CONFIG['freq']
    smp_method = CONFIG['inter_sample']
    if smp_method == 'non-overlap':
        sel_recs = rec_list_df
    elif smp_method.startswith('1segNmin-'):
        n_minutes = int(smp_method.split('-')[1])
        sel_idx = np.arange(0, rec_list_df.shape[0], n_minutes)
        sel_recs = rec_list_df.iloc[sel_idx].copy().reset_index(drop=True)
        sel_recs['seg_end_step'] = sel_recs['seg_start_step'] + seg_len_step
    else:
        raise ValueError(f"Unknown inter_sample={smp_method}")

    sel_recs_np = sel_recs.to_numpy()
    seg_lengths = [seg_len_step] * len(sel_recs_np)
    strides = [seg_len_step] * len(sel_recs_np)
    if smp_method == 'non-overlap':
        keep_ending = [False] * len(sel_recs_np)
    else:
        keep_ending = [True] * len(sel_recs_np)
    segs_np = []
    with ProcessPoolExecutor() as executor:
        for one in executor.map(split_rec_to_segs, sel_recs_np, seg_lengths, strides, keep_ending):
            if len(one) > 0:
                segs_np.append(one)

    segs_np = np.concatenate(segs_np, axis=0)
    my_print(3, f"len(sel_recs_np)={len(sel_recs_np):,}  len(segs_np)={len(segs_np):,}")

    n_seg_limit = CONFIG['n_seg_limit']
    if segs_np.shape[0]  > n_seg_limit:
        n_seg_old = segs_np.shape[0]
        idx = np.random.choice(segs_np.shape[0], size=n_seg_limit, replace=False)
        segs_np = segs_np[idx, :]
        my_print(2, f"down-sample n_segments of interictal from {n_seg_old} to {segs_np.shape[0]}")

    seg_list_df = pd.DataFrame(segs_np, columns=rec_list_df.columns)
    seg_list_df['seg_epoch'] = seg_list_df['rec_start_epoch'] + seg_list_df['seg_start_step'] / CONFIG['freq']
    seg_list_df.sort_values(by='seg_epoch', inplace=True)

    f = get_rec_seg_list_name(pat, 'seg', 'inter')
    seg_list_df.to_csv(f, index=False)
    my_print(1, f"{f.name} has been saved to {f.parent}: n_segs={len(seg_list_df):,}")
    return seg_list_df


def generate_seg_list_pre(pat, n_segs_expect=None):
    my_print(1, f"{pat}: Generating preictal segment list...")
    f_rec = get_rec_seg_list_name(pat, 'rec', 'pre')
    assert f_rec.exists(), f"{f_rec.name} does not exist."
    rec_list_df = pd.read_csv(f_rec)

    smp_method = CONFIG['pre_sample']
    assert smp_method in ['non-overlap', 'fixed']
    if smp_method == 'fixed' and n_segs_expect is None:
        raise ValueError(f"if pre_sample is fixed, n_segs shouldn't be None.")

    recs_np = rec_list_df.to_numpy()
    seg_len_step = CONFIG['seg_len_sec'] * CONFIG['freq']
    seg_lengths = [seg_len_step] * len(recs_np)

    if smp_method == 'non-overlap':
        keep_ending = [False] * len(recs_np)
        strides = np.repeat(seg_len_step, len(rec_list_df))
        n_segs_expect = -1
    else:
        keep_ending = [True] * len(recs_np)
        strides = get_slide_strides(rec_list_df, n_segs_expect)

    segs_np = []
    with ProcessPoolExecutor() as executor:
        for one in executor.map(split_rec_to_segs, recs_np, seg_lengths, strides, keep_ending):
            if len(one) > 0:
                segs_np.append(one)
    segs_np = np.concatenate(segs_np, axis=0)
    my_print(2,
             f"n_pre_segs={len(segs_np):,} sample_method={smp_method}  expected={n_segs_expect:,}")

    if smp_method == 'fixed' and len(segs_np) > n_segs_expect:
        idx = np.random.choice(range(len(segs_np)), n_segs_expect, replace=False)
        segs_np = segs_np[idx, :]

    seg_list_df = pd.DataFrame(segs_np, columns=rec_list_df.columns)
    seg_list_df['seg_epoch'] = seg_list_df['rec_start_epoch'] + seg_list_df['seg_start_step'] / CONFIG['freq']
    seg_list_df.sort_values(by='seg_epoch', inplace=True)

    f = get_rec_seg_list_name(pat, 'seg', 'pre')
    seg_list_df.to_csv(f, index=False)
    my_print(1, f"{f.name} has been saved to {f.parent}: n_segs={len(seg_list_df):,}")
    return seg_list_df


def gen_seg_list_pat(pat):
    f_seg = get_rec_seg_list_name(pat, 'seg', 'inter')
    if f_seg.exists():
        segs_list_inter = pd.read_csv(f_seg)
        my_print(1, f"{f_seg.name} exists.")
    else:
        segs_list_inter = generate_seg_list_inter(pat)

    f_seg = get_rec_seg_list_name(pat, 'seg', 'pre')
    if f_seg.exists():
        segs_list_pre = pd.read_csv(f_seg)
        my_print(1, f"{f_seg.name} exists.")
    else:
        segs_list_pre = generate_seg_list_pre(pat, n_segs_expect=segs_list_inter.shape[0])

    if CONFIG['trn_label'] == 'ictal':
        f_seg = get_rec_seg_list_name(pat, 'seg', 'ictal')
        if f_seg.exists():
            segs_list_ictal = pd.read_csv(f_seg)
            my_print(1, f"{f_seg.name} exists.")
        else:
            segs_list_ictal = generate_seg_list_ictal(stride=4000)
    else:
        segs_list_ictal = None

    return segs_list_inter, segs_list_pre, segs_list_ictal



def split_segment_list_one(pat, label):
    f_seglist = get_rec_seg_list_name(pat, 'seg', label)
    if not f_seglist.exists():
        if label == 'ictal':
            generate_seg_list_ictal(stride=4000)
        else:
            gen_seg_list_pat(pat)
    list_df = pd.read_csv(f_seglist).reset_index(drop=True)

    list_df['label'] = 1 if label == 'pre' else 0
    if label == 'ictal': list_df['label'] = -1

    trn_list, val_list, tst_list  = split_segment_base(list_df, pat, CONFIG)
    my_print(1, f"Split segments for label {label}: n_segs_trn={len(trn_list)}   n_segs_val={len(val_list)}    "
                f"n_tst_seg={len(tst_list)} ")

    return trn_list, val_list, tst_list


def split_segment_base(list_df, pat, config):
    f = WORK_PATH / f"data/{pat}/{pat}_sz_id_used.csv"
    assert f.exists(), f"ERROR: {f.name} does not exist."
    sz_id = np.sort(np.loadtxt(f).astype(int))
    trn_split_idx = int(len(sz_id) * config['trn_sz_pct'])
    val_split_idx = int(len(sz_id) * (config['trn_sz_pct'] + config['val_sz_pct']))
    # Divide unique 'sz_id' into three groups
    trn_ids = sz_id[:trn_split_idx]
    my_print(2, f"sz_ids in training:{trn_ids}")
    val_ids = sz_id[trn_split_idx:val_split_idx]
    my_print(2, f"sz_ids in validation:{val_ids}")
    tst_ids = sz_id[val_split_idx:]
    my_print(2, f"sz_ids in testing:{tst_ids}")

    # Create dataframes for trn, val, and tst based on 'sz_id'
    trn_list = list_df[list_df['sz_id'].isin(trn_ids)].sort_values('seg_epoch').reset_index(drop=True)
    val_list = list_df[list_df['sz_id'].isin(val_ids)].sort_values('seg_epoch').reset_index(drop=True)
    tst_list = list_df[list_df['sz_id'].isin(tst_ids)].sort_values('seg_epoch').reset_index(drop=True)
    return trn_list, val_list, tst_list


def split_segment_list(pat, label):
    if label is None:
        label = CONFIG['trn_label']
    assert label in ['pre', 'inter', 'both', 'ictal'], f"ERROR: wrong label={label}."

    trn_list_pre, val_list_pre, tst_list_pre = split_segment_list_one(pat, 'pre')
    trn_list_inter, val_list_inter, tst_list_inter = split_segment_list_one(pat, 'inter')

    if label == 'ictal':
        trn_list_ictal, val_list_ictal, _ = split_segment_list_one(pat, 'ictal')
        train_df = trn_list_ictal
        val_df = val_list_ictal
    elif label== 'pre':
        train_df = trn_list_pre
        val_df = val_list_pre
    elif label == 'inter':
        train_df = trn_list_inter
        val_df = val_list_inter
    else: # label == 'both'
        train_df = pd.concat([trn_list_pre, trn_list_inter], ignore_index=True)
        val_df = pd.concat([val_list_pre, val_list_inter], ignore_index=True)

    test_df = pd.concat([tst_list_pre, tst_list_inter], ignore_index=True)
    return train_df, val_df, test_df



def clean_duplicate_seg_list(pat):
    folder = INFO_PATH / f"segment_list/{pat}"
    f_segs = [f for f in folder.iterdir() if '_non-overlap_non-overlap_' in f.name]
    for f in f_segs:
        seg_list = pd.read_csv(f)
        seg_list['rec_start_epoch'] = seg_list['rec_start_epoch'].astype(int)
        seg_list['seg_epoch'] = seg_list['seg_epoch'].astype(int)
        # Step 1: Remove all duplicate rows based on all columns
        seg_list = seg_list.drop_duplicates()
        # Step 2: Find 'seg_epoch' values that are not unique and remove rows with those 'seg_epoch' values
        # This can be done by marking rows with duplicated 'seg_epoch' (both first and subsequent occurrences)
        duplicated_seg_epochs = seg_list.duplicated('seg_epoch', keep=False)
        # Keep rows where 'seg_epoch' is not duplicated
        seg_list = seg_list[~duplicated_seg_epochs]
        seg_list.to_csv(f, index=False)
# </editor-fold>



# <editor-fold desc="------ record related ------">
def roundup_datetime(dt, m, mode='ceil'):
    # Calculate the total minutes (hours * 60 + minutes)
    total_minutes = dt.hour * 60 + dt.minute

    if mode == 'ceil':
        # Calculate the remaining minutes to reach the next m-minute mark
        remainder = m - (total_minutes % m)

        # If there are any seconds or microseconds, go to the next m-minute mark
        if dt.second > 0 or dt.microsecond > 0:
            remainder = m - (total_minutes % m)
    elif mode == 'floor':
        # Calculate the remaining minutes since the last m-minute mark
        remainder = total_minutes % m
    else:
        raise ValueError("Invalid mode. Use 'ceil' or 'floor'.")

    # Calculate the new total minutes
    new_total_minutes = total_minutes + remainder if mode == 'ceil' else total_minutes - remainder

    # Calculate the new hour and minute
    new_hour, new_minute = divmod(new_total_minutes, 60)

    # Adjust the day if necessary
    new_day = dt.day
    if mode == 'ceil':
        if new_hour >= 24:
            new_hour -= 24
            new_day += 1  # Note: This simplistic addition may not work correctly at month or year boundaries
    elif mode == 'floor':
        if new_hour > dt.hour or (new_hour == dt.hour and new_minute > dt.minute):
            dt -= timedelta(days=1)
            new_day = dt.day

    # Round the datetime to the nearest m minutes based on the mode
    return dt.replace(day=new_day, hour=new_hour, minute=new_minute, second=0, microsecond=0)


def list_interval_epochs(t1, t2, minutes):
    # Convert epoch to datetime objects
    dt1 = datetime.fromtimestamp(t1)
    dt2 = datetime.fromtimestamp(t2)

    # convert d1 and d2 to the nearst next minute
    dt1 = roundup_datetime(dt1, minutes, 'floor')
    dt2 = roundup_datetime(dt2, minutes, 'ceil')

    # Initialize an empty list to store the epochs
    epoch_list = []

    # Loop through each minute between t1 and t2
    current_time = dt1
    while current_time < dt2:
        # Append the epoch time of the current minute to the list
        cur_epoch = current_time.timestamp()
        epoch_list.append(cur_epoch)

        # Increment the current time by one minute
        current_time += timedelta(minutes=minutes)

    return epoch_list


def list_candidate_rec(pat, zones):
    # zones = get_inter_pre_zones(pat)
    rec_list = []
    for i in range(zones.shape[0]):
        start_epoch = zones['inter_starts'].iloc[i]
        end_epoch = zones['pre_ends'].iloc[i]

        # list the starting epoch of each recording
        rec_starts = list_interval_epochs(start_epoch, end_epoch, CONFIG['rec_len_min'])
        for rs in rec_starts:
            f = epoch_to_path(rs, pat)
            if f.exists():
                rec_list.append(f)

    rec_list = np.unique(rec_list)
    return rec_list


def generate_seg_list_ictal(config=CONFIG, stride=4000):
    pat = config['patient']
    anno_df = pd.read_csv(ANNO_PATH / f"{pat}_Annots.csv")
    anno_df = anno_df[anno_df['sz_type'].isin([1, 2])]
    pat_start_epoch = anno_df['pat_start_epoch'].iloc[0]
    sz_durations = anno_df['sz_duration'].to_numpy()
    sz_offset_us = anno_df['us_from_pat_start'].to_numpy()
    sz_starts = pat_start_epoch + sz_offset_us / (1000 * 1000) # convert mircosecond to second
    sz_id = anno_df['sz_id'].to_numpy()
    freq = config['freq']
    seg_len_step = config['seg_len_sec'] * freq

    seg_list = []
    for i in range(len(sz_starts)):
        sz_start = sz_starts[i]
        sz_end = sz_start + sz_durations[i]
        rec_starts_epoch = list_interval_epochs(sz_start, sz_end, config['rec_len_min'])

        for idx,rs in enumerate(rec_starts_epoch):
            f = epoch_to_path(rs, pat)
            if f.exists():
                rec_np, _ = load_mat_to_np(f)
                # names of row_np:
                # ['file_name', 'rec_start_epoch','rec_steps', 'sz_id', 'seg_start_step', 'seg_end_step']
                seg_start_step = int((max(sz_start, rs) - rs) * freq)
                seg_end_step = int((min(sz_end, rs + config['rec_len_min'] * 60) - rs) * freq)
                rec_one = [str(f), rs, rec_np.shape[0], sz_id[i], seg_start_step, seg_end_step]
                segs_from_rec = split_rec_to_segs(rec_one, seg_len_step, stride=stride, keep_ending=True)
                if len(segs_from_rec) > 0:
                    # check dropout
                    for index, seg_one in enumerate(segs_from_rec):
                        seg_np = rec_np[int(seg_one[-2]):int(seg_one[-1]), :]
                        # fill NA as mean
                        col_means = np.nanmean(seg_np, axis=0)
                        for sn_i in range(seg_np.shape[1]):
                                seg_np[np.isnan(seg_np[:, sn_i]), sn_i] = col_means[sn_i]
                    seg_list.extend(segs_from_rec)

    cols = ['file_name', 'rec_start_epoch','rec_steps', 'sz_id', 'seg_start_step', 'seg_end_step']
    seg_list = pd.DataFrame(np.array(seg_list), columns=cols)
    cols = ['file_name', 'rec_start_epoch', 'n_dropout', 'rec_steps', 'sz_id', 'seg_start_step', 'seg_end_step', 'seg_epoch']
    seg_list['seg_epoch'] = seg_list['rec_start_epoch'].astype(float) + seg_list['seg_start_step'].astype(float) / freq
    seg_list['n_dropout'] = -1
    seg_list = seg_list[cols]
    seg_list['label'] = 2  # 0:interictal  1:preictal

    # f_name = INFO_PATH / f"segment_list/{pat}/{pat}_segment_list_ictal_{CONFIG['seg_len_sec']}_{stride}.csv"
    f_name = get_rec_seg_list_name(pat, 'seg', 'ictal', config=config)
    seg_list.to_csv(f_name, index=False)
    my_print(1, f"{f_name.name} (n_records={seg_list.shape[0]:,}) has been saved to {f_name.parent}.")
    return seg_list


def process_segment_i(j, i, zones, config, freq, seg_len_step):
    if j == 0:
        start, end = zones['pre_starts'].iloc[i], zones['pre_ends'].iloc[i]
    else:
        start, end = zones['inter_starts'].iloc[i], zones['inter_ends'].iloc[i]
    sz_id = zones['sz_id'].iloc[i]
    rec_epoch = list_interval_epochs(start, end, config['rec_len_min'])
    seg_list = []
    for idx, rs in enumerate(rec_epoch):
        f = epoch_to_path(rs, config['patient'])
        if f.exists():
            rec_np, _ = load_mat_to_np(f)
            if len(rec_np) == 0:
                continue
            seg_start_step = int((max(start, rs) - rs) * freq)
            seg_end_step = int((min(end, rs + config['rec_len_min'] * 60) - rs) * freq)
            rec_one = [str(f), rs, rec_np.shape[0], sz_id, seg_start_step, seg_end_step]
            segs_from_rec = split_rec_to_segs(rec_one, seg_len_step, stride=seg_len_step, keep_ending=False)
            if len(segs_from_rec) > 0:
                for index, seg_one in enumerate(segs_from_rec):
                    seg_np = rec_np[int(seg_one[-2]):int(seg_one[-1]), :]
                    col_means = np.nanmean(seg_np, axis=0)
                    for sn_i in range(seg_np.shape[1]):
                        seg_np[np.isnan(seg_np[:, sn_i]), sn_i] = col_means[sn_i]
                seg_list.extend(segs_from_rec)
    return seg_list

def update_label(pat, seg_list):
    # Load the annotations
    anno = pd.read_csv(ANNO_PATH / f'{pat}_Annots.csv')
    merged = seg_list.merge(anno[['sz_id', 'sz_epoch']], on='sz_id', how='left')
    merged['label'] = ((merged['sz_epoch'] - merged['seg_epoch']) <= CONFIG['pre_itvl_min'] * 60).astype(int)
    merged.drop(columns=['sz_epoch'], inplace=True)
    return merged


class PAT_DICTconfig:
    pass


def generate_non_overlap_seg_list(config):
    pat = PAT_DICT[config['pat_id']]
    f_base = INFO_PATH / f"segment_list/{pat}/{pat}_segment_list_10sec_non-overlap_non-overlap_base.csv"
    if f_base.exists():
        seg_list = generate_non_overlap_seglist_from_base(pat, f_base, config['pre_itvl_min'], config['inter_itvl_min'])
        # f_name = get_rec_seg_list_name(pat, 'seg', label=None, config=config)
        # seg_list.to_csv(f_name, index=False)
    else:
        seg_list = generate_non_overlap_seglist_base(config, pat, f_base)

    labels = seg_list['label'].to_numpy()
    my_print(1, f"{pat} Load segments list: n_pre={np.sum(labels)}  n_inter={len(labels) - np.sum(labels)}")

    # split
    trn_list, val_list, tst_list = split_segment_base(seg_list, pat, config)
    my_print(1, f"{pat} Split segments list: n_trn={len(trn_list)}  n_val={len(val_list)}  n_tst={len(tst_list)}")
    return trn_list, val_list, tst_list


def generate_non_overlap_seglist_base(config, pat, f_name):
    start = time.time()
    my_print(1, f"Generating segment list {f_name.name} for {pat}....")
    zones = get_inter_pre_zones(pat,
                                pred_itvl_min=config['pred_itvl_min'],
                                pre_itvl_min=config['pre_itvl_min'],
                                inter_itvl_min=config['inter_itvl_min'])
    freq = config['freq']
    seg_len_step = config['seg_len_sec'] * freq

    seg_list = pd.DataFrame()
    for j in range(2):  # iterate interictal and preictal
        # Parallel execution for each 'i'
        results = Parallel(n_jobs=-1)(
            delayed(process_segment_i)(j, i, zones, config, freq, seg_len_step) for i in range(zones.shape[0]))

        # Combine results
        all_segs = []
        for segs in results:
            all_segs.extend(segs)

        seg_df = pd.DataFrame(np.array(all_segs),
                              columns=['file_name', 'rec_start_epoch', 'rec_steps', 'sz_id', 'seg_start_step',
                                       'seg_end_step'])
        seg_df['seg_epoch'] = seg_df['rec_start_epoch'].astype(float) + seg_df['seg_start_step'].astype(
            float) / freq
        seg_df['label'] = 1 if j == 0 else 0
        seg_df['n_dropout'] = -1
        seg_df['seg_start_step'] = seg_df['seg_start_step'].astype(int)
        seg_df['seg_end_step'] = seg_df['seg_end_step'].astype(int)
        seg_list = pd.concat([seg_list, seg_df], axis=0, ignore_index=True)

    seg_list.to_csv(f_name, index=False)
    my_print(1, f"{f_name.name} (n_records={seg_list.shape[0]:,}) "
                f"has been saved to {f_name.parent}, taking {int(time.time() - start)} seconds.")
    return seg_list


def generate_rec_list_pat(pat):
    """
    - Generate a record list for a patient containing the eligible records information for extracting segments from.
    - The record list is only related to a permit zone, i.e., [inter_start, pre_end) to make it reused in the case of
    adjusting inter_interval and pre_interval;
    - Each row in the record list corresponds to a 1-minute record. If the original record files are longer than 1
    minute, it will be split to several rows in the record list.
    """
    assert pat in PAT_LIST_ALL


    f_name = get_rec_seg_list_name(pat, 'rec')
    if f_name.exists():
        my_print(1, f"{f_name.name} exists, no need to generate.")
        df = pd.read_csv(f_name)
        separate_rec_list(pat, df)
        return
    my_print(1, f'\nGenerating record list for patient {pat}: {f_name.name}...')

    freq = CONFIG['freq']
    rec_len_min = CONFIG['rec_len_min']
    if rec_len_min < 1:
        raise ValueError(f"Only support record length is equal or larger than 1 minute, current rec_len_min={rec_len_min}.")

    zones = get_inter_pre_zones(pat).set_index('sz_id')
    f_records = list_candidate_rec(pat, zones)
    rec_list = []
    for idx_f, f in enumerate(f_records):
        if idx_f % 5000 == 0:
            my_print(2, f"{idx_f:,}/{len(f_records):,} ...")
        rec_start_epoch, _ = path_to_time(f)

        # find the seizures the record is related to
        idx = np.searchsorted(zones['inter_starts'], rec_start_epoch, side='right') - 1
        if idx < 0 or rec_start_epoch >= zones['pre_ends'].iloc[idx]:
            my_print(2, f"{str(f)} is excluded due to not in permit zone.")
            continue

        # f is eligible
        sz_id = zones.index[idx]
        rec_end_epoch = rec_start_epoch + rec_len_min * 60
        rec_split_start_epoch = max(rec_start_epoch, zones['inter_starts'].iloc[idx])
        rec_split_end_epoch = min(rec_end_epoch, zones['pre_ends'].iloc[idx])
        rec_steps = rec_len_min * 60 * freq - REC_OFFSET

        # save to multiple rows if rec_len_min > 1
        for i_row in range(int(rec_len_min // 1)):
            row_split_start_epoch = rec_split_start_epoch + i_row * 60
            row_split_start_step = int(( row_split_start_epoch - rec_start_epoch) * freq)
            row_split_end_epoch = min(row_split_start_epoch + 60, rec_split_end_epoch)
            row_split_end_step = int(min((row_split_end_epoch - rec_start_epoch) * freq, rec_steps))

            rec_one = [str(f), rec_start_epoch, rec_steps,
                       sz_id, row_split_start_step, row_split_end_step]
            rec_list.append(rec_one)

    col_names = ['file_name', 'rec_start_epoch', 'rec_steps', 'sz_id', 'seg_start_step', 'seg_end_step']
    df = pd.DataFrame(rec_list, columns=col_names)
    f_name = get_rec_seg_list_name(pat, 'rec')
    df.to_csv(f_name, index=False)
    my_print(1, f"{f_name.name} (n_records={df.shape[0]:,}) has been saved to {f_name.parent}.")

    separate_rec_list(pat, df)
    return df


def separate_rec_list(pat, rec_list=None):
    my_print(1, f"\nSeparating record list to interictal and preictal ...")
    if rec_list is None:
        f = get_rec_seg_list_name(pat, 'rec')
        rec_list = pd.read_csv(f)

    f_inter = get_rec_seg_list_name(pat, 'rec', label='inter')
    f_pre = get_rec_seg_list_name(pat, 'rec', label='pre')
    if f_inter.exists() and f_pre.exists():
        my_print(1, f"{f_inter.name} and {f_pre.name} exist, no need to generate.")
        return


    zones = get_inter_pre_zones(pat).set_index('sz_id')
    zones_dict = zones.to_dict(orient='index')
    seg_len_step = CONFIG['seg_len_sec'] * CONFIG['freq']

    # filter out records whose length less than a segment
    valid_mask = (rec_list['seg_end_step'] - rec_list['seg_start_step']) >= seg_len_step
    rec_list = rec_list[valid_mask].reset_index(drop=True)

    seg_start_epoch = (rec_list['rec_start_epoch'] + rec_list['seg_start_step'] / CONFIG['freq']).to_numpy()
    seg_end_epoch = (rec_list['rec_start_epoch'] + rec_list['seg_end_step'] / CONFIG['freq']).to_numpy()
    seg_start_step = rec_list['seg_start_step'].to_numpy()
    seg_end_step = rec_list['seg_end_step'].to_numpy()

    inter_rec_list = []
    pre_rec_list = []
    for idx in range(len(rec_list)):
        sz_id = rec_list.at[idx, 'sz_id']
        inter_end = zones_dict.get(sz_id, {}).get('inter_ends', None)
        pre_start = zones_dict.get(sz_id, {}).get('pre_starts', None)
        if inter_end is None or pre_start is None:
            continue

        if seg_end_epoch[idx] <= inter_end:
            inter_rec_list.append(rec_list.iloc[idx].to_dict())
        elif seg_start_epoch[idx] >= pre_start:
            pre_rec_list.append(rec_list.iloc[idx].to_dict())
        else:
            if (inter_end - seg_start_step[idx]) >= seg_len_step:
                row = rec_list.iloc[idx].copy()
                row['seg_end_step'] = inter_end
                inter_rec_list.append(row.to_dict())

            if (seg_end_step[idx] - pre_start) >= seg_len_step:
                row = rec_list.iloc[idx].copy()
                row['seg_start_step'] = pre_start
                pre_rec_list.append(row.to_dict())

    inter_rec_df = pd.DataFrame(inter_rec_list, columns=rec_list.columns)
    f = get_rec_seg_list_name(pat, 'rec', label='inter')
    inter_rec_df.to_csv(f, index=False)
    my_print(1, f"{f.name } has been saved to {f.parent}: n_inter_records={len(inter_rec_df):,}")

    pre_rec_df = pd.DataFrame(pre_rec_list, columns=rec_list.columns)
    f = get_rec_seg_list_name(pat, 'rec', label='pre')
    pre_rec_df.to_csv(f, index=False)
    my_print(1, f"{f.name} has been saved to {f.parent}: n_pre_records={len(pre_rec_df):,}")

    return inter_rec_df, pre_rec_df
# </editor-fold>


# <editor-fold desc="------ coherence related ------">
def compute_coh_channel(data, i, j, fs, detrend):
    if detrend == 'none':
        detrend = False
    f, Cxy = signal.coherence(data[:, i], data[:, j], fs=fs, detrend=detrend)
    return i, j, f, Cxy


def compute_coherence(data, fs, n_channels, detrend, freq_range=None):
    coherence_results = Parallel(n_jobs=-1)(delayed(compute_coh_channel)(data, i, j, fs, detrend)
                                            for i in range(n_channels)
                                            for j in range(i, n_channels))
    coh = np.zeros((n_channels, n_channels, len(coherence_results[0][2])))
    for i, j, f, Cxy in coherence_results:
        coh[i, j, :] = Cxy
        coh[j, i, :] = Cxy

    # select frequency range
    if freq_range is not None:
        idx = np.where((f >= freq_range[0]) & (f < freq_range[1]))[0]
        f = f[idx]
        coh = coh[:, :, idx]

    stack_coh = []
    # Iterate over the first axis and extract the values under the diagonal
    # now shape of coh: (n_channel, n_channel, n_frequency)
    for i in range(1, coh.shape[0]):
        one = coh[:i, i, :]  # shape: (i, n_frequency)
        stack_coh.append(one)
    # Stack the extracted 2D arrays vertically to form a single 2D array
    coh = np.vstack(stack_coh)

    # coh shape (120, 109), i.e., (n_channel * (n_channel-1)/2, n_freuqency)
    return coh, f


def combine_coh_batch(pat, marker):
    folder = WORK_PATH / f"data/{pat}"
    coh_batches = [f for f in folder.iterdir() if (f.suffix == '.pkl') and (marker in f.name)]
    if len(coh_batches) == 0:
        print(f"No pkl file is found under folder={folder}, marker={marker}")
        return

    combined_dict = {}
    for f in coh_batches:
        print(f"combining from {f.name}...")
        with open(f, 'rb') as file:
            data = pickle.load(file)
            combined_dict.update(data)

    f_coh = folder / f"{pat}_coh_{len(combined_dict)}_non-overlap_non-overlap_coh-0-170-stack_tst.pkl"
    # Save the combined dictionary to a new .pkl file
    with open(f_coh, 'wb') as file:
        pickle.dump(combined_dict, file)
    print(
        f"{f_coh.name} has been saved to {f_coh.parent}, combining from {len(coh_batches)} files: n_seg={len(combined_dict)}")

    # Delete the individual .pkl files
    for f in coh_batches:
        if f != f_coh:
            f.unlink()
            print(f"Deleted {f.name}")


def get_coh_name(pat, data_type, config=CONFIG):
    preprocess = 'coh-0-170-stack'
    if data_type == 'ictal':
        f_coh = WORK_PATH / f"data/{pat}/{pat}_coh_non-overlap_non-overlap_10_coh-0-170-stack_ictal.pkl"
    elif data_type == 'tst':
        start = config['inter_itvl_min'] + config['pre_itvl_min'] + config['pred_itvl_min']
        end = config['pred_itvl_min']
        f_coh = WORK_PATH / (f"data/{pat}/{pat}_coh_{config['seg_len_sec']}_{start}_{end}_{config['inter_sample']}_"
                             f"{config['pre_sample']}_{preprocess}_tst.pkl")
    else:
        f_name = get_rec_seg_list_name(pat, 'seg', label=None, config=config).stem
        f_name = f_name.replace('_segment_list_', '_coh_')
        if '_1segNmin-1_fixed_' in f_name:
            f_name = f_name.replace('_1segNmin-1_fixed_', '_non-overlap_')
        f_coh = WORK_PATH / f"data/{pat}/{f_name}_{preprocess}_{data_type}.pkl"
    return f_coh


def compute_coh_pat(pat, data_type, test_flag=False):
    assert (pat in PAT_LIST)
    assert (data_type in ['tst', 'ictal'])

    config = CONFIG
    config['inter_sample'] = 'non-overlap'
    config['pre_sample'] = 'non-overlap'

    config['inter_itvl_min'] = 3 * 60
    config['test_inter_itvl'] = 3 * 60
    #     # to fit the GPU memory
    #     config['inter_itvl_min'] = 3 * 60
    #     config['test_inter_itvl'] = 3 * 60
    # else:
    #     config['inter_itvl_min'] = 1 * 24 * 60
    #     config['test_inter_itvl'] = 1 * 24 * 60
    config['pre_itvl_min'] = 4
    config['pat_id'] = pat  # TODO

    f_coh = get_coh_name(pat, data_type, config=config)
    my_print(1, f"Generating {f_coh.name}...")
    if data_type == 'tst':
        _, _, seg_list = generate_non_overlap_seg_list(config)
    elif data_type == 'ictal':
        seg_list = generate_seg_list_ictal(config, stride=4000)

    seg_dataset = EEGDataset(seg_list, data_type, config=config, phase='analysis', rank=rank)
    batch_size = config['batch_size']
    tst_loader = DataLoader(seg_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_idx, (data, _, _) in enumerate(tst_loader):
            if data.size(0) == 1:
                continue
            if test_flag and batch_idx >= 2: break
            my_print(1, f"Batch [{batch_idx + 1}/{len(tst_loader)}]...")
    f_coh.parent.mkdir(parents=True, exist_ok=True)
    seg_dataset.save_segments_dic(f_coh)

    with open(f_coh, 'rb') as f:
        coh_load = pickle.load(f)
        print(f"Load the saved coh file, len(coh)={len(coh_load)}")

    print("Done")
    return


def compute_coherence_one_segment(pat_id, epoch, file_name, start_step=None, end_step=None, device='cpu'):
    rec_np, _ = load_mat_to_np(file_name)
    if start_step is None or end_step is None:
        start, end = get_segment_steps(pat_id, epoch)
    else:
        start, end = start_step, end_step
    if start is None or end is None: return None
    seg_one_org = rec_np[start:end, :]
    seg_one_org = fill_na(seg_one_org)
    # compute coherence
    image_one, _ = compute_coherence(seg_one_org,
                                     CONFIG['freq'],
                                     CONFIG['n_chn'],
                                     CONFIG['detrend'],
                                     freq_range=[0, 170])
    image_one = image_one.T # convert shape to (109, 120)
    image_one = torch.tensor(image_one, dtype=torch.float32, device=device)
    return image_one

def move_coh_to_cpu(pat):
    f_coh = get_coh_name(pat, 'tst', config=CONFIG)
    f_coh = get_saved_coh_filename(f_coh)
    print(f"load coherence data from {f_coh.name}")
    if f_coh.exists():
        with open(f_coh, 'rb') as f:
            coh = pickle.load(f)
            print(f"load {f_coh.name} : n_segs={len(coh)}")
            # Check the device of the first tensor in the dictionary
            for key, value in coh.items():
                if torch.is_tensor(value):
                    print(f"Key: {key}, Device: {value.device}")
                    coh_device = str(value.device)
                else:
                    print(f"Key: {key} contains a non-tensor value or a nested structure.")
                break  # Stop after checking the first item
        print(coh_device)
        if coh_device and coh_device != 'cpu':
            print('converting to on cpu')
            segments_dic_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in coh.items()}
            with open(f_coh, 'wb') as file:
                pickle.dump(segments_dic_cpu, file)
            print(f"{f_coh.name} has been saved to {f_coh.parent}: n_segs={len(coh)}")
    print('done.')


# </editor-fold>


def generate_base_from_45m(pat):
    f_base = INFO_PATH/f"segment_list/{pat}/{pat}_segment_list_10_-1_46_1_non-overlap_non-overlap_10.csv"
    seglist_base = pd.read_csv(f_base)
    seglist_base['rec_start_epoch'] = seglist_base['rec_start_epoch'].astype(int)
    seglist_base['seg_epoch'] = seglist_base['seg_epoch'].astype(int)
    seglist_base.to_csv(f_base, index=False)
    seglist_base.drop(['n_dropout', 'label'], axis=1, inplace=True)

    anno = pd.read_csv(INFO_PATH/f"annotation/{pat}_Annots.csv").sort_values(by='sz_id')
    anno['prev_sz_epoch'] = anno['sz_epoch'].shift(1).fillna(0).astype(int)

    merged_df = seglist_base.merge(anno[['sz_id', 'sz_epoch', 'prev_sz_epoch']], on='sz_id', how='left', suffixes=('', '_next'))
    # Calculate the seconds from the previous seizure
    merged_df['sec_from_prev_sz'] = merged_df['seg_epoch'] - merged_df['prev_sz_epoch']
    merged_df['sec_to_next_sz'] = merged_df['sz_epoch'] - merged_df['seg_epoch']

    # Drop the temporary columns used for calculations
    merged_df.drop(['prev_sz_epoch'], axis=1, inplace=True)
    f_base = INFO_PATH / f"segment_list/{pat}/{pat}_segment_list_10sec_non-overlap_non-overlap_base.csv"
    merged_df.to_csv(f_base, index=False)
    my_print(1, f"{f_base.name} saved to {f_base.parent}.")
    return merged_df


def generate_non_overlap_seglist_from_base(pat, f_base, pre_itvl, inter_itvl):
    # f_base = INFO_PATH / f"segment_list/{pat}/{pat}_segment_list_10sec_non-overlap_non-overlap_base.csv"
    assert f_base.exists(), f"ERROR: {f_base} does not exist."
    seg_base = pd.read_csv(f_base)

    # Filtering out rows that do not meet any condition
    pre_itvl = pre_itvl * 60  # convert minutes to seconds
    inter_itvl = inter_itvl if  inter_itvl < 0 else inter_itvl * 60
    if inter_itvl > 0:
        seg_list = seg_base[(seg_base['sec_to_next_sz'] > 60) &
                            (seg_base['sec_to_next_sz'] < (pre_itvl + inter_itvl + 60))].copy()
    else:
        seg_list = seg_base[seg_base['sec_to_next_sz'] > 60].copy()

    # Apply labeling
    seg_list['label'] = 0
    seg_list.loc[seg_list['sec_to_next_sz'] < pre_itvl + 60, 'label'] = 1
    seg_list['label'] = seg_list['label'].astype(int)
    my_print(1, f"generate_non_overlap_seglist_from_base, n_sz={len(seg_list['sz_id'].unique())}, "
                f"pre_itvl={pre_itvl} seconds, inter_itvl={inter_itvl} seconds, "
                f"n_pre={(seg_list['label'] == 1).sum()}, n_inter={(seg_list['label'] == 0).sum()}")

    return seg_list



# ---------- organize the coherence data on my local computer ------------
def reordered_channel_pairs(n_channels=EEG_CHN, lead_size=4, save_path=None):
    """
    Reorder EEG channel pairs to prioritize intra-lead connections before cross-lead connections.

    Parameters:
    - n_channels (int): Total number of EEG channels.
    - lead_size (int): Number of channels per lead (default is 4).
    - save_path (str or None): If provided, the reordered (i, j) channel pairs will be saved to this file.

    Returns:
    - reordered_indices (np.ndarray): Indices of reordered channel pairs based on original all_pairs order.
    """
    # Generate original channel pair list
    all_pairs = [(i, j) for i in range(1, n_channels + 1) for j in range(i + 1, n_channels + 1)]

    # Group channels into leads
    leads = [list(range(i * lead_size + 1, (i + 1) * lead_size + 1)) for i in range(n_channels // lead_size)]

    # Collect intra-lead pairs
    intra_lead_pairs = [(i, j) for lead in leads for i in lead for j in lead if i < j]

    # Get cross-lead pairs
    intra_lead_set = set(intra_lead_pairs)
    cross_lead_pairs = [pair for pair in all_pairs if pair not in intra_lead_set]
    # Reorder channel pairs
    reordered_pairs = intra_lead_pairs + cross_lead_pairs

    # Map reordered pairs to original indices
    index_map = {pair: idx for idx, pair in enumerate(all_pairs)}
    reordered_indices = np.array([index_map[pair] for pair in reordered_pairs])

    # Save reordered pairs to file (not the indices)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(save_path, reordered_pairs, fmt='%d', delimiter=',')
        print(f"Reordered channel pairs saved to {save_path}")

    return reordered_indices


def convert_pkl_to_npy(input_dir, output_dir, chn_pair_order=None, f_list_save=None):
    """
    Convert all .pkl files in a directory into .npy files by extracting
    each key-value pair from the dictionary inside and saving the value as a .npy file.
    Duplicate keys (based on raw key string) across files will be ignored.

    Parameters:
    - input_dir (str or Path): Directory containing .pkl files.
    - output_dir (str or Path): Directory where .npy files will be saved.
    - chn_pair_order (List[int] or np.ndarray, optional): Optional reordering of channel-pair dimensions.
    - f_list_save (str or Path, optional): Path to save the CSV file listing all saved filenames.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    coh_dir = Path(output_dir) / "coh_reordered"
    coh_dir.mkdir(parents=True, exist_ok=True)

    saved_keys = set()
    saved_filenames = []  # List of saved .npy filenames (without extension)

    # List all .pkl files in input_dir
    pkl_files = list(input_dir.glob("*.pkl"))
    print(f"\n{input_dir.name}: Found {len(pkl_files)} pkl files.")
    for pkl_file in pkl_files:
        print(f"Processing: {pkl_file.name}")
        with open(pkl_file, 'rb') as f:
            try:
                data_dict = pickle.load(f)
            except Exception:
                f.seek(0)  # rewind the file pointer
                try:
                    data_dict = torch.load(f, map_location='cpu')
                except Exception:
                    print(f"❌ Failed to load {pkl_file.name}")
                    continue

        for key, value in data_dict.items():
            key_str = str(key)
            if key_str in saved_keys:
                continue  # Skip duplicates

            # Safe file name
            safe_key = key_str.replace(os.sep, "_").replace(" ", "_")
            npy_path = coh_dir / f"{safe_key}.npy"

            # Optional channel pair reordering
            # shape of value: [109, 120]
            if chn_pair_order is not None:
                value = value[:, chn_pair_order]

            # Move to CPU and convert to NumPy if it's a CUDA tensor
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()

            # save to numpy files
            np.save(npy_path, value)
            saved_keys.add(key_str)
            saved_filenames.append(safe_key)
        print(f"Saved {len(saved_keys)} unique segments so far.")

    # Save file list if requested
    if f_list_save is not None:
        f_list_save = Path(f_list_save)
        f_list_save.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({'filename': saved_filenames})
        df.to_csv(f_list_save, index=False)
        print(f"Saved file list with {len(saved_filenames)} entries to {f_list_save}")


def annotate_coh_list_sz(pat_id):
    """
    Annotates EEG segment epochs from coh_list.txt with:
    - the ID and type of the seizure they belong to (if within a seizure),
      or the next upcoming seizure with lead_12 == True
    - time in seconds to the next seizure
    - ecto = 1 if within a seizure, 0 otherwise

    Saves a CSV file with these annotations.
    """
    coh_list_path = DATA_PATH / f"Pat{pat_id}/coh_list.txt"
    assert coh_list_path.exists(), f"Missing: {coh_list_path}"

    # Load segment epochs
    segment_epochs = pd.read_csv(coh_list_path)
    segment_epochs.columns = ['filename']
    segment_epochs['filename'] = pd.to_numeric(segment_epochs['filename'], errors='coerce')
    segment_epochs = segment_epochs.dropna().reset_index(drop=True)

    # Load seizure annotations (lead_12 only)
    anno_file = ANNO_PATH / f"Pat{pat_id}_Annots.csv"
    assert anno_file.exists(), f"Missing: {anno_file}"
    df_annots = pd.read_csv(anno_file)
    df_annots = df_annots[df_annots['lead_12'] == True].copy()
    df_annots = df_annots.sort_values('sz_epoch')

    results = []
    for epoch in segment_epochs['filename']:
        # Check if this epoch is inside any seizure
        inside_mask = (
                (df_annots['sz_epoch'] <= epoch) &
                (epoch < df_annots['sz_4h'])
        )
        if inside_mask.any():
            sz_row = df_annots[inside_mask].iloc[0]
            ictal_postictal = 1
        else:
            # Find next seizure in the future
            future_sz = df_annots[df_annots['sz_epoch'] > epoch]
            if not future_sz.empty:
                sz_row = future_sz.iloc[0]
                ictal_postictal = 0
            else:
                sz_row = None
                ictal_postictal = 0

        # Fill values
        if sz_row is not None:
            sz_id = int(sz_row['sz_id'])
            sz_type = int(sz_row['sz_type'])
            seconds_to_sz = sz_row['sz_epoch'] - epoch
        else:
            sz_id = -1
            sz_type = -1
            seconds_to_sz = None

        results.append({
            'segment_epoch': epoch,
            'sz_id': int(sz_id),
            'sz_type': int(sz_type),
            'seconds_to_sz': seconds_to_sz,
            'ictal_postictal': ictal_postictal
        })

    # Save to CSV, sorted by epoch
    df_out = pd.DataFrame(results)
    df_out = df_out.sort_values('segment_epoch').reset_index(drop=True)
    f_save = coh_list_path.with_name("coh_list_sz.csv")
    df_out.to_csv(f_save, index=False)
    print(f"Annotated segment list saved to {f_save}")


def summarize_coh_available():
    summary = []

    for pid in PAT_ID_LIST:
        coh_csv = DATA_PATH / f"Pat{pid}/coh_list_used.csv"
        if not coh_csv.exists():
            print(f"[Skip] Missing: {coh_csv}")
            continue

        df_used = pd.read_csv(coh_csv)

        total_seizures = df_used['sz_id'].nunique()
        total_segments = len(df_used)

        patient_summary = {
            "pat_id": pid,
            "n_seizures_type12": total_seizures,
            "n_segments_type12": total_segments,
        }

        # Optional: add per sz_type stats if needed
        sz_types = df_used['sz_type'].unique()
        for sz_type in sz_types:
            df_temp = df_used[df_used['sz_type'] == sz_type]
            patient_summary[f"n_seizures_type{sz_type}"] = df_temp['sz_id'].nunique()
            patient_summary[f"n_segments_type{sz_type}"] = len(df_temp)

        summary.append(patient_summary)

    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.sort_values("pat_id").reset_index(drop=True)

    output_csv = DATA_PATH / "available_coh_summary.csv"
    df_summary.to_csv(output_csv, index=False)
    print(f"Summary saved to: {output_csv}")


def count_train_test_segments():
    sop_sec = 1 * 60  # SOP length in seconds
    preictal_minutes_list = [4, 15, 30, 45, 60, 90, 120, 150, 180]
    pct_train_seizures = np.arange(0.1, 0.7, 0.1)
    results = []

    for pid in PAT_ID_LIST:
        print(f"Processing Pat{pid} ...")
        coh_csv = DATA_PATH / f"Pat{pid}/coh_list_used.csv"
        if not coh_csv.exists():
            print(f"[Skip] Missing data for Pat{pid}")
            continue

        df_coh = pd.read_csv(coh_csv)
        df_coh = df_coh.dropna(subset=['sz_id', 'seconds_to_sz'])
        df_coh['sz_id'] = df_coh['sz_id'].astype(int)
        df_coh['seconds_to_sz'] = pd.to_numeric(df_coh['seconds_to_sz'], errors='coerce')

        sz_ids_sorted = sorted(df_coh['sz_id'].unique())
        n_sz = len(sz_ids_sorted)

        for preictal_min in preictal_minutes_list:
            pre_sec_start = preictal_min * 60
            pre_sec_end = sop_sec
            interictal_threshold = pre_sec_start + pre_sec_end

            for split_pct in pct_train_seizures:
                n_sz_trn = max(1, math.floor(n_sz * split_pct))
                sz_trn = sz_ids_sorted[:n_sz_trn]
                sz_tst = sz_ids_sorted[n_sz_trn:]

                df_trn = df_coh[df_coh['sz_id'].isin(sz_trn)]
                df_tst = df_coh[df_coh['sz_id'].isin(sz_tst)]

                # Define masks
                trn_pre_mask = (df_trn['seconds_to_sz'] < pre_sec_start) & (df_trn['seconds_to_sz'] > pre_sec_end)
                trn_inter_mask = df_trn['seconds_to_sz'] > interictal_threshold

                tst_pre_mask = (df_tst['seconds_to_sz'] < pre_sec_start) & (df_tst['seconds_to_sz'] > pre_sec_end)
                tst_inter_mask = df_tst['seconds_to_sz'] > interictal_threshold

                results.append({
                    "pat_id": pid,
                    "preictal_min": preictal_min,
                    "train_pct": round(split_pct, 2),
                    "n_train_sz": len(sz_trn),
                    "n_test_sz": len(sz_tst),
                    "preictal_train": int(trn_pre_mask.sum()),
                    "interictal_train": int(trn_inter_mask.sum()),
                    "preictal_test": int(tst_pre_mask.sum()),
                    "interictal_test": int(tst_inter_mask.sum()),
                })

    df_summary = pd.DataFrame(results)
    summary_path = DATA_PATH / "coh_train_test_segment_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
