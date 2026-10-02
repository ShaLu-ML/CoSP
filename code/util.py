import ast
import pickle
import shutil
import time

import math
from matplotlib import pyplot as plt

from config import VERBOSE, INFO_PATH, CONFIG, RST_PATH, COMPUTER, f_log, MODEL_PATH, STAT_PATH, WORK_PATH, TEST, \
    PAT_LIST_ALL, PAT_LIST, FINAL_INFO, PAT_ID_LIST, LIME_PATH
import h5py, os
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set the display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

###############   basics   #####################
def range01(weight):
    weight_min = weight.min()
    weight_max = weight.max()
    return (weight - weight_min) / (weight_max - weight_min)

def my_save(data, f_name):
    f_path = Path(f_name) if isinstance(f_name, str) else f_name
    f_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the file suffix
    suffix = f_path.suffix.lower()
    # Save based on file suffix
    if suffix in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']:
        if isinstance(data, plt.Figure):
            data.savefig(f_path, bbox_inches='tight')
        else:
            plt.savefig(f_path, bbox_inches='tight')
        plt.close()
    elif suffix in ['.csv', '.tsv', '.txt']:
        if isinstance(data, pd.DataFrame):
            if suffix == '.tsv':
                data.to_csv(f_path, sep='\t', index=False)
            else:
                data.to_csv(f_path, index=False)
        else:
            if suffix == '.tsv':
                pd.DataFrame(data).to_csv(f_path, sep='\t', index=False)
            else:
                pd.DataFrame(data).to_csv(f_path, index=False)
    elif suffix in ['.npy', '.npz']:
        np.save(f_path, data)
    else:
        raise ValueError(f"ERROR: Unsupported file format {suffix}")
    print(f"{f_path.name} has been saved to {f_path.parent}.")


# Example usage:
# mysave(df, "path/to/data.csv")
# mysave(df, "path/to/data.tsv")
# mysave(df, "path/to/data.txt")



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def my_print(verbose, *args, **kwargs):
    """
    v = 0: no print
    v = 1: normal process
    v = 2: detailed process
    v = 3: debugging
    """
    if verbose <= VERBOSE:
        if not TEST:
            with open(f_log, 'a') as file:
                print(*args, **kwargs, file=file)
        print(*args, **kwargs)

    return


def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

#
# def load_mat_to_np(f_mat):
#     print(f_mat)
#     f_mat = Path(f_mat) if isinstance(f_mat, str) else f_mat
#
#     # Check the type of the .mat file
#     file_info = os.popen(f"file {f_mat}").read()
#     np_arrays, keys = [], []
#     if "Hierarchical Data Format" in file_info:
#         with h5py.File(f_mat, 'r') as f:
#             for key in f.keys():
#                 keys.append(str(key))
#                 da = f[key][()]
#                 np_arrays.append(da)
#
#     elif "Matlab" in file_info:
#         data = loadmat(f_mat)
#         for key, value in data.items():
#             keys.append(str(key))
#             if isinstance(value, np.ndarray):
#                 np_arrays.append(value)
#     else:
#         raise ValueError(f'Error when loading file {f_mat.name}')
#
#     # Concatenate along axis 1 (columns)
#     if len(np_arrays) > 0:
#         final_array = np.concatenate(np_arrays, axis=1)
#     else:
#         final_array = np.array([])  # Empty array
#
#     # return shape: [n_channel, n_time_steps]
#     return final_array.T, keys


def load_mat_to_np(f_mat):
    f_mat = Path(f_mat) if isinstance(f_mat, str) else f_mat
    file_info = os.popen(f"file {f_mat}").read()
    np_arrays, keys = [], []

    try:
        if "Hierarchical Data Format" in file_info:
            with h5py.File(f_mat, 'r') as f:
                for key in f.keys():
                    try:
                        keys.append(str(key))
                        da = f[key][()]
                        np_arrays.append(da)
                    except Exception as e:
                        print(f"Error reading dataset {key} in file {f_mat}: {e}")

        elif "Matlab" in file_info:
            try:
                data = loadmat(f_mat)
                for key, value in data.items():
                    keys.append(str(key))
                    if isinstance(value, np.ndarray):
                        np_arrays.append(value)
            except Exception as e:
                print(f"Error loading MATLAB file {f_mat}: {e}")

        else:
            raise ValueError(f'Error when loading file {f_mat.name}')

        if len(np_arrays) > 0:
            final_array = np.concatenate(np_arrays, axis=1)
        else:
            final_array = np.array([])  # Empty array

    except Exception as e:
        print(f"Error processing file {f_mat}: {e}")
        final_array = np.array([])  # Return empty array in case of error

    #return shape: [n_timesteps, n_channels]
    return final_array.T, keys


###############   file process   #####################
def replace_str_in_filename(folder, marker, old, new):
    """
    Replace specific substrings in filenames (that contain a target substring or all files if marker is None)
    within a given folder using pathlib.

    Parameters:
    - folder (str or Path): The path to the folder containing the files.
    - marker (str or None): The substring to identify target files or None to apply to all files.
    - old (str): The substring to be replaced.
    - new (str): The substring to replace with.

    Returns:
    - None. The filenames in the folder are modified.
    """
    folder = Path(folder)
    f_list = [f for f in folder.rglob('*') if f.is_file()]
    count = 0
    for f in f_list:
        # If marker is None or the filename contains the marker
        if marker is None or marker in f.name:
            # Check if the filename contains the old substring
            if old in f.name:
                # Generate the new filename
                new_filename = f.name.replace(old, new)
                # Rename the file
                f.rename(f.parent / new_filename)
                count += 1
    print(f"Have changed {count} filenames.")


def delete_files(folder, marker=None):
    folder = Path(folder)

    if not folder.exists() or not folder.is_dir():
        print(f"Folder '{folder}' does not exist or is not a directory.")
        return

    if marker is None:
        # Remove the folder directly without checking files
        shutil.rmtree(folder)
        print(f"Folder '{folder}' has been removed directly because marker is None.")
        return

    all_files = [f for f in folder.rglob('*') if f.is_file()]

    for f in all_files:
        # Check if the filename contains the target substring
        if marker in f.name:
            # Delete the file
            f.unlink()
            print(f"{f.name} has been deleted.")

    # Check if the folder is empty and remove it if true
    if not any(folder.iterdir()):
        shutil.rmtree(folder)
        print(f"Folder '{folder}' is empty and has been removed.")
    else:
        print(f"Folder '{folder}' still contains files or subdirectories.")



def move_files(source_folder, target_folder, marker=None):
    """
    Move files from source_folder to target_folder. If a marker is provided, only move files
    whose names contain the marker. If no marker is provided, move all files.

    Parameters:
    - source_folder (str or Path): The path to the source folder.
    - target_folder (str or Path): The path to the target folder.
    - marker (str, optional): The marker to identify files to be moved. If None, move all files.

    Returns:
    - None. Files are moved from source to target folder.
    """

    source_path = Path(source_folder)
    assert source_path.exists(), f"ERROR: unkonow directory {source_path}"
    target_path = Path(target_folder)

    # Check if target folder exists, if not, create it
    target_path.mkdir(parents=True, exist_ok=True)

    # Iterate over all files in the source folder
    for file_path in source_path.iterdir():
        # Check if the marker is provided and if the filename contains the marker
        if (marker is None) or (marker in file_path.name):
            # Move the file to the target folder
            file_path.rename(target_path / file_path.name)


def rename_column(f_name, old_name, new_name):
    f_name = Path(f_name)
    if not f_name.exists():
        raise ValueError(f"{f_name} does not exist.")

    df = pd.read_csv(f_name)
    df.rename(columns={old_name: new_name}, inplace=True)
    df.to_csv(f_name, index=False)
    my_print(1, f"rename {old_name} to {new_name}")
    my_print(1, f"current columns names: {df.columns}")


def drop_column(f_name, col_name):
    f_name = Path(f_name)
    if not f_name.exists():
        raise ValueError(f"{f_name} does not exist.")

    df = pd.read_csv(f_name)
    if col_name not in df.columns:
        raise ValueError(f"{col_name} does not exist in {df.columns}.")

    df = pd.read_csv(f_name).drop(col_name, axis=1)
    df.to_csv(f_name, index=False)
    my_print(1, f"{col_name} has been deleted from {str(f_name)}.")
    my_print(1, f"current column names: {df.columns}.")



###############   data related   #####################
def get_rec_seg_list_name(pat, list_type, label=None, config=CONFIG):
    if label == 'ictal':
        # 100: stride
        f = INFO_PATH / f"segment_list/{pat}/{pat}_segment_list_ictal_{config['seg_len_sec']}_{4000}.csv"
        return f

    if label not in ['inter', 'pre']:
        label = None
    assert list_type in ['rec', 'seg']
    if config['inter_itvl_min'] > 0:
        zone_start = config['pred_itvl_min'] + config['pre_itvl_min'] + config['inter_itvl_min']
    else:
        zone_start = -1
    zone_end = config['pred_itvl_min']
    path = INFO_PATH/f'record_list/{pat}' if list_type == 'rec' else INFO_PATH/f'segment_list/{pat}'
    inter_pre_sep = config['pred_itvl_min'] + config['pre_itvl_min']
    seg_len = config['seg_len_sec']

    if label is None:
        if list_type == 'rec':
            f = f"{pat}_record_list_{zone_start}_{zone_end}.csv"
        else:
            f = (f"{pat}_segment_list_{seg_len}_{zone_start}_{inter_pre_sep}_{zone_end}"
                 f"_{config['inter_sample']}_{config['pre_sample']}_{config['dropout_thd']}.csv")
    elif label == 'inter':
        if list_type == 'rec':
            f = f"{pat}_record_list_{label}_{zone_start}_{inter_pre_sep}.csv"
        else:
            f = (f"{pat}_segment_list_{label}_{seg_len}_{zone_start}_{inter_pre_sep}"
                 f"_{config['inter_sample']}_{config['n_seg_limit']}.csv")
    else:
        if list_type == 'rec':
            f = f"{pat}_record_list_{label}_{inter_pre_sep}_{zone_end}.csv"
        else:
            seg_len = config['seg_len_sec']
            if config['pre_sample'] == 'fixed':
                f = (f"{pat}_segment_list_{label}_{seg_len}_{zone_start}_{inter_pre_sep}_{zone_end}"
                     f"_{config['pre_sample']}_{config['n_seg_limit']}.csv")
            else:
                f = f"{pat}_segment_list_{label}_{seg_len}_{inter_pre_sep}_{zone_end}_{config['pre_sample']}.csv"
    f = path / f
    f.parent.mkdir(parents=True, exist_ok=True)
    return f


def get_saved_coh_filename(f_coh):
    key_name = '_'.join(f_coh.name.split('_')[6:])
    f_others = [f for f in f_coh.parent.iterdir() if key_name in f.name]
    if f_others:
        f_len_other = [int(f.name.split('_')[4]) for f in f_others]
        f_new = f_others[f_len_other.index(max(f_len_other))]
    else:
        f_new = f_coh
    return f_new


###############   results related   #####################
def save_result(model_ids, trn_info=None, config=CONFIG, rst_info=None, tst_info=None):
    if all(x is None for x in [trn_info, config, rst_info, tst_info]):
        raise ValueError("All saved information are None.")

    model_id, round_id = model_ids['model_id'], model_ids['round_id']
    f_rst = RST_PATH / "result.csv"
    if f_rst.exists():
        rst_df = load_rst_df(f_rst)
        rst_df = rst_df.astype(str)
        # Convert 'model_id' and 'round_id' columns to integers before comparison
        rst_df['model_id'] = rst_df['model_id'].astype(int)
        rst_df['round_id'] = rst_df['round_id'].astype(int)

        # Find the index where both 'model_id' and 'round_id' match the given values
        row_idx = rst_df[(rst_df['model_id'] == model_id) & (rst_df['round_id'] == round_id)].index
        if row_idx.empty:
            # new row
            row_idx = len(rst_df)
            rst_df.loc[row_idx] = {'model_id': model_id, 'round_id': round_id}
        else:
            # existing row
            row_idx = row_idx[0]
    else:
        rst_df = pd.DataFrame(columns=['patient_id', 'model_id', 'round_id'])
        row_idx = 0
        rst_df.loc[row_idx] = {'model_id': model_id, 'round_id': round_id}

    def update_df(df, row, col_prefix, data_dict):
        # df = df.astype(str)
        for key, value in data_dict.items():
            value = '' if value is None else str(value)   # save to string for comparison
            col_name = f"{col_prefix}_{key}"
            if col_name not in df.columns:
                df[col_name] = ''
            df.at[row, col_name] = value

    if trn_info is not None:
        update_df(rst_df, row_idx, 'TRN', trn_info)
    if config is not None:
        update_df(rst_df, row_idx, 'CFG', config)
    if rst_info is not None:
        update_df(rst_df, row_idx, 'RST', rst_info)
    if tst_info is not None:
        update_df(rst_df, row_idx, 'TST', tst_info)

    all_columns = ['patient_id', 'model_id', 'round_id'] + \
                  [col for col in rst_df.columns if col.startswith('RST_')] + \
                  [col for col in rst_df.columns if col.startswith('CFG_')] + \
                  [col for col in rst_df.columns if col.startswith('TRN_')] + \
                  [col for col in rst_df.columns if col.startswith('TST_')]
    rst_df = rst_df[all_columns]
    rst_df.at[row_idx, 'patient_id'] = int(rst_df.at[row_idx, 'CFG_pat_id']) + 1

    # Remove rows where all columns are empty
    rst_df.replace({'NA': np.nan, '': np.nan}, inplace=True)
    rst_df.dropna(how='all', inplace=True)
    rst_df.to_csv(f_rst, na_rep='', index=False)
    rst_df.to_csv(f_rst.parent/f'{f_rst.stem}_backup.csv', na_rep='', index=False)
    my_print(1, f"{model_ids}: {f_rst.name} has been saved to {f_rst.parent} with: "
                f"trn_info: {bool(trn_info)}; config: {bool(config)}; "
                f"rst_info: {bool(rst_info)}; tst_info: {bool(tst_info)}  ")
    return model_id, round_id


def get_saved_info(model_ids, info_type):
    assert (info_type in ['trn', 'cfg', 'tst', 'rst'])
    info_type = f'{info_type.upper()}_'

    f_rst = RST_PATH / "result.csv"
    if not f_rst.exists():
        return None
    # Read the CSV file into a DataFrame
    rst_df = load_rst_df(f_rst)
    model_id, round_id  = model_ids['model_id'], model_ids['round_id']

    # Filter rows based on model_id and round_id
    row_idx = rst_df[(rst_df['model_id'] == model_id) & (rst_df['round_id'] == round_id)].index

    if row_idx.empty:
        return None

    # Fetch the info
    info_columns = [col for col in rst_df.columns if col.startswith(info_type)]
    info_series = rst_df.loc[row_idx, info_columns].iloc[0]
    # remove prefix
    info_series = info_series.rename(lambda x: x[4:])
    # Convert it into a dictionary
    info = info_series.to_dict()

    return info


def is_list_column(col):
    # Check if at least one non-null element in the column is a string that looks like a list
    flag = any(
        col.dropna().astype(str).apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')))
    return flag


def load_rst_df(f_rst):
    rst_df = pd.read_csv(f_rst, keep_default_na=False, na_values='')
    rst_df.replace({'NA': np.nan, '': np.nan}, inplace=True)
    rst_df.dropna(how='all', inplace=True)

    numeric_placeholder = -1
    for col in rst_df.columns:
        # replace nan, NA and empty rows
        if pd.api.types.is_numeric_dtype(rst_df[col]):
            # Replace NaN with numeric placeholder
            rst_df[col].fillna(numeric_placeholder, inplace=True)
        else:
            # Replace NaN with 'NA' for non-numeric columns
            rst_df[col].fillna('', inplace=True)

        # convert to list if needed
        if is_list_column(rst_df[col]):
            rst_df[col] = rst_df[col].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) and x.startswith('[') and x.endswith(']') else x)

        # convert to integer if needed
        if rst_df[col].dtype == 'float':
            # Check if all non-NaN values in this column are essentially integers
            if rst_df[col].dropna().apply(lambda x: x == np.floor(x)).all():
                # Convert to integer
                rst_df[col] = rst_df[col].astype(int)

    return rst_df


def get_model_ids(config):
    f_rst = RST_PATH / "result.csv"
    if not f_rst.exists():
        model_id, round_id = 1, 1
    else:
        rst_df = load_rst_df(f_rst)
        cfg_cols = set(col[4:] for col in rst_df.columns if col.startswith('CFG_'))
        # Find the intersection of columns that exist in both df and config dictionary
        common_cols = set(config.keys()).intersection(cfg_cols)
        new_keys = set(config.keys()) - cfg_cols

        if common_cols and not new_keys:
            df_com = rst_df[[f'CFG_{col}' for col in common_cols]].astype(str)  # convert to string for comparison
            config_series = pd.Series({key: config[key] for key in common_cols}).rename(
                index=lambda x: f'CFG_{x}').astype(str) # convert to string for comparison

            mask = np.all(df_com.eq(config_series), axis=1)
            if mask.any():
                model_id = rst_df.loc[mask, 'model_id'].iloc[0]
                round_id = rst_df[rst_df['model_id'] == model_id]['round_id'].max() + 1
            else:
                model_id, round_id = rst_df['model_id'].astype(int).max() + 1, 1
        else:
            model_id, round_id = rst_df['model_id'].astype(int).max() + 1, 1

    if model_id is None or math.isnan(model_id):  # Use math.isnan() to check for NaN
        model_id = 1

    model_ids = {'model_id': model_id, 'round_id': round_id}
    return model_ids


def delete_result(model_ids):
    model_id, round_id = model_ids['model_id'], model_ids['round_id']
    # Check if round_id is -1 to delete all files with the model_id
    if round_id == -1:
        model_filter = lambda f: f.name.startswith(f'model{model_id}_')
    else:
        model_filter = lambda f: f.name.startswith(f'model{model_id}_round{round_id}')

    # Update the result.csv file
    pat = None
    f_rst = RST_PATH / "result.csv"
    if f_rst.exists():
        rst_df = load_rst_df(f_rst)
        if rst_df['model_id'].isin([model_id]).any():

            # model_id is in rst_df['model_id']
            pat = rst_df.loc[rst_df['model_id'] == model_id, 'CFG_pat_id'].to_numpy()[0]
            if round_id == -1:
                # Remove all rows with the specified model_id
                rst_df = rst_df[rst_df['model_id'] != model_id]
            else:
                # Remove specific row with model_id and round_id
                row_idx = rst_df[(rst_df['model_id'] == model_id) & (rst_df['round_id'] == round_id)].index
                if not row_idx.empty:
                    row_idx = row_idx[0]
                    rst_df.drop(row_idx, inplace=True)
            rst_df.to_csv(f_rst, index=False)

    # remove relevant files under MODEL_PATH, STAT_PATH RST_PATH
    for path in [MODEL_PATH, STAT_PATH/f"{pat}/", RST_PATH / f'detail']:
        if not path.exists():
            continue
        f_list = [f for f in path.iterdir() if model_filter(f)]
        for f in f_list:
            f.unlink()

    print(f"Test results of model_id={model_id}"
          f"{' and round_id=' + str(round_id) if round_id != -1 else ''} have been deleted.")


def copy_result(model_ids, new_model_ids):
    model_id, round_id = model_ids['model_id'], model_ids['round_id']
    new_model_id, new_round_id = new_model_ids['model_id'], new_model_ids.get('round_id', round_id)

    # Define file copy lambda function
    def copy_file(f, mid, rid):
        new_name = f.name.replace(f'model{model_id}', f'model{mid}')
        if rid != -1:
            new_name = new_name.replace(f'_round{round_id}', f'_round{rid}')
        return f.with_name(new_name)

    for path in [MODEL_PATH, RST_PATH / f'detail']:
        if round_id == -1:
            # Copy all files with the specified model_id
            f_list = [f for f in path.iterdir() if f.name.startswith(f'model{model_id}_')]
        else:
            # Copy specific files with model_id and round_id
            f_list = [f for f in path.iterdir() if f.name.startswith(f'model{model_id}_round{round_id}')]

        for f in f_list:
            target_file = copy_file(f, new_model_id, new_round_id)
            if not target_file.exists():  # Only copy if the file doesn't already exist
                shutil.copy(f, target_file)

    print(f"Test result of model_id={model_id}{' and round_id=' + str(round_id) if round_id != -1 else ''} "
          f"have been copied to model_id={new_model_id}{' and round_id=' + str(new_round_id) if new_round_id != -1 else ''}.")



def rename_result(model_ids, model_ids_new):
    model_id, round_id = model_ids['model_id'], model_ids['round_id']
    new_model_id, new_round_id = model_ids_new['model_id'], model_ids_new.get('round_id', round_id)
    # Define file renaming lambda function
    def rename_file(f, mid, rid):
        new_name = f.name.replace(f'model{model_id}', f'model{mid}')
        if rid != -1:
            new_name = new_name.replace(f'_round{round_id}', f'_round{rid}')
        return f.with_name(new_name)

    # Update the result.csv file
    f_rsts = ["result.csv", "result_backup.csv", "result.csv", "result_full.csv"]
    for f in f_rsts:
        f_rst = RST_PATH / f
        pat = None
        if f_rst.exists():
            rst_df = load_rst_df(f_rst)
            pat = rst_df.loc[rst_df['model_id'] == model_id, 'CFG_patient'].to_numpy()[0]
            if round_id == -1:
                # Rename model_id for all matching entries
                row_idx = rst_df[rst_df['model_id'] == model_id].index
                rst_df.loc[row_idx, 'model_id'] = new_model_id
            else:
                # Rename specific entry with model_id and round_id
                row_idx = rst_df[(rst_df['model_id'] == model_id) & (rst_df['round_id'] == round_id)].index
                if not row_idx.empty:
                    rst_df.loc[row_idx, ['model_id', 'round_id']] = [new_model_id, new_round_id]
            rst_df.to_csv(f_rst, index=False)

    # Rename files under MODEL_PATH and RST_PATH
    for path in [MODEL_PATH, STAT_PATH/f"{pat}", RST_PATH / f'detail']:
        if round_id == -1:
            # Rename all files with the specified model_id
            f_list = [f for f in path.iterdir() if f.name.startswith(f'model{model_id}_')]
        else:
            # Rename specific files with model_id and round_id
            f_list = [f for f in path.iterdir() if f.name.startswith(f'model{model_id}_round{round_id}')]
        for f in f_list:
            f.rename(rename_file(f, new_model_id, new_round_id))

    print(f"Test result of model_id={model_id}{' and round_id=' + str(round_id) if round_id != -1 else ''} "
          f"have been renamed to model_id={new_model_id}{' and round_id=' + str(new_round_id) if new_round_id != -1 else ''}.")


def print_dict(d):
    for key, value in d.items():
        if isinstance(value, float):
            my_print(1, f"{key}: {value:.3f}")
        else:
            my_print(1, f"{key}: {value}")



def fill_na(eeg_np):
    seg = np.copy(eeg_np)
    # Compute column means, ignoring NaNs
    col_means = np.nanmean(eeg_np, axis=0)
    # Replace NaN column means with zero
    col_means[np.isnan(col_means)] = 0
    # Replace NaNs in the data with the corresponding column means
    nan_inds = np.where(np.isnan(eeg_np))
    seg[nan_inds] = np.take(col_means, nan_inds[1])
    return seg


def add_patid_rst():
    f_rst = RST_PATH / "result.csv"
    rst_df = load_rst_df(f_rst)
    rst_df['patient_id'] = rst_df['CFG_patient'].apply(
        lambda x: PAT_LIST_ALL.index(x) + 1 if x in PAT_LIST_ALL else None)
    column_order = ['patient_id'] + [col for col in rst_df.columns if col != 'patient_id']
    rst_df = rst_df[column_order]
    rst_df.replace({'NA': np.nan, '': np.nan}, inplace=True)
    rst_df.dropna(how='all', inplace=True)
    rst_df.to_csv(f_rst, na_rep='', index=False)


def down_sample_inter(df, n_limit):
    if n_limit <= 0:
        return df
    df_inter = df[df['label'] == 0].reset_index(drop=True)
    df_pre = df[df['label'] == 1].reset_index(drop=True)

    interval = max(1, len(df_inter) // n_limit)
    df_down = df_inter.iloc[::interval]
    df_final = pd.concat([df_down, df_pre]).reset_index(drop=True)
    my_print(1, f"Down-sampling from {len(df)} to {len(df_final)}.")
    return df_final


def get_sz_info_from_epochs(pat, epoch_list, data_type):
    """
    Retrieves seizure IDs (sz_id) associated with epochs, specifically the nearest sz_id
    for each epoch in epoch_list where the corresponding seizure epoch (sz_epoch) is nearest to
    but greater than the given epoch.

    Parameters:
    - pat (str): The identifier for the patient, used to locate the annotation file.
    - epoch_list (List[int]): A list of epoch numbers for which the nearest (but greater than) seizure IDs are to be found.

    Returns:
    - np.ndarray: An array of seizure IDs corresponding to the nearest (but greater than) seizure epochs in `epoch_list` for the specified patient.
    """

    f_anno = INFO_PATH / f'annotation/{pat}_Annots.csv'
    anno = pd.read_csv(f_anno)
    sz_ids = []
    sz_types = []


    for epoch in epoch_list:
        # epoch = epoch_list[100]
        # Get epochs and ids where sz_epoch is greater than the current epoch
        if data_type == 'ictal':
            valid_anno = anno[anno['sz_epoch'] <= epoch]
        else:
            valid_anno = anno[anno['sz_epoch'] > epoch]
        valid_anno = valid_anno[valid_anno['lead_12'] == True]

        if not valid_anno.empty:
            closest_epoch_idx = abs(epoch - valid_anno['sz_epoch']).idxmin()
            sz_id = valid_anno.loc[closest_epoch_idx, 'sz_id']
            sz_ids.append(sz_id)
            sz_type = valid_anno.loc[closest_epoch_idx, 'sz_type']
            sz_types.append(sz_type)
        else:
            # Append None or a placeholder if no valid sz_epoch is found
            sz_ids.append(-1)
            sz_types.append(-1)

    sz_ids = np.array(sz_ids)
    sz_types = np.array(sz_types)
    return sz_ids, sz_types



def evaluate_result_seizure_level(pred_pd, thd):
    sz_ids = np.sort(pred_pd['sz_id'].unique())
    pred_pd['prob'] = sigmoid(pred_pd['prediction'].to_numpy())
    sz_preds = []
    n_seg_eva = len(pred_pd)
    for idx, sz in enumerate(sz_ids):
        pred_sz = pred_pd.loc[pred_pd['sz_id'] == sz].reset_index(drop=True)
        pre_sz = pred_sz.loc[pred_sz['label']==1].reset_index(drop=True)
        if len(pre_sz) == 0:
            # some seizures do not have preictals, remove from evaluation
            n_seg_eva = n_seg_eva - len(pred_sz)
            continue
        if isinstance(thd, list):
            y_one = pre_sz['prob'] >= thd[idx]
        else:
            y_one = pre_sz['prob'] >= thd
        sz_preds.append(y_one.any())
        # print(f"seizure {id}: n_seg_pre={len(pre_sz)}, n_seg_inter={len(pred_sz)-len(pre_sz)}, "
        #       f"n_detected_pre={np.sum(pre_sz['prob'] > thd)},  sz_detected={any(pre_sz['prob'] > thd)},  "
        #       f"detect_time_second={np.max(sec_to_sz[y_one])}")
    sens = np.mean(sz_preds)
    y_pred = get_label_from_thd(pred_pd, thd)
    tiw = np.mean(y_pred)
    pp = sens * (1 - tiw)
    return pp, sens, tiw, len(sz_preds), n_seg_eva



def evaluate_result_system_level(pat, pre_itvl, pred_pd, thd, y_pred):
    anno = pd.read_csv(INFO_PATH / f'annotation/{pat}_Annots.csv')
    ph_min = pre_itvl + 1

    sz_ids = pred_pd['sz_id'].unique()
    sz_preds, detect_time = [], []
    fp, fw = 0, 0
    for id in sz_ids:
        pred_sz = pred_pd.loc[pred_pd['sz_id'] == id].reset_index(drop=True)
        sz_epoch = anno.loc[anno['sz_id'] == id, 'sz_epoch'].values[0]
        sec_to_sz = sz_epoch - pred_sz['seg_epoch'].to_numpy()
        sec_to_sz_pre = sec_to_sz[pred_sz['label']==1]

        pre_sz = pred_sz.loc[pred_sz['label']==1].reset_index(drop=True)
        if len(pre_sz) == 0:
            # some seizures do not have preictals
            continue

        # specificity related
        # false postive
        y_fp = (pred_sz['label'] == 0) & (pred_sz['prob'] >= thd)
        fp += np.sum(pred_sz.loc[y_fp, 'prob'].notnull())
        # false warning with PH= pre_itvl + pred_itvl (1)

        # ph_sec =  * 60
        sec_to_sz_fp = sec_to_sz[y_fp]
        fw_flag = np.repeat(True, len(sec_to_sz_fp))
        for idx, sec in enumerate(sec_to_sz_fp):
            if fw_flag[idx] == False:
                continue
            warnings_end = sec - ph_min * 60
            fw_flag[(idx+1):][np.where(sec_to_sz_fp[(idx+1):] > warnings_end)[0]] = False
        fw += np.sum(fw_flag)

        y_one = pre_sz['prob'] >= thd
        if y_one.any():
            detect_time.append(np.max(sec_to_sz_pre[y_one]))
        sz_preds.append(y_one.any())
        # print(f"seizure {id}: n_seg_pre={len(pre_sz)}, n_seg_inter={len(pred_sz)-len(pre_sz)}, "
        #       f"n_detected_pre={np.sum(pre_sz['prob'] > thd)},  sz_detected={any(pre_sz['prob'] > thd)},  "
        #       f"detect_time_second={np.max(sec_to_sz[y_one])}")
    sens = np.mean(sz_preds)
    tiw = y_pred.mean()
    pp = sens * (1 - tiw)
    duration_hour = len(y_pred) * 10 / (60 * 60)
    n_alarm = fw + any(sz_preds)
    wr =  n_alarm / duration_hour # warning rate per hour
    fwr = fw / duration_hour  # False Warning Rate per hour
    # fpr = fp / duration_hour  # False Positive Rate per hour

    print(fw, duration_hour)
    print(len(sz_preds), ph_min, n_alarm, duration_hour, wr, fwr)
    return pp, sens, tiw, np.mean(detect_time)


def get_thd_max_pp(model_id, inter_itvl=1440, type='val', smooth_min=0):
    mid, rid = model_id['model_id'], model_id['round_id']
    # model_cfg = get_saved_info(model_id, 'cfg')
    if type in ['val', 'tst']:
        if type == 'val':
            # only use interictal_interval of 1440 for threshold computation
            f_pred = RST_PATH / f"detail/model{mid}_round{rid}_pred_1440_val.csv"
        else:
            f_pred = RST_PATH / f"detail/model{mid}_round{rid}_pred_{inter_itvl}_tst.csv"
        assert f_pred.exists(), f"ERROR: {f_pred.name} doesn't exist."
        pred_pd = pd.read_csv(f_pred, index_col=False)
        if smooth_min > 0:
            pred_pd['prediction'] = smooth_logit_per_sz(pred_pd, window_length=smooth_min * 5)

        best_thd = get_threshold_one(pred_pd)
    elif type == 'dynamic':
        pred_val = pd.read_csv(RST_PATH / f"detail/model{mid}_round{rid}_pred_1440_val.csv")
        val_base = pred_val.copy()
        pred_tst = pd.read_csv(RST_PATH / f"detail/model{mid}_round{rid}_pred_{inter_itvl}_tst.csv")
        if smooth_min > 0:
            val_base['prediction'] = smooth_logit_per_sz(val_base, window_length=smooth_min * 5)
            pred_tst['prediction'] = smooth_logit_per_sz(pred_tst, window_length=smooth_min * 5)

        ids_tst = np.sort(pred_tst['sz_id'].unique())
        best_thd = []
        for idx in range(len(ids_tst)):
            if idx == 0:
                pred_base = val_base.copy()
            else:
                id_sel = ids_tst[:idx]
                tst_base = pred_tst.loc[pred_tst['sz_id'].isin(id_sel)].reset_index(drop=True).copy()
                pred_base = pd.concat([val_base, tst_base], axis=0)
            best_thd.append(get_threshold_one(pred_base))
    else:
        raise ValueError(f"ERROR: unknow type of threshold: {type} ")

    return best_thd


def get_threshold_one(pred_pd):
    max_pp = 0
    best_thd = 0
    for thd in np.arange(0.05, 1, 0.025):
        pp, sens, tiw, _, _ = evaluate_result_seizure_level(pred_pd, thd)
        if pp > max_pp:
            best_thd = thd
            max_pp = pp
    return best_thd


def average_pred_general(pred_data, win_length):
    pred_data.sort_values(by='seg_epoch', inplace=True)
    pred_data.reset_index(drop=True, inplace=True)
    pred_data['prediction'] = pred_data['prediction'].transform(
        lambda x: x.rolling(window=win_length, min_periods=1).mean())
    return pred_data


def average_pred_val_two(pred_data, win_length, model_id):
    pred_data = average_pred_validation(pred_data, win_length, model_id)
    pred_data.drop(columns=['prediction'], inplace=True)
    return pred_data


def average_pred_validation(pred_data, win_length, model_id):
    # Sort pred_data
    pred_data = pred_data.sort_values(by='seg_epoch').reset_index(drop=True)

    # Load validation predictions
    f_val = RST_PATH / f"detail/model{model_id['model_id']}_round{model_id['round_id']}_pred_1440_val.csv"
    pred_val = pd.read_csv(f_val).sort_values(by='seg_epoch')

    # Initialize numpy arrays for preictal and interictal prediction averages
    pre_avg = np.zeros(len(pred_data))
    inter_avg = np.zeros(len(pred_data))
    # Obtain numpy arrays for predictions
    tst_predictions = pred_data['prediction'].values
    # Obtain preictal and interictal predictions as numpy arrays, padded to ensure at least win_length elements
    pre_predictions = np.pad(pred_val[pred_val['label'] == 0]['prediction'].values[-win_length:],
                             (max(0, win_length - len(pred_val[pred_val['label'] == 0])), 0),
                             mode='constant', constant_values=np.nan)
    inter_predictions = np.pad(pred_val[pred_val['label'] == 1]['prediction'].values[-win_length:],
                               (max(0, win_length - len(pred_val[pred_val['label'] == 1])), 0),
                               mode='constant', constant_values=np.nan)

    # Compute rolling mean for preictal and interictal predictions
    for i in range(len(tst_predictions)):
        pre_avg[i] = np.nanmean(np.append(tst_predictions[i], pre_predictions))
        inter_avg[i] = np.nanmean(np.append(tst_predictions[i], inter_predictions))

    # Assign computed averages back to pred_data
    pred_data['prediction_pre'] = pre_avg
    pred_data['prediction_inter'] = inter_avg

    return pred_data


def average_pred_groundtruth(pred_data, win_length):
    pred_data.sort_values(by='seg_epoch', inplace=True)
    pred_data.reset_index(drop=True, inplace=True)
    pred_data['prediction'] = pred_data.groupby('label')['prediction'].transform(
        lambda x: x.rolling(window=win_length, min_periods=1).mean())
    return pred_data


def average_pred_test(pred_data, win_length, data_type, model_id):
    pred_data = pred_data.sort_values(by='seg_epoch').reset_index(drop=True)
    if data_type == 'tst':
        f_val = RST_PATH / f"detail/model{model_id['model_id']}_round{model_id['round_id']}_pred_1440_val.csv"
        pred_base = pd.read_csv(f_val).sort_values(by='seg_epoch')
    else:
        pred_base = pd.DataFrame()

    preds = pred_data['prediction'].to_numpy()
    labels = pred_data['label'].to_numpy()
    pre_avg, inter_avg = [], []

    # Concatenate base predictions if applicable
    if data_type == 'tst':
        base_preds = pred_base['prediction'].to_numpy()
        base_labels = pred_base['label'].to_numpy()
    else:
        base_preds = np.array([])
        base_labels = np.array([])

    for i in range(len(preds)):
        # Concatenate relevant predictions from base and pred_data up to current index
        if i == 0 and data_type == 'tst':
            combined_preds = np.concatenate((base_preds, [preds[i]]))
            combined_labels = np.concatenate((base_labels, [labels[i]]))
        else:
            combined_preds = np.concatenate((base_preds, preds[:i + 1]))
            combined_labels = np.concatenate((base_labels, labels[:i + 1]))

        # Filter predictions by label for pre and inter calculations
        combined_labels[-1] = 1
        pre_indices = np.where(combined_labels == 1)[0]
        combined_labels[-1] = 0
        inter_indices = np.where(combined_labels == 0)[0]

        pre_values = combined_preds[pre_indices][-win_length:]  # Take last win_length elements
        inter_values = combined_preds[inter_indices][-win_length:]  # Take last win_length elements

        # Calculate and append averages
        pre_avg.append(np.mean(pre_values) if pre_values.size > 0 else np.nan)
        inter_avg.append(np.mean(inter_values) if inter_values.size > 0 else np.nan)

    # Assigning calculated averages back to pred_data
    pred_data['prediction_pre'] = pre_avg
    pred_data['prediction_inter'] = inter_avg

    return pred_data




def average_pred_labelwise(pred_data, win_length, data_type, model_id):
    pred_data.sort_values(by='seg_epoch', inplace=True)
    pred_data.reset_index(drop=True, inplace=True)

    if data_type == 'tst':
        thd = get_thd_max_pp(model_id, type='val')
        pred_data['avg_label'] = (sigmoid(pred_data['prediction']) > thd).astype(int)
        avg_preds = pred_data['prediction'].copy()  # Use copy to avoid modifying original data

        for label in pred_data['avg_label'].unique():
            label_indices = pred_data[pred_data['avg_label'] == label].index

            for i in label_indices:
                start = max(0, i - win_length + 1)
                valid_range = pred_data.iloc[start:i + 1]
                matching_range = valid_range[valid_range['label'] == label]

                # Check if matching_range is not empty to compute the mean
                if not matching_range.empty:
                    mean_val = matching_range['prediction'].mean()
                    # If mean_val is NaN, retain the original prediction; otherwise, assign the mean_val
                    avg_preds.iloc[i] = mean_val if not np.isnan(mean_val) else pred_data['prediction'].iloc[i]
                else:
                    # If no matching rows, retain the original prediction
                    avg_preds.iloc[i] = pred_data['prediction'].iloc[i]
        pred_data['prediction'] = avg_preds
    else:
        pred_data['prediction'] = pred_data.groupby('label')['prediction'].transform(
            lambda x: x.rolling(window=win_length, min_periods=1).mean())
    return pred_data


def smooth_logit_per_sz(pred_df, window_length):
    if window_length <= 0:
        return pred_df['prediction'].to_numpy()

    smoothed_logit = pred_df.groupby('sz_id')['prediction'].transform(
        lambda x: x.rolling(window=window_length, min_periods=1, center=True).mean())
    return smoothed_logit


def get_label_from_thd(pred_pd, thd):
    if isinstance(thd, list):
        ids = np.sort(pred_pd['sz_id'].unique())
        y_pred = []
        for idx, sz in enumerate(ids):
            pred_one = pred_pd.loc[pred_pd['sz_id'] == sz]
            y_pred.extend((pred_one['prob'] >= thd[idx]).astype(int))
    else:
        y_pred = (pred_pd['prob'] >= thd).astype(int)
    return y_pred


def get_opt_thd(pat_id, preictal_itvl):
    assert preictal_itvl == 45, 'ERROR: only support preictal interval of 45 now.'
    row = FINAL_INFO[FINAL_INFO['pat_id'] == pat_id]
    if row.empty:
        raise ValueError(f"pat_id {pat_id} not found in the dataframe.")
    return row['thd45_opt'].values[0]


def load_tst_coh(pat_id):
    pat = PAT_LIST_ALL[pat_id - 1]
    # Search for the correct file
    coh_dir = WORK_PATH / f"data/{pat}"
    coh_file = None
    for file in os.listdir(coh_dir):
        if "_non-overlap_non-overlap_coh-0-170-stack_tst.pkl" in file:
            coh_file = coh_dir / file
            break
    if coh_file is None:
        raise FileNotFoundError("Coherence file not found in the specified directory.")

    with open(coh_file, 'rb') as f:
        print(f"loading file {coh_file.name}...")
        coh = pickle.load(f)
    return coh


def get_segment_steps(pat_id, seg_epoch):
    pat = PAT_LIST_ALL[pat_id-1]
    file_path = INFO_PATH / f"segment_list/{pat}/{pat}_segment_list_10_-1_46_1_non-overlap_non-overlap_10.csv"
    df = pd.read_csv(file_path)

    # Find the row where seg_epoch matches the target
    target_row = df[df['seg_epoch'] == seg_epoch]

    # Check if a matching row was found
    if target_row.empty:
        print(f"No segment found for seg_epoch: {seg_epoch}")
        return None, None

    # Extract seg_start_step and seg_end_step
    seg_start_step = target_row['seg_start_step'].values[0]
    seg_end_step = target_row['seg_end_step'].values[0]

    return int(seg_start_step), int(seg_end_step)


def get_final_model_id(pat_id, preictal_itvl):
    model_col = f'model{preictal_itvl}'
    if model_col not in FINAL_INFO.columns:
        raise ValueError(f"Preictal interval {preictal_itvl} is not valid.")
    row = FINAL_INFO[FINAL_INFO['pat_id'] == pat_id]
    if row.empty:
        raise ValueError(f"pat_id {pat_id} not found in the dataframe.")
    return row[model_col].values[0]


def get_pat_from_pid(pat_id):
    """
    Retrieves the value of the `pat` column for a given `pat_id`.
    """

    # Filter the DataFrame for the given pat_id
    row = FINAL_INFO[FINAL_INFO["pat_id"] == pat_id]
    if not row.empty:
        return row["pat"].iloc[0]  # Return the first match
    else:
        print(f"No match found for pat_id: {pat_id}")
        return None


def get_pid_from_pat(patient):
    """
    Retrieves the value of the `pat_id` column for a given `pat`.
    """
    # Filter the DataFrame for the given pat_id
    row = FINAL_INFO[FINAL_INFO["pat"] == patient]
    if not row.empty:
        return row["pat_id"].iloc[0]
    else:
        print(f"No match found for patient: {patient}")
        return None

def get_anno_df(pid):
    f_anno = INFO_PATH / f"annotation/Pat{pid}_Annots.csv"
    assert f_anno.exists(), f"ERROR: {f_anno.name} doesn't exist."
    anno_df = pd.read_csv(f_anno, index_col=False)
    return anno_df


def calculate_perf_metrics(y_true, prediction, prediction_type, thd=0.5):
    """
    Calculates and returns various performance metrics for a classification model.

    Args:
        prediction_type:
    """
    if prediction_type == 'prob':
        assert (thd >= 0 ) and (thd <=1), f"ERROR, invalid threshold={thd}."
        y_pred = prediction > thd
        roc_auc = roc_auc_score(y_true, prediction) if len(set(y_true)) > 1 else None
    elif prediction_type == 'label':
        roc_auc = None
        y_pred = prediction

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)  # tp/(tp+fp)
    recall = recall_score(y_true, y_pred, zero_division=0) # sensitivity, tp/(tp+fn)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None # true negative rate, specificity, tn/(tn+fp)
    seg_tiw = (tp + fp) / len(y_true)
    seg_pp = recall * (1 - seg_tiw)
    metrics = {
        "ROC AUC": roc_auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "TNR": tnr,
        'Seg_TiW': seg_tiw,
        'Seg_PP': seg_pp,
    }

    # Return all metrics
    return metrics
