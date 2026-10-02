import os
import shutil
import pandas as pd
from torch.utils.data import DataLoader
from config import CONFIG, DEVICE, f_log, TEST, RST_PATH, MODEL_PATH, PAT_LIST, FINAL_INFO, PAT_LIST_ALL, \
    INFO_PATH, PRE_ITVL_TESTED, MIDS_USED, THD_APPROACHES, PAT_ID_LIST, PAT_DICT
from util import my_print, save_result, get_model_ids, get_saved_info, print_dict, load_mat_to_np, copy_result, \
    sigmoid, evaluate_result_seizure_level, get_thd_max_pp, get_label_from_thd, smooth_logit_per_sz, get_pat_from_pid, \
    get_final_model_id, get_anno_df, calculate_perf_metrics
import torch
import time
import numpy as np
from data import EEGDataset, generate_non_overlap_seg_list, update_label, split_segment_list
from models import get_model
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix



def test(trn_model_ids, model_type='CoSP', rank=None, data_type='tst', inter_itvl=1440, save=True):
    assert(data_type in ['val', 'tst', 'trn', 'ictal'])
    device = DEVICE if rank is None else rank
    trn_info = get_saved_info(trn_model_ids, 'trn')
    config = get_saved_info(trn_model_ids, 'cfg')
    if config is None:
        raise ValueError(f"ERROR: can not get config info from model={trn_model_ids}")
    print_dict(config)

    config['inter_sample'] = 'non-overlap'
    config['pre_sample'] = 'non-overlap'
    config['batch_size'] = 64
    pat = PAT_DICT[config['pat_id']]
    trn_mid = trn_model_ids['model_id']
    config['inter_itvl_min'] = inter_itvl # update for segment labelling
    config['test_inter_itvl'] = inter_itvl
    my_print(1,
             f"\n{pat}: Evaluating {model_type} model, "
             f"model_ids={trn_mid}-{trn_model_ids['round_id']}, data_type={data_type}, device={device}")
    model = get_model(model_type=model_type, model_id=trn_model_ids, config=config, device=rank)

    # Load data
    if data_type == 'ictal':
        seg_name = INFO_PATH / f"segment_list/{pat}/{pat}_segment_list_ictal_10_4000.csv"
        assert seg_name.exists()
        seg_list = pd.read_csv(seg_name)
    else:
        trn_list, val_list, tst_list = generate_non_overlap_seg_list(config)
        if data_type == 'trn': seg_list = trn_list
        if data_type == 'val': seg_list = val_list
        if data_type == 'tst': seg_list = tst_list
    my_print(1, f"{pat}: n_seg={len(seg_list)}, data_type={data_type}  #sz={len(seg_list['sz_id'].unique())}")

    # if CONFIG['test_with_config']:
    #     my_print(1, f"{pat}: test with the configuration: base_model={CONFIG['test_model']}, "
    #                 f"old_prev_itvl={config['pre_itvl_min']}, new_pre_itvl={CONFIG['pre_itvl_min']}")
    #     # support test with different preictal interval
    #     # update label according to pre_itvl_min
    #     n_pre_old = np.sum(seg_list['label'].values)
    #     seg_list = update_label(pat, seg_list)
    #     n_pre_new = np.sum(seg_list['label'].values)
    #     my_print(1, f"{pat}: Update the labels of test set: n_pre_old={n_pre_old}  n_pre_new={n_pre_new}")
    #
    #     # update pre_itvl_min and inter_itvl_min
    #     total_min_old = config['pre_itvl_min'] + config['inter_itvl_min']
    #     config['pre_itvl_min'] = CONFIG['pre_itvl_min']
    #     config['inter_itvl_min'] = total_min_old - config['pre_itvl_min']
    #     config['test_model'] = trn_mid
    #     save_model_ids = get_model_ids(config)
    #     # need to save to result.csv to avoid the model_id are allocate to other tasks when parallel runing
    #     if save: save_result(save_model_ids, trn_info, config=config)
    # else:
    save_model_ids = trn_model_ids
    # config['test_model'] = -1

    tst_dataset = EEGDataset(seg_list, data_type, config=config, phase='test', rank=rank)
    batch_size = config['batch_size']
    tst_loader = DataLoader(tst_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds = []
    true_labels = []
    seg_epochs = []
    n_segs_tst = 0
    n_batch = 0
    inference_time = 0
    with torch.no_grad():
        for batch_idx, (data, label, seg_epoch, sz_ids) in enumerate(tst_loader):
            if data.size(0) == 1:
                continue
            # data shape: [n_segments, n_steps, n_channels]
            if TEST and batch_idx >= 2: break

            n_segs_tst += data.shape[0]
            n_batch += 1
            label_one = label.cpu().numpy().tolist()
            true_labels.extend(label_one)
            seg_epochs.extend(seg_epoch.cpu().numpy().tolist())

            start_time = time.time()
            data, label = data.to(rank), label.to(rank)
            if data.ndim == 3:
                data = data.unsqueeze(1)
            output = model(data)
            preds.extend(output.flatten().cpu().numpy())
            end_time = time.time()

            # compute scores
            score_batch = torch.sigmoid(output).flatten().cpu().numpy()
            y_pred = np.zeros(len(score_batch)).astype(np.int32)
            y_pred[score_batch >= 0.5] = 1
            acc = np.sum(np.array(y_pred) == np.array(label_one)) / len(y_pred)
            my_print(2,
                     f"{pat}, trn_mid={trn_mid}, Batch [{batch_idx + 1}/{len(tst_loader)}], score: {np.mean(score_batch):.3f}, accuracy:{acc:.3f},"
                     f"   #preictal={torch.sum(label).item()}   #interictal={len(label)-torch.sum(label).item()}")
            inference_time += end_time - start_time
    if TEST:
        true_labels = np.random.choice([0, 1], size=len(true_labels))

    # evaluate and save
    pred_df = pd.DataFrame({'seg_epoch': seg_epochs, 'prediction': preds, 'label': true_labels})
    pred_df['seg_epoch'] = pred_df['seg_epoch'].astype(int)

    mid, rid = save_model_ids['model_id'],  save_model_ids['round_id']
    if data_type == 'ictal':
        f_pred = RST_PATH / f"detail/model{mid}_round{rid}_pred_ictal.csv"
    else:
        f_pred = RST_PATH / f"detail/model{mid}_round{rid}_pred_{config['test_inter_itvl']}_{data_type}.csv"
    info_df = seg_list
    info_df['seg_epoch'] = info_df['seg_epoch'].astype(pred_df['seg_epoch'].dtype)
    pred_full = pd.merge(pred_df, info_df[['seg_epoch', 'file_name', 'rec_start_epoch', 'sz_id']], on='seg_epoch',how='inner')

    # remove duplicate seg_epoch in prediction
    pred_full = pred_full.drop_duplicates(subset='seg_epoch', keep=False)
    # remove segments related to the seizures without preictals
    del_sz = pred_full.groupby('sz_id')['label'].sum().loc[lambda x: x == 0].index
    pred_full = pred_full.loc[~pred_full['sz_id'].isin(del_sz)].reset_index(drop=True)

    # save predictions
    pred_full.to_csv(f_pred, index=False)
    my_print(1, f"{pat}: {f_pred.name} has been saved to {f_pred.parent}.")

    # save evaluations only for test set
    if (data_type == 'tst') and save:
        # result evaluation
        eva_info = evaluate_result(save_model_ids, thd_from='tst', inter_itvl=inter_itvl)
        print_dict(eva_info)
        # save training and evaluation into result.csv
        avg_inference_time = inference_time * 1000 / n_segs_tst
        rst_dict = {'roc_auc': eva_info['roc_auc'], 'accuracy': eva_info['accuracy'], 'threshold': eva_info['threshold'],
                    'sensitivity': eva_info['sensitivity'], 'tiw': eva_info['tiw'], 'perf_prod': eva_info['pp']}
        tst_info = {'infer_time_per_seg_ms': avg_inference_time, 'n_segs_tst': n_segs_tst}
        my_print(1, f"result={rst_dict}")
        save_result(save_model_ids, config=config, rst_info=rst_dict, tst_info=tst_info)
        if os.path.isfile(f_log):
            shutil.move(f_log, MODEL_PATH / f"model{mid}_round{rid}_test.log")

        # optional
        # summarize_results(mids=[save_model_ids['model_id']], thd_from='tst', inter_itvl=inter_itvl, save=True)
        # gen_results_full()

    # free the GPU memory, avoid  Memory Fragmentation Error
    # torch.cuda.empty_cache()
    return


def evaluate_result(model_id, thd_from='val', thd_fixed=None, pre_itvl=45, inter_itvl=1440, pat_id=None, smooth_min=0):
    if model_id == 'random':
        random_predictor = True
        assert pat_id in PAT_ID_LIST, f"ERROR: unknown patient id={pat_id}"
        mid = FINAL_INFO.loc[FINAL_INFO['pat_id'] == pat_id, f'model{pre_itvl}'].values[0]
        rid = 1
        model_id = {'model_id': mid, 'round_id': rid}
    else:
        random_predictor = False
        mid, rid = model_id['model_id'], model_id['round_id']
    model_cfg = get_saved_info(model_id, 'cfg')
    pat_idx = model_cfg['pat_id']
    pat = PAT_DICT[pat_idx]
    # pat_idx = PAT_ID_LIST[PAT_LIST.index(pat)]
    pre_min = model_cfg['pre_itvl_min']
    print(f"\nEvaluate model {mid}-{rid}: threshold={thd_from}, pat_idx={pat_idx}, patient={pat}, "
          f"preictal_min={pre_min}, interictal_min={inter_itvl}, random_predictor={random_predictor}")

    if thd_from == 'fixed':
        assert(thd_fixed is not None)
        thd_fixed = thd_fixed
    else:
        thd_fixed = get_thd_max_pp(model_id, inter_itvl=inter_itvl, type=thd_from, smooth_min=smooth_min)
        print(f"threshold={thd_fixed}")

    f_pred = RST_PATH / f"detail/model{mid}_round{rid}_pred_{inter_itvl}_tst.csv"
    assert f_pred.exists(), f"ERROR: {f_pred.name} doesn't exist."
    pred_pd = pd.read_csv(f_pred, index_col=False)
    if smooth_min > 0:
        sm_logits = smooth_logit_per_sz(pred_pd, window_length=smooth_min * 5)
        pred_pd['prediction'] = sm_logits

    if random_predictor:
        my_print(1, "generate random predicted probabilites.")
        # Set the 'prob' column to random values between 0 and 1
        pred_pd['prob'] = np.random.rand(len(pred_pd))
        # Clip values to avoid log(0) issues
        epsilon = 1e-10
        pred_pd['prob'] = pred_pd['prob'].clip(epsilon, 1 - epsilon)
        pred_pd['prediction'] = np.log(pred_pd['prob'] / (1 - pred_pd['prob']))
    else:
        pred_pd['prob'] = sigmoid(pred_pd['prediction'].to_numpy())

    # segment-level metrics
    perf_metrics = calculate_perf_metrics(y_true=pred_pd['label'],
                                          prediction=pred_pd['prob'],
                                          prediction_type='prob',
                                          thd=0.5)


    # seizure-level metrics
    pp, sens, tiw, n_sz, n_seg_eva = evaluate_result_seizure_level(pred_pd.copy(), thd_fixed)
    thd_avg = thd_fixed if thd_from not in ['dynamic'] else np.mean(thd_fixed)
    metrics = {
        'pat_idx': PAT_LIST_ALL.index(pat) + 1,
        'patient': pat,
        'n_sz_eva': n_sz,
        'n_seg_eva': n_seg_eva, # some seizures do not have preictal, exclude from seizure-level evaluation
        'model_id': 'random' if random_predictor else  mid,
        'preictal_itvl': pre_min,
        'interictal_itvl': inter_itvl,
        'smooth_minutes': smooth_min,
        'thd_from': thd_from,
        'thd_avg': thd_avg,
        'threshold': thd_fixed,
        'roc_auc': perf_metrics['ROC AUC'],
        'accuracy': perf_metrics['Accuracy'],
        'precision': perf_metrics['Precision'],
        'recall': perf_metrics['Recall'],
        'f1': perf_metrics['F1 Score'],
        'specificity': perf_metrics['TNR'],
        'sensitivity': sens,
        'tiw': tiw,
        'pp': pp
    }
    return metrics


def summarize_results(mids=None, thd_from='all', pre_itvl=45, inter_itvl=1440, pat_id=None, save=True, smooth_min=0):
    assert thd_from in THD_APPROACHES + ['all'], f"ERROR: unknown thresholding approach={thd_from}"
    assert (pre_itvl in PRE_ITVL_TESTED), f"ERROR: unknown preictal interval={pre_itvl}"
    assert (inter_itvl in [-1, 1440]), f"ERROR: unknown interictal interval={inter_itvl}"

    thd_list = ['val', 'tst', 'dynamic', 'fixed']  if thd_from == 'all' else [thd_from]

    results = []
    for mid in mids:
        model_id = mid if mid == 'random' else {'model_id': mid, 'round_id': 1}
        for thd_from in thd_list:
            rst = evaluate_result(model_id, thd_from=thd_from, thd_fixed=0.5, pre_itvl=pre_itvl, inter_itvl=inter_itvl,
                                  pat_id=pat_id, smooth_min=smooth_min)
            print_dict(rst)
            results.append(rst)

    if save:
        # Convert the list of dictionaries to a DataFrame
        results_df = pd.DataFrame(results)
        save_to_summary_csv(results_df)
    return pd.DataFrame(results)


def save_to_summary_csv(results_df):
    summary_file = RST_PATH / 'result_summary.csv'
    if summary_file.exists():
        data = pd.read_csv(summary_file)
        data = pd.concat([data, results_df], ignore_index=True)
    else:
        data = results_df

    # Remove duplicate rows based on specific columns
    data = data.drop_duplicates(subset=[
        'pat_idx', 'patient', 'n_sz_eva', 'n_seg_eva', 'model_id', 'preictal_itvl', 'interictal_itvl', 'smooth_minutes', 'thd_from'])
    data.to_csv(summary_file, index=False)
    print(f'{summary_file} has been updated.')


def gen_results_full():
    # Load the CSV files into DataFrames
    df_sum = pd.read_csv(RST_PATH / 'result_summary.csv')
    df_rst = pd.read_csv(RST_PATH / 'result.csv')

    # Merge the DataFrames on the 'model_id' column
    # Using 'inner' join to keep only rows that have a model_id in both DataFrames
    df_sum['model_id'] = df_sum['model_id'].astype(str)
    df_rst['model_id'] = df_rst['model_id'].astype(str)
    combined_df = pd.merge(df_sum, df_rst, on='model_id', how='inner')

    # remove duplicated performance columns
    drop_cols = [i for i in combined_df.columns if i.startswith('RST_')]
    combined_df = combined_df.drop(columns=drop_cols)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(RST_PATH / 'results_full.csv', index=False)
    print(f"results_full.csv has been saved to {RST_PATH}")


def summarize_result_random_predictor(pat_id, pre_itvl=4, inter_itvl=1440, thd_from='tst', n_rounds=50, save=True):
    results = []
    for i in range(n_rounds):
        rst = summarize_results(mids=['random'], thd_from=thd_from, pre_itvl=pre_itvl, inter_itvl=inter_itvl,
                                pat_id=pat_id, save=False)
        results.append(rst)

    results_df = pd.DataFrame(results)
    col_mean = [
        'thd_avg',
        'roc_auc',
        'accuracy',
        'precision',
        'recall',
        'f1',
        'specificity',
        'sensitivity',
        'tiw',
        'pp'
    ]
    # Compute the average values of the specified columns
    avg_rst = results_df[col_mean].mean()
    last_row = results_df.iloc[-1]
    last_row[col_mean] = avg_rst
    avg_rst = last_row.to_frame().T
    print(avg_rst)
    if save: save_to_summary_csv(avg_rst)
    return avg_rst


def relabel_predictions(pat_id=None, pre_itvl_min=None, inter_itvl=1440, data_type=None, save=False):
    """
    Relabels the prediction DataFrame based on seizure annotations.
    """
    rid = 1
    pids = [pat_id] if pat_id else FINAL_INFO['pat_id'].to_numpy()
    pre_itvls = [pre_itvl_min] if pre_itvl_min else [int(i[5:]) for i in FINAL_INFO.columns if i.startswith('model')]
    types = [data_type] if data_type else ['tst', 'val']

    for pid in pids:
        anno_df = get_anno_df(pid)

        for pre_itvl in pre_itvls:
            for d_type in types:
                mid = get_final_model_id(pid, pre_itvl)
                f_pred = RST_PATH / f"detail/model{mid}_round{rid}_pred_{inter_itvl}_{d_type}.csv"
                if not f_pred.exists():
                    print(f"WARNING: {f_pred.name} doesn't exist.")
                    continue
                pred_df = pd.read_csv(f_pred, index_col=False)
                old_label = pred_df["label"].to_numpy()
                pred_df["label"] = 0
                for _, seizure in anno_df.iterrows():
                    pre_start = seizure["sz_epoch"] - (pre_itvl + CONFIG['pred_itvl_min']) * 60
                    pre_end = seizure["sz_epoch"]

                    # Relabel segments within the preictal window
                    pred_df.loc[
                        (pred_df["seg_epoch"] >= pre_start) & (pred_df["seg_epoch"] < pre_end),
                        "label"
                    ] = 1

                changed_labels = np.sum(old_label != pred_df["label"].to_numpy())
                print(f"pid={pid}, pre_itvl={pre_itvl}, type={d_type}: #different_labels={changed_labels}")

                if save and (changed_labels > 0):
                    pred_df.to_csv(f_pred, index=False)
                    print(f"{f_pred} has been updated.")
    return



def check_model_patient_mapping():
    """
    Checks if all model IDs in FINAL_INFO correspond to the same patient ID in result.csv.
    """
    # Load the FINAL_INFO and result.csv files
    f_rst = RST_PATH / 'result.csv'
    result_df = pd.read_csv(f_rst, index_col=False)

    model_names = [i for i in FINAL_INFO.columns if i.startswith("model")]

    # Create a mapping of model_id to pat_id from result.csv
    result_mapping = result_df[["model_id", "patient_id"]].drop_duplicates()

    # Check each model_id in FINAL_INFO
    mismatches = []
    for _, row in FINAL_INFO.iterrows():
        for col in model_names:
            model_id = row[col]
            pat_id_final_info = row["pat_id"]

            # Get the pat_id for the model_id in result.csv
            pat_id_result = result_mapping.loc[result_mapping["model_id"] == model_id, "patient_id"].values

            # Check for mismatches
            if len(pat_id_result) == 0:
                print(f"WARNING: model_id {model_id} not found in result.csv.")
            elif pat_id_final_info != pat_id_result[0]:
                mismatches.append((model_id, pat_id_final_info, pat_id_result[0]))

        # Print results
        if not mismatches:
            print("All model IDs in FINAL_INFO correspond to the correct patient IDs in result.csv.")
        else:
            print("The following mismatches were found:")
            for model_id, pat_id_final_info, pat_id_result in mismatches:
                print(f"model_id={model_id}: FINAL_INFO pat_id={pat_id_final_info}, result.csv pat_id={pat_id_result}")
