from collections import deque
import numpy as np

from config import RST_PATH, FINAL_INFO, INFO_PATH, CONFIG, PAT_ID_LIST
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from util import sigmoid, get_pat_from_pid, get_final_model_id, get_anno_df, print_dict, calculate_perf_metrics
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier


DEBUG = False


def warning_no_retrigger(group, anno_df, warn_len, pred_col, pred_type, thd, seg_sec):
    """
    Processes a single seizure group for warnings and computes metrics.
    No retriggering
    """
    in_warning = False
    warning_start = 0
    warning_end = 0
    n_tp_alarm = 0
    n_fp_alarm = 0
    time_in_warning = 0

    # compute interictal duration in seconds
    inter_epochs = group[group["label"] == 0]["seg_epoch"]
    inter_seconds = inter_epochs.max() - inter_epochs.min() + seg_sec
    test_seconds = group["seg_epoch"].max() - group["seg_epoch"].min() + seg_sec + warn_len * 60

    for _, row in group.iterrows():
        seg_time = row["seg_epoch"]
        trigger = False

        # Handle warning state expiration
        if in_warning and seg_time >= warning_end:
            in_warning = False
            time_in_warning += warning_end - warning_start

        # If not in warning state, check for new warnings
        if not in_warning:
            if pred_type == 'prob':
                if row[pred_col] > thd: trigger = True
            else:
                if row[pred_col] == 1: trigger = True

            if trigger:
                in_warning = True
                warning_start = seg_time
                warning_end = seg_time + warn_len * 60
                # Check if any seizures occur during the warning state
                seizures_during_warning = anno_df[
                    (anno_df["sz_epoch"] >= seg_time) & (anno_df["sz_epoch"] < warning_end)
                    ]

                if len(seizures_during_warning) > 0:
                    n_tp_alarm += 1
                    # Set warning_end to the time of seizure occurrence
                    warning_end = seizures_during_warning["sz_epoch"].min()
                else:
                    n_fp_alarm += 1

    # Add remaining time in warning state if active at the end of the group
    if in_warning:
        time_in_warning += warning_end - warning_start
    # Ensure time_in_warning does not exceed test_seconds
    time_in_warning = min(time_in_warning, test_seconds)

    rst = {
        "n_tp_alarm": n_tp_alarm,
        "n_fp_alarm": n_fp_alarm,
        "time_in_warning": time_in_warning,
        "inter_seconds": inter_seconds,
        "test_seconds": test_seconds,
    }
    # print("\n")
    # print(group['sz_id'].iloc[0])
    # print_dict(rst)
    return rst


def warning_retrigger(group, anno_df, warn_len, pred_col, pred_type, thd, seg_sec):
    """
    Processes a single seizure group for warnings and computes metrics with retriggering.
    """
    in_warning = False
    warning_start = 0
    warning_end = 0
    n_tp_alarm = 0
    n_fp_alarm = 0
    time_in_warning = 0
    sz_occure = False

    # Compute interictal and test durations
    inter_epochs = group[group["label"] == 0]["seg_epoch"]
    inter_seconds = inter_epochs.max() - inter_epochs.min() + seg_sec if len(inter_epochs) > 0 else 0
    # warning time may exceed the last prediction for warn_len * 60 seconds
    test_seconds = group["seg_epoch"].max() - group["seg_epoch"].min() + seg_sec + warn_len * 60

    for _, row in group.iterrows():
        seg_time = row["seg_epoch"]

        # Handle warning state expiration
        if in_warning and seg_time >= warning_end:
            in_warning = False

            # compute tiw no matter interictal or preictal
            time_in_warning += warning_end - warning_start

            # # compute tiw only in interictal
            # valid_warning_time = warning_end - warning_start
            # preictal_time = \
            # group[(group["seg_epoch"] >= warning_start) & (group["seg_epoch"] < warning_end) & (group["label"] == 1)][
            #     "seg_epoch"].count() * seg_sec
            # time_in_warning += max(0, valid_warning_time - preictal_time)

            if sz_occure:
                n_tp_alarm += 1
            else:
                n_fp_alarm += 1
            sz_occure = False

        # Check for a trigger
        trigger = (
                (pred_type == 'prob' and row[pred_col] > thd) or
                (pred_type == 'label' and row[pred_col] == 1)
        )
        if trigger:
            if not in_warning:
                in_warning = True
                warning_start = seg_time

            # Check if any seizures occur during the warning state
            seizures_during_warning = anno_df[
                (anno_df["sz_epoch"] >= seg_time) & (anno_df["sz_epoch"] < (seg_time + warn_len * 60))
                ]
            if len(seizures_during_warning) > 0:
                sz_occure = True
                warning_end = seizures_during_warning["sz_epoch"].min()
            else:
                warning_end = seg_time + warn_len * 60

    # Handle any active warning state after the group's end
    if in_warning:
        # Warning end may extend beyond the group's end
        time_in_warning += warning_end - warning_start
        if sz_occure:
            n_tp_alarm += 1
        else:
            n_fp_alarm += 1

    # Ensure time_in_warning does not exceed test_seconds
    time_in_warning = min(time_in_warning, test_seconds)
    rst = {
        "n_tp_alarm": n_tp_alarm,
        "n_fp_alarm": n_fp_alarm,
        "time_in_warning": time_in_warning,
        "inter_seconds": inter_seconds,
        "test_seconds": test_seconds,
    }
    if DEBUG:
        print_dict(rst)
        print('\n')

    return rst


def seizure_warning_parallel(pid, pred_df, warn_len, pred_col, pred_type, thd=0.5, seg_sec=10, retrigger=False):
    """
    Implements a seizure warning system based on moving average predictions using parallel computation.

    Args:
        pred_type:
    """
    assert pred_col in pred_df.columns, f"ERROR: unknown column name {pred_col}"
    anno_df = get_anno_df(pid)

    # Group the DataFrame by seizure ID (sz_id)
    grouped = pred_df.groupby("sz_id")

    # Process each group in parallel using joblib
    if retrigger:
        results = Parallel(n_jobs=-1)(
            delayed(warning_retrigger)(group, anno_df, warn_len, pred_col, pred_type, thd, seg_sec)
            for _, group in grouped
        )
    else:
        results = Parallel(n_jobs=-1)(
            delayed(warning_no_retrigger)(group, anno_df, warn_len, pred_col, pred_type, thd, seg_sec)
            for _, group in grouped
        )

    # Aggregate results
    total_tp_alarm = sum(r["n_tp_alarm"] for r in results)
    total_fp_alarm = sum(r["n_fp_alarm"] for r in results)
    total_time_in_warning = sum(r["time_in_warning"] for r in results)
    total_inter_seconds = sum(r["inter_seconds"] for r in results)
    total_test_seconds = sum(r["test_seconds"] for r in results)
    total_seizures = len(np.unique(pred_df["sz_id"].to_numpy()))

    if DEBUG:
        print("============================================")
        print("total_tp_alarm = ", total_tp_alarm)
        print("total_fp_alarm = ", total_fp_alarm)
        print("total_time_in_warning = ", total_time_in_warning)
        print("total_inter_seconds = ", total_inter_seconds)
        print("total_test_seconds = ", total_test_seconds)
        print("total_seizures = ", total_seizures)
        print("============================================\n")


    # Compute performance metrics
    seizure_sensitivity = total_tp_alarm / total_seizures if total_seizures > 0 else 0
    fpr_per_hour = total_fp_alarm / (total_test_seconds / 3600) if total_test_seconds > 0 else 0
    tiw = total_time_in_warning / total_test_seconds if total_test_seconds > 0 else 0
    # tiw = total_time_in_warning / total_inter_seconds if total_inter_seconds > 0 else 0

    return {
        "SS": seizure_sensitivity,
        "FPR/h": fpr_per_hour,
        "Seizure_TiW": tiw,
        "Seizure_PP": seizure_sensitivity * (1 - tiw),
    }

#
# def seizure_warning_system(pid, pred_df, warn_len, prob_col, thd=0.5):
#     """
#     Implements a seizure warning system based on moving average predictions.
#     """
#     assert prob_col in pred_df.columns, f"ERROR: unknown column name {prob_col}"
#     anno_df = get_anno_df(pid)
#
#     # Initialize variables
#     in_warning = False
#     warning_end_time = 0
#     n_tp_alarm = 0
#     n_fp_alarm = 0
#     time_in_warning = 0
#     total_seizures = len(np.unique(pred_df['sz_id'].to_numpy()))
#
#     # Iterate through prediction rows
#     for _, row in pred_df.iterrows():
#         seg_time = row["seg_epoch"]
#
#         # Handle warning state expiration
#         if in_warning and seg_time >= warning_end_time:
#             in_warning = False
#
#         # If not in warning state, check for new warnings
#         if not in_warning and row[prob_col] > thd:
#             in_warning = True
#             warning_end_time = seg_time + warn_len * 60  # Convert pre_itvl to seconds
#
#             # Check if any seizures occur during the warning state
#             seizures_during_warning = anno_df[
#                 (anno_df["sz_epoch"] >= seg_time) & (anno_df["sz_epoch"] < warning_end_time)
#                 ]
#
#             if len(seizures_during_warning) > 0:
#                 n_tp_alarm += 1
#             else:
#                 n_fp_alarm += 1
#
#         # Count time spent in warning state
#         if in_warning and row["label"] == 0:  # Interictal time only
#             time_in_warning += 10  # Each segment represents 10 seconds
#
#     # Compute performance metrics
#     inter_seconds = np.sum(pred_df["label"] == 0) * 10  # Total interictal time in seconds
#     seizure_sensitivity = n_tp_alarm / total_seizures if total_seizures > 0 else 0
#     fpr_per_hour = n_fp_alarm / (inter_seconds / 3600) if inter_seconds > 0 else 0
#     tiw = time_in_warning / inter_seconds if inter_seconds > 0 else 0
#
#     return {
#         "SS": seizure_sensitivity,
#         "FPR/h": fpr_per_hour,
#         "Seizure_TiW": tiw,
#         "Seizure_PP": seizure_sensitivity * (1- tiw),
#     }




def plot_prediction_distribution(predictions, labels, show_prob=False, title=None):
    if show_prob:
        predictions = sigmoid(predictions.to_numpy())
    all_segments = predictions
    interictal_segments = predictions[labels == 0]  # label == 0
    preictal_segments = predictions[labels == 1]  # label == 1

    # Plot the distributions
    plt.figure(figsize=(12, 8))

    # Plot for all segments
    # sns.histplot(all_segments, bins=50, kde=True, color='blue', label="All", stat="density")
    # Plot for interictal segments
    sns.histplot(interictal_segments, bins=50, kde=True, color='green', label="Interictal",
                 stat="density")
    # Plot for preictal segments
    sns.histplot(preictal_segments, bins=50, kde=True, color='red', label="Preictal", stat="density")

    # Add titles and labels
    if title is None:
        pred_type = 'Probabilities' if show_prob else 'Logits'
        title = f"Predicted {pred_type} Distribution"
    plt.title(title, fontsize=16)
    plt.xlabel("Prediction Logits", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()


def filter_predictions(pred_df, pre_itvl, pid, threshold=None):
    """
    Filters seizures based on the number of interictal and preictal segments.

    Args:
        pred_df (pd.DataFrame): DataFrame containing the prediction data, including 'sz_id' and 'label'.
        pre_itvl (int): Interval parameter used to calculate the filtering threshold.
        pid (int): Patient ID for logging purposes.
        threshold (float, optional): Minimum count of segments for filtering.
                                     Defaults to pre_itvl * 5 / 2.

    Returns:
        pd.DataFrame: Filtered DataFrame sorted by `seg_epoch`, containing only seizures that meet the threshold.
    """
    # Default threshold: each minute has 5 segments, set minimum to at least half
    if threshold is None:
        threshold = pre_itvl * 5 / 2

    # Group by `sz_id` and count interictal (label=0) and preictal (label=1) segments
    result = pred_df.groupby("sz_id")["label"].value_counts().unstack(fill_value=0)
    result.columns = ["Interictal", "Preictal"]

    # Filter seizures meeting the threshold for both interictal and preictal segments
    filtered_result = result[(result["Interictal"] >= threshold) & (result["Preictal"] >= threshold)]

    # Log removed seizures
    removed_seizures_count = len(result) - len(filtered_result)
    # print(
    #     f"P{pid}, pre_itvl={pre_itvl}: remove {removed_seizures_count} seizure(s) due to too few segments, {len(filtered_result)} seizure(s) are left.")

    # Filter original DataFrame to include only retained `sz_id` and sort by `seg_epoch`
    filtered_ids = filtered_result.index
    filtered_data = pred_df[pred_df["sz_id"].isin(filtered_ids)].sort_values(by="seg_epoch")

    return filtered_data


def moving_average(pred_df, window_min, col_name):
    """
    Computes a moving average for the predictions in pred_df with the specified window size in minutes on column col_name.
    """
    assert col_name in pred_df.columns, f"ERROR: Unknown target column name {col_name} in moving average."
    n_seg_window = window_min * 5  # Convert minutes to number of segments

    # Sort the DataFrame by seizure ID and segment time
    pred_df = pred_df.sort_values(by=["sz_id", "seg_epoch"]).reset_index(drop=True)
    ma_logits_col = f"{col_name}_ma{window_min}m_logits"
    pred_df[ma_logits_col] = np.nan

    # Process each seizure independently
    for sz_id, group in pred_df.groupby("sz_id"):
        # group = group.reset_index(drop=True)
        earliest_time = group["seg_epoch"].iloc[0]
        window_start_time = earliest_time + (window_min * 60)

        # Compute initial average between earliest_time and window_start_time
        initial_group = group[group["seg_epoch"] <= window_start_time]
        current_avg = initial_group[col_name].mean() if not initial_group.empty else 0

        moving_avg_list = []
        for i, row in group.iterrows():
            if row["seg_epoch"] <= window_start_time:
                # For rows before window_start_time, no moving average
                moving_avg_list.append(np.nan)
                continue

            start_epoch = row["seg_epoch"] - (window_min * 60)
            end_epoch = row["seg_epoch"]

            # Filter rows in the current window
            group_one = group[(group["seg_epoch"] > start_epoch) & (group["seg_epoch"] <= end_epoch)]
            pred_one = group_one[col_name].to_numpy()

            # Handle gaps and incomplete windows
            if len(pred_one) < n_seg_window:
                pred_one = np.concatenate([pred_one, [current_avg] * (n_seg_window - len(pred_one))])

            # Compute the moving average
            current_avg = np.mean(pred_one)
            moving_avg_list.append(current_avg)

        # Assign computed moving averages back to the main DataFrame
        pred_df.loc[group.index, ma_logits_col] = moving_avg_list

    pred_df.dropna(subset=[ma_logits_col], inplace=True)
    # print(f"Moving average with window length {window_min}m have been saved to {ma_logits_col} column.")
    ma_prob_col = ma_logits_col.replace('logits', 'prob')
    pred_df[ma_prob_col] = sigmoid(pred_df[ma_logits_col].to_numpy())
    # print(f"The probability of moving average have been saved to {ma_prob_col} column.")

    return pred_df, ma_logits_col, ma_prob_col


def compute_moving_average_for_group(group, window_min, col_name, ma_col):
    """
    Computes the moving average for a single group (seizure).
    """
    n_seg_window = window_min * 5  # Convert minutes to number of segments
    earliest_time = group["seg_epoch"].iloc[0]
    window_start_time = earliest_time + (window_min * 60)

    # Compute initial average between earliest_time and window_start_time
    initial_group = group[group["seg_epoch"] <= window_start_time]
    current_avg = initial_group[col_name].mean() if not initial_group.empty else 0

    moving_avg_list = []
    for i, row in group.iterrows():
        if row["seg_epoch"] <= window_start_time:
            # For rows before window_start_time, no moving average
            moving_avg_list.append(np.nan)
            continue

        start_epoch = row["seg_epoch"] - (window_min * 60)
        end_epoch = row["seg_epoch"]

        # Filter rows in the current window
        group_one = group[(group["seg_epoch"] > start_epoch) & (group["seg_epoch"] <= end_epoch)]
        pred_one = group_one[col_name].to_numpy()

        # Handle gaps and incomplete windows
        if len(pred_one) < n_seg_window:
            pred_one = np.concatenate([pred_one, [current_avg] * (n_seg_window - len(pred_one))])

        # Compute the moving average
        current_avg = np.mean(pred_one)
        moving_avg_list.append(current_avg)

    # Add the results as a new column in the group
    group[ma_col] = moving_avg_list
    return group

def moving_average_parallel(pred_df, window_min, col_name):
    """
    Computes a moving average for the predictions in pred_df with the specified window size in minutes on column col_name.
    """
    assert col_name in pred_df.columns, f"ERROR: Unknown target column name {col_name} in moving average."

    # Sort the DataFrame by seizure ID and segment time
    pred_df = pred_df.sort_values(by=["sz_id", "seg_epoch"]).reset_index(drop=True)
    ma_logits_col = f"ma_pi_logits"
    ma_prob_col = f"ma_pi_prob"

    # Use joblib to parallelize the computation across groups
    grouped = pred_df.groupby("sz_id")
    results = Parallel(n_jobs=-1)(
        delayed(compute_moving_average_for_group)(group, window_min, col_name, ma_logits_col) for _, group in grouped
    )

    # Combine results back into a single DataFrame
    pred_df = pd.concat(results, ignore_index=True)

    # Drop rows without computed moving average
    pred_df.dropna(subset=[ma_logits_col], inplace=True)

    # Compute probabilities from the moving average logits
    pred_df[ma_prob_col] = sigmoid(pred_df[ma_logits_col].to_numpy())

    return pred_df, ma_logits_col, ma_prob_col


def evaluate_both_level(pid, pred_df, pre_itvl, pred_col, pred_type='prob', sz_level_thd=0.5, seg_sec=10,
                        retrigger=True):
    """
    Evaluates the model performance by calculating segment-level metrics and seizure warning metrics.

    Args:
        pred_type:
    """
    # Ensure the required column exists in the DataFrame
    assert pred_col in pred_df.columns, f"ERROR: Column '{pred_col}' not found in the DataFrame."

    # Initialize performance metrics dictionary
    perf_metrics = {'pid': pid, 'pre_itval': pre_itvl, 'thd': round(sz_level_thd,4), 'method': pred_col}
    # Segment-level performance metrics
    segment_metrics = calculate_perf_metrics(pred_df['label'], pred_df[pred_col], pred_type, 0.5)
    perf_metrics.update(segment_metrics)

    # Seizure warning system metrics
    warning_metrics = seizure_warning_parallel(pid, pred_df, pre_itvl, pred_col, pred_type, thd=sz_level_thd,
                                               seg_sec=seg_sec, retrigger=retrigger)
    perf_metrics.update(warning_metrics)
    perf_metrics['retrigger'] = retrigger
    return perf_metrics


def prepare_minute_data(pred_df, prob_col):
    """
    Prepares 1-minute data points from 10-second predictions.
    """

    # Ensure required columns exist
    required_columns = {"seg_epoch", "prob_10s", prob_col, "label", "sz_id"}
    assert required_columns.issubset(pred_df.columns), f"ERROR: Missing required columns in DataFrame."

    # Group by minute (truncate seg_epoch to the nearest minute)
    pred_df["seg_epoch"] = (pred_df["seg_epoch"] // 60) * 60
    grouped = pred_df.groupby("seg_epoch")

    minute_data = []

    for minute, group in grouped:
        # Ensure exactly 5 predictions for the minute
        if len(group) < 5:
            # Fill gaps with the average values of the current minute
            avg_prob_10s = group["prob_10s"].mean() if not group["prob_10s"].isna().all() else 0
            avg_ma_prob = group[prob_col].mean() if not group[prob_col].isna().all() else 0

            while len(group) < 5:
                # Add a row with average values
                group = pd.concat(
                    [group, pd.DataFrame({"prob_10s": [avg_prob_10s], prob_col: [avg_ma_prob]})],
                    ignore_index=True,
                )

        # Extract features
        prob_10s_features = group["prob_10s"].head(5).to_list()
        ma_prob_features = group[prob_col].head(5).to_list()
        features = prob_10s_features + ma_prob_features

        # Get the label for the minute (assume the majority label for the minute)
        label = group["label"].mode()[0] if not group["label"].isna().all() else 0

        # Append to minute data
        minute_data.append({"features": features,
                            "label": label,
                            "sz_id": group['sz_id'].iloc[0],
                            "seg_epoch": minute})

    # Convert to DataFrame
    minute_df = pd.DataFrame(minute_data)
    feature_columns = [f"feature_{i}" for i in range(10)]
    for i, col in enumerate(feature_columns):
        minute_df[col] = minute_df["features"].apply(lambda x: x[i])
    minute_df = minute_df.drop(columns=["features"])

    return minute_df


def build_and_evaluate_model(minute_df):
    """
    Builds and evaluates a Random Forest model for 1-minute data points.

    Args:
        minute_df (pd.DataFrame): DataFrame with features and labels.

    Returns:
        None
    """
    # Separate features and target
    X = minute_df.drop(columns=["label"])
    y = minute_df["label"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # X_train, X_test, y_train, y_test = X, y, X, y

    # Build Random Forest with class weights to handle imbalance
    rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = rf_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def prepare_prediction(pid, pre_itvl, inter_itvl=1440, data_type='tst', mid=None):
    if mid is None:
         mid = get_final_model_id(pid, pre_itvl)

    f_pred = RST_PATH / f"detail/model{mid['model_id']}_round{mid['round_id']}_pred_{inter_itvl}_{data_type}.csv"
    assert f_pred.exists(), f"ERROR: {f_pred.name} doesn't exist."
    pred_df = pd.read_csv(f_pred, index_col=False)
    # Rename the 'prediction' column to 'logits10s'
    pred_df = pred_df.rename(columns={"prediction": "logits_10s"})
    pred_df["prob_10s"] = sigmoid(pred_df["logits_10s"].to_numpy())
    # filter out seizures with small number of segments.
    pred_filtered = filter_predictions(pred_df, pre_itvl, pid)
    return pred_filtered


def combine_prob(prob_10s, prob_ma, com_method, w_short=0.4, w_long=0.6):
    """
    Combines probabilities in a vectorized manner for the specified combination method.
    Args:
        prob_10s (pd.Series): Short-term probabilities.
        prob_ma (pd.Series): Long-term probabilities (moving average).
        com_method (str): Combination method to use.
        w_short (float): Weight for short-term probabilities (used only for weighted average).
        w_long (float): Weight for long-term probabilities (used only for weighted average).

    Returns:
        pd.Series: Combined probabilities.
    """
    if com_method == 'simple_avg':
        return (prob_10s + prob_ma) / 2
    elif com_method == 'weighted_avg46' or com_method == 'weighted_avg37':
        return w_short * prob_10s + w_long * prob_ma
    elif com_method == 'weighted_avg_dynamic':
        return w_short * prob_10s + w_long * prob_ma
    elif com_method == 'geometric_mean':
        return (prob_10s * prob_ma) ** 0.5
    elif com_method == 'harmonic_mean':
        denom = prob_10s + prob_ma
        return np.where(denom == 0, 0, 2 * prob_10s * prob_ma / denom)
    else:
        raise ValueError(f"Unknown combination method: {com_method}")


def poisson_chance_predictor(pred_df, thd_type='match_cosp'):
    """
    Generates chance predictions for a given prediction DataFrame using a Poisson process.
    ref:
    Snyder, D. E., et al. (2008-09-30). "The statistics of a practical seizure warning system." Journal of Neural Engineering 5(4).
    """
    time_intervals = pred_df['seg_epoch'].to_numpy()  # Extract time column
    pred_df['chance_pred_label'] = 0  # Initialize all labels as non-preictal

    if thd_type == 'match_cosp':
        # Compute Poisson rate (λ) as number of preictal events per second matching CoSP
        lambda_w = np.sum(pred_df['label_cosp']) / ( len(pred_df) * 10 )
    elif thd_type == 'pre_pct':
        lambda_w = np.sum(pred_df['label']) / (len(pred_df) * 10)
    else:
        print(f"ERROR: unkonwn thd_type={thd_type}")
        return

    if lambda_w == 0:
        print("Warning: No preictal events found. Returning original DataFrame with all labels 0.")
        return pred_df, 'chance_pred_label'

    # print(f"Computed Poisson rate λ = {lambda_w:.10f} events per second")

    # Start at the first timestamp
    current_time = time_intervals[0]
    max_time = time_intervals[-1]
    col_name = 'chance_pred_label'

    # Generate Poisson event times using Exponential inter-arrival sampling
    while current_time < max_time:
        inter_arrival_time = np.random.exponential(1 / lambda_w)  # Sample from Exp(λ)
        current_time += inter_arrival_time  # Move forward in time

        # Assign to closest available time in seg_epoch
        if current_time < max_time:
            closest_idx = np.searchsorted(time_intervals, current_time)  # Find nearest valid timestamp
            # print(f"Current time: {current_time}, Closest index: {closest_idx}")

            # Ensure closest_idx is within valid range
            if closest_idx < len(time_intervals):
                pred_df.iloc[closest_idx, pred_df.columns.get_loc(
                    col_name)] = 1  # Use .iloc to prevent new row creation
    print(f"true_preictal_fraction={(np.sum(pred_df['label'])/len(pred_df)):.2f}, "
          f"true_n_pre={np.sum(pred_df['label'])},  "
          f"chance_n_pre={np.sum(pred_df[col_name])}, "
          f"cosp_n_pre={int(np.sum(pred_df['label_cosp']))}")
    return pred_df, col_name


def remove_method_from_rst():
    # Load the CSV file
    file_path = RST_PATH / 'rst_warning.csv'
    df = pd.read_csv(file_path)

    # Remove rows where 'method' column equals 'chance_pred_label'
    df_filtered = df[df['method'] != 'chance_pred_label']

    # Save the cleaned file
    df_filtered.to_csv(file_path, index=False)
    print('done.')


def evaluate_combine_1m_10s(all_metrics, pid, pre_itvl, pred_tst, pred_val, retrigger, thd_type, thds):
    pred_val_ma, _, prob_col_val = moving_average_parallel(pred_val, pre_itvl, 'logits_10s')
    pred_tst_ma, _, prob_col_tst = moving_average_parallel(pred_tst, pre_itvl, 'logits_10s')
    for thd_type in ['0.5', 'val']:
        # for thd_type in ['val']:
        # for com_method in ['simple_avg', 'geometric_mean', 'harmonic_mean']:
        for com_method in ['harmonic_mean', 'simple_avg']:
            if thd_type == 'val':
                pred_val_ma['combined_prob'] = combine_prob(pred_val_ma['prob_10s'],
                                                            pred_val_ma[prob_col_val],
                                                            com_method)
                pps = []
                for thd_one in thds:
                    perf_metrics = evaluate_both_level(pid, pred_val_ma, pre_itvl, 'combined_prob',
                                                       sz_level_thd=thd_one, retrigger=retrigger)
                    pps.append(perf_metrics['Seizure_PP'])  # Store the PP metric
                # Select the best threshold based on maximum PP
                thd_best = thds[np.argmax(pps)]
            else:
                thd_best = 0.5

            # evaluate on the test set
            pred_tst_ma['combined_prob'] = combine_prob(pred_tst_ma['prob_10s'],
                                                        pred_tst_ma[prob_col_tst],
                                                        com_method)

            perf_metrics = evaluate_both_level(pid, pred_tst_ma, pre_itvl, 'combined_prob',
                                               sz_level_thd=thd_best, retrigger=retrigger)
            perf_metrics['method'] = f"combined_{com_method}"
            perf_metrics['thd_type'] = thd_type
            all_metrics.append(perf_metrics)
    return all_metrics


def evaluate_ma_pre_itval(all_metrics, inter_itvl, pid, pre_itvl, pred_tst, pred_val, retrigger, thd_type, thds):

    pred_tst_ma, _, prob_col_tst = moving_average_parallel(pred_tst, pre_itvl, 'logits_10s')
    # for thd_type in ['0.5', 'val']:  ## Final reported results of 'val' !!!
    # for thd_type in ['val']:  ## Final reported results of 'val' !!!
    if thd_type == 'val':
        pred_val_ma, _, prob_col_val = moving_average_parallel(pred_val, pre_itvl, 'logits_10s')
        # Get threshold from validation data
        print("Grid search for threshold from validation set...")
        pps = []
        for thd_one in thds:
            perf_metrics = evaluate_both_level(pid, pred_val_ma, pre_itvl, prob_col_val,
                                               sz_level_thd=thd_one, retrigger=retrigger)
            pps.append(perf_metrics['Seizure_PP'])  # Store the PP metric
        # Select the best threshold based on maximum PP
        thd_best = thds[np.argmax(pps)]
    else:
        thd_best = 0.5

        print("evaluating on test set...")
        # evaluate on testset
        perf_metrics = evaluate_both_level(pid, pred_tst_ma, pre_itvl, prob_col_tst,
                                           sz_level_thd=thd_best, retrigger=retrigger)
        perf_metrics['thd_type'] = thd_type
        all_metrics.append(perf_metrics)

        # update the prediction dataframes
        if thd_type == 'val':
            mid = get_final_model_id(pid, pre_itvl)

            pred_val_ma['thd_from_val'] = thd_best
            pred_val_ma['label_cosp'] = (pred_val_ma[prob_col_val] > thd_best).astype(int)
            f_pred = RST_PATH / f"detail/model{mid}_round1_pred_{inter_itvl}_val_final.csv"
            pred_val_ma.to_csv(f_pred, index=False)
            print(f"{f_pred} has been saved.")

            pred_tst_ma['thd_from_val'] = thd_best
            pred_tst_ma['label_cosp'] = (pred_tst_ma[prob_col_tst] > thd_best).astype(int)
            f_pred = RST_PATH / f"detail/model{mid}_round1_pred_{inter_itvl}_tst_final.csv"
            pred_tst_ma.to_csv(f_pred, index=False)
            print(f"{f_pred} has been saved.")
    return all_metrics


def evaluate_prob_10s(all_metrics, pid, pre_itvl, pred_tst, retrigger):
    prob_col = 'prob_10s'
    perf_metrics = evaluate_both_level(pid, pred_tst, pre_itvl, prob_col, sz_level_thd=0.5,
                                       retrigger=retrigger)
    perf_metrics['thd_type'] = '0.5'
    all_metrics.append(perf_metrics)
    return all_metrics


def evaluate_rf_1m(all_metrics, pid, pre_itvl, pred_tst, pred_val, thd_type, thds):
    # Prepare validation predictions using the helper method
    pred_val_ma, _, prob_col_val = moving_average_parallel(pred_val, pre_itvl, 'logits_10s')
    minute_df_val = prepare_minute_data(pred_val_ma, prob_col_val)
    # Prepare training data for the random forest model
    X_train = minute_df_val.drop(columns=["label", "sz_id", "seg_epoch"])
    y_train = minute_df_val["label"]
    rf_model = BalancedRandomForestClassifier(
        n_estimators=1000,
        sampling_strategy="auto",  # Automatically balance the classes
        replacement=True,  # Allow oversampling with replacement
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    # # Downsample only class 0
    # class_counts = y_train.value_counts()
    # n_class_1 = class_counts[1]  # Count of class 1 samples
    # # Downsample class 0 to twice the size of class 1
    # sampling_strategy = {0: n_class_1 * 2, 1: n_class_1}
    # undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    # X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
    #
    # # Train the random forest model
    # rf_model = RandomForestClassifier(
    #     n_estimators=100,
    #     class_weight={0: 1, 1: 2},  # Adjust weights for imbalance
    #     random_state=42,
    #     n_jobs=-1
    # )
    # rf_model.fit(X_resampled, y_resampled)
    # Prepare test predictions using the helper method
    pred_tst_ma, _, prob_col_tst = moving_average_parallel(pred_tst, pre_itvl, 'logits_10s')
    minute_df = prepare_minute_data(pred_tst_ma, prob_col_tst)
    # Predict and evaluate segment-level performance
    X_test = minute_df.drop(columns=["label", "sz_id", "seg_epoch"])
    y_test = minute_df["label"]
    minute_df["label"] = y_test
    minute_df["rf_1m_prob"] = rf_model.predict_proba(X_test)[:, 1]  # Probability of class 1
    # Evaluate both segment and seizure-level performance
    for thd_type in ['0.5', 'val']:
        # for thd_type in ['0.5']:
        if thd_type == 'val':
            # Get threshold from validation data
            pps = []
            minute_df_val["rf_1m_prob"] = rf_model.predict_proba(X_train)[:, 1]
            for thd_one in thds:
                perf_metrics_val = evaluate_both_level(pid, minute_df_val, pre_itvl, "rf_1m_prob",
                                                       sz_level_thd=thd_one)
                pps.append(perf_metrics_val['Seizure_PP'])  # Choose the metric to optimize

            # Select the best threshold based on maximum Seizure_PP
            thd_best = thds[np.argmax(pps)]
        else:
            thd_best = 0.5
        perf_metrics = evaluate_both_level(pid, minute_df, pre_itvl, "rf_1m_prob",
                                           sz_level_thd=thd_best, seg_sec=60)
        perf_metrics['thd_type'] = thd_type
        all_metrics.append(perf_metrics)
    return all_metrics


def evaluate_chance(all_metrics, inter_itvl, pid, pre_itvl, pred_tst, retrigger, thd_type):
    mid = get_final_model_id(pid, pre_itvl)
    f_pred = RST_PATH / f"detail/model{mid}_round1_pred_{inter_itvl}_tst_final.csv"
    assert f_pred.exists(), f"ERROR: {f_pred.name} doesn't exist."
    pred_cosp = pd.read_csv(f_pred, index_col=False)
    pred_tst['label_cosp'] = pred_cosp['label_cosp'].astype(int)
    metrics_list = []
    for _ in range(10):
        pred_chance, pred_col = poisson_chance_predictor(pred_tst, thd_type=thd_type)  # predictions are labels
        perf_metrics = evaluate_both_level(pid, pred_chance, pre_itvl, pred_col, 'label',
                                           seg_sec=10, retrigger=retrigger)
        perf_metrics['thd_type'] = thd_type
        perf_metrics['thd'] = ''
        metrics_list.append(perf_metrics)
    df_metrics = pd.DataFrame(metrics_list)
    col_order = df_metrics.columns
    print(df_metrics)
    # Select only numeric columns for mean computation
    numeric_cols = df_metrics.select_dtypes(include=['number'])
    # Compute the mean of numeric columns only
    mean_metrics = numeric_cols.mean().to_dict()
    # Store non-numeric metadata separately (e.g., method, thd_type)
    non_numeric_values = df_metrics.iloc[0][
        df_metrics.columns.difference(numeric_cols.columns)].to_dict()
    mean_metrics.update(non_numeric_values)
    mean_metrics = pd.Series(mean_metrics)[col_order].to_dict()
    print(mean_metrics)
    # Store the final processed metrics
    all_metrics.append(mean_metrics)
    return all_metrics


def evaluate_methods_all(pids, pre_itvls, methods=('ma_pre_itvl',),
                         inter_itvl=1440, thd_type='val', retrigger=True, save=False):
    """Evaluate explicitly selected private participants, intervals and methods."""
    all_metrics = []
    thds = np.arange(0, 1, 0.05)
    for pid in pids:
        for pre_itvl in pre_itvls:
            for method in methods:
                print(f"evluating: pid={pid}, pre_itvl={pre_itvl}, method={method}")
                perf_metrics = {'pid': pid, 'pre_itval': pre_itvl, 'thd_type': '0.5', 'thd': '', 'method': method}
                # prepare predictions
                pred_tst = prepare_prediction(pid, pre_itvl, inter_itvl=inter_itvl, data_type='tst')
                if method not in ['chance']:
                    pred_val = prepare_prediction(pid, pre_itvl, inter_itvl=inter_itvl, data_type='val')

                if method == 'chance':
                    all_metrics = evaluate_chance(all_metrics, inter_itvl, pid, pre_itvl, pred_tst, retrigger, thd_type)

                if method == 'rf_1m':
                    all_metrics = evaluate_rf_1m(all_metrics, pid, pre_itvl, pred_tst, pred_val, thd_type, thds)

                if method == 'prob_10s':
                    all_metrics = evaluate_prob_10s(all_metrics, pid, pre_itvl, pred_tst, retrigger)

                if method == 'ma_pre_itvl':
                    all_metrics = evaluate_ma_pre_itval(all_metrics, inter_itvl, pid, pre_itvl, pred_tst, pred_val,
                                                     retrigger, thd_type, thds)

                if method == 'combined':
                    all_metrics = evaluate_combine_1m_10s(all_metrics, pid, pre_itvl, pred_tst, pred_val, retrigger,
                                                       thd_type, thds)

            metrics_df = pd.DataFrame(all_metrics)
            all_metrics = []
            print(metrics_df)

            if save:
                # Save the DataFrame to a CSV file
                output_file = RST_PATH / f"rst_warning.csv"
                if output_file.exists():
                    # Append new data without writing the header
                    metrics_df.to_csv(output_file, mode='a', header=False, index=False)
                    print(f"New performance metrics appended to {output_file}")
                else:
                    # Write data with the header for the first time
                    metrics_df.to_csv(output_file, mode='w', header=True, index=False)
                    print(f"Performance metrics saved to {output_file}")
