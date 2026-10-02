import shutil
from pathlib import Path
from config import CONFIG, COMPUTER, MODEL_PATH, f_log, TEST, PAT_DICT, LOSS_PP_WEIGHT, PP_SCALE_FACTOR
from data import generate_rec_list_pat, gen_seg_list_pat, \
    split_segment_list, EEGDataset, get_coh_name
from models import get_model
from util import my_print, count_model_parameters, get_model_ids, \
    save_result, print_dict
from torch.utils.data import DataLoader
import torch, os, time, datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

class PolynomialDecayLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_updates,
        tot_updates,
        lr,
        end_lr,
        power,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False


###############   Training   #####################
def soft_pp_loss(logits, labels, sz_ids):
    """
    logits: Tensor [B] — raw logits from model
    labels: Tensor [B] — segment labels (0 or 1)
    sz_ids: Tensor [B] — seizure IDs for each segment
    """
    probs = torch.sigmoid(logits)  # [B], requires_grad=True

    # Seizure-level sensitivity (soft any)
    unique_sz_ids = torch.unique(sz_ids)
    seizure_preds = []
    for sz in unique_sz_ids:
        mask = (sz_ids == sz) & (labels == 1)
        if mask.sum() == 0:
            continue
        probs_sz = probs[mask]
        seizure_prob = 1 - torch.prod(1 - probs_sz + 1e-6)  # Soft "any" function
        # seizure_prob = 1 - torch.exp(torch.sum(torch.log(1 - probs_sz + 1e-6)))  # Improve numerical stability
        seizure_preds.append(seizure_prob)

    if len(seizure_preds) == 0:
        # print("⚠️ soft_pp_loss: No valid seizures in this batch.")
        return torch.tensor(0.0, device=logits.device, requires_grad=True)  # safe fallback

    sens = torch.stack(seizure_preds).mean()

    tiw = probs.mean()

    pp_proxy = sens * (1 - tiw)
    loss = -pp_proxy  # we want to maximize PP
    return loss


def train(rank=None):
    rank = 'cpu' if rank is None else rank
    pat_id = CONFIG['pat_id']
    pat = PAT_DICT[pat_id]
    if Path(f_log).exists(): os.remove(f_log)

    # generate segment list
    generate_rec_list_pat(pat)
    gen_seg_list_pat(pat)
    # print("\n")
    # print_dict(CONFIG)
    # print("\n")

    model = get_model(model_type=CONFIG['model'], config=CONFIG, device=rank)
    model_ids = get_model_ids(CONFIG)
    model_id, round_id = model_ids['model_id'], model_ids['round_id']
    # using .png to make sure the model is trained completely
    f_png = MODEL_PATH / f"model{model_id}_round{round_id}_training.png"
    if f_png.exists():
        my_print(1, f'Model {f_png.stem} exists. No need to train.')
        return None
    # save model_ids and config info
    if not TEST:
        save_result(model_ids, trn_info=None, config=CONFIG, rst_info=None, tst_info=None)

    # load EEG data
    pat = PAT_DICT[CONFIG['pat_id']]
    trn_list, val_list, _ = split_segment_list(pat, CONFIG['trn_label'])

    # Hyperparameters
    n_epochs = CONFIG['n_epochs']
    lr = CONFIG['learning_rate']
    lr_decay = CONFIG['lr_decay_step'] > 0
    patience = CONFIG['patience']  # for early stopping

    my_print(1, f"\nRank{rank} Training {CONFIG['model']} model using device {rank}...")
    my_print(1, f"Rank{rank} model_id={model_id}  round_id={round_id}   trn_label={CONFIG['trn_label']}   "
                f"n_epochs={CONFIG['n_epochs']}   batch_size={CONFIG['batch_size']}   "
                f"learning_rate={lr}")
    f_model = MODEL_PATH / f"model{model_id}_round{round_id}.pth"

    # # Load data
    batch_size = CONFIG['batch_size']
    trn_dataset = EEGDataset(trn_list, data_type='trn', config=CONFIG, phase='train', rank=rank)
    val_dataset = EEGDataset(val_list, data_type='val', config=CONFIG, phase='train', rank=rank)
    trn_shuffle = False if CONFIG['trn_label'] in ['pre', 'inter'] else True
    # Create DataLoader objects
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=trn_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize criterion, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss().to(rank)
    # PolynomialDecayLR
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = PolynomialDecayLR(optimizer,
                                  warmup_updates=0,
                                  tot_updates=n_epochs * 120,
                                  lr=lr,
                                  end_lr=1e-6,
                                  power=1.0)
    save_time = None
    train_losses = []
    train_bce_losses = []
    train_pp_losses = []
    val_losses = []
    early_stop_counter = 0
    best_val_loss, best_trn_loss = float('inf'), float('inf')
    best_epoch = -1
    n_segs_trn = 0
    n_segs_val = 0
    last_lr = -1

    start_time = time.time()
    for epoch in range(n_epochs):
        y_true_trn, y_pred_trn, y_true_val, y_pred_val = [], [], [], []
        if TEST and epoch > 1: break

        # --- training ---
        model.train()
        train_loss = 0
        loss_bce = 0
        loss_pp = 0
        for batch_idx, (data, label, _, sz_ids) in enumerate(trn_loader):
            if data.size(0) == 1:
                continue
            if TEST and batch_idx > 1: break
            if epoch == 0:
                n_segs_trn += data.shape[0]

            label = label.to(rank)
            sz_ids = sz_ids.to(rank)
            data, output = get_model_output(batch_idx, data, label, model, optimizer, rank, 'trn')
            y_true_trn.extend(label)
            # print(f"1============ data.shape={data.shape}, output.shape={output.shape}")
            loss_pp = soft_pp_loss(output, label, sz_ids) * PP_SCALE_FACTOR
            # loss.backward()

            loss_bce, label_pred = compute_loss_train(criterion, label, output)
            loss = (1 - LOSS_PP_WEIGHT) * loss_bce + LOSS_PP_WEIGHT * loss_pp
            # loss = loss_pp
            loss.backward()
            # print("2=========== loss.requires_grad: ", loss.requires_grad)  # should be True
            # print("3=========== loss: ", loss)  # should be a tensor, not float

            optimizer.step()
            # print(f"[Train] Epoch {epoch + 1}, Batch {batch_idx + 1}: "
            #       f"BCE Loss = {loss_bce.item():.4f}, "
            #       f"PP Loss = {loss_pp.item():.4f}, "
            #       f"Total Loss = {loss.item():.4f}")

            acc = None
            if label_pred is not None:
                y_pred_trn.extend(label_pred)
                acc = torch.sum(label == label_pred) / len(label)
            # my_print(2,
            #          f"{pat} Rank{rank} Epoch [{epoch + 1}/{n_epochs}], "
            #          f"batch [{batch_idx + 1}/{len(trn_loader)}], Loss: {loss.item():.8f}, Acc: {acc:.3f}")
            train_loss += loss.item()
            loss_bce += loss_bce.item()
            loss_pp += loss_pp.item()
        train_losses.append(train_loss / (batch_idx + 1))
        train_bce_losses.append(loss_bce / (batch_idx + 1))
        train_pp_losses.append(loss_pp / (batch_idx + 1))

        if lr_decay > 0:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            if current_lr != last_lr:
                print(f"[Epoch {epoch + 1}] Learning rate updated to: {current_lr:.6f}")
                last_lr = current_lr

        # --- validation ---
        model.eval()
        val_loss = 0
        n_batches = 0
        # print("4---=================: in validation now")
        with torch.no_grad():
            for batch_idx, (data, label, _, sz_ids) in enumerate(val_loader):
                if data.size(0) == 1:
                    continue
                if TEST and batch_idx > 1: break
                if epoch == 0: n_segs_val += data.shape[0]

                label = label.to(rank)
                sz_ids = sz_ids.to(rank)

                data, output = get_model_output(batch_idx, data, label, model, optimizer, rank, 'val')
                y_true_val.extend(label)
                loss_pp = soft_pp_loss(output, label, sz_ids)
                # loss.backward()
                loss_bce, label_pred = compute_loss_train(criterion, label, output)
                # loss = loss_bce + LOSS_PP_WEIGHT * loss_pp
                loss = loss_pp

                # print(f"[Val] Epoch {epoch + 1}, Batch {batch_idx + 1}: "
                #       f"BCE Loss = {loss_bce.item():.4f}, "
                #       f"PP Loss = {loss_pp.item():.4f}, "
                #       f"Total Loss = {loss.item():.4f}")

                val_loss += loss.item()
                if label_pred is not None:
                    y_pred_val.extend(label_pred)
                    acc = torch.sum(label == label_pred) / len(label)

                # my_print(2, f"{pat} Rank{rank} Validation: Epoch [{epoch + 1}/{n_epochs}], "
                #             f"batch [{batch_idx + 1}/{len(val_loader)}], Loss: {loss.item():.8f}, Acc: {acc:.3f}  "
                #             f"Average Label: {torch.mean(label.float()).item():.1f}")
                n_batches += 1
            val_loss /= n_batches
            val_losses.append(val_loss)

        if len(y_pred_trn) == len(y_true_trn):
            y_pred_trn, y_true_trn = torch.tensor(y_pred_trn), torch.tensor(y_true_trn)
            acc_trn = torch.sum(y_pred_trn == y_true_trn) / len(y_pred_trn)
        else:
            acc_trn = -1
        if len(y_pred_val) == len(y_true_val):
            y_pred_val, y_true_val = torch.tensor(y_pred_val), torch.tensor(y_true_val)
            acc_val = torch.sum(y_pred_val == y_true_val) / len(y_pred_val)
        else:
            acc_val = -1


        # --- Early stopping ---
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_trn_loss = train_losses[-1]
            best_epoch = epoch
            early_stop_counter = 0
            # Save best model
            torch.save(model.state_dict(), f_model)
            print(f"Best model saved at epoch {epoch+1}")
            save_time = time.time()
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                # Update the shared variable to indicate other processes should stop
                # with stop_flag.get_lock():
                #     stop_flag.value = 1
                my_print(1,
                         f"{pat} Rank{rank} Early stopping: epoch={best_epoch}, best_trn_loss={best_trn_loss:.8f}  best_val_loss={best_val_loss:.8f}.")
                break
        if early_stop_counter:
            epoch_stop = patience - early_stop_counter
        else:
            epoch_stop = n_epochs - epoch
        my_print(1, f"{pat},Rank{rank}, Epoch {epoch + 1}/{n_epochs},{epoch_stop} to stop ===> "
                    f"Train: loss={train_losses[-1]:.5f} (bce={train_bce_losses[-1]:.5f},pp={train_pp_losses[-1]:.5f}), accuracy={acc_trn:.3f},  "
                    f"Validation: loss={val_losses[-1]:.5f}, accuracy={acc_val:.3f}"
                 )

        # if epoch == 0:
        #     # save the segments_dic
        #     f_coh = get_coh_name(pat, 'trn')
        #     if not f_coh.exists():
        #         f_coh.parent.mkdir(parents=True, exist_ok=True)
        #         trn_dataset.save_segments_dic(f_coh)
        #
        #     f_coh = get_coh_name(pat, 'val')
        #     if not f_coh.exists():
        #         f_coh.parent.mkdir(parents=True, exist_ok=True)
        #         val_dataset.save_segments_dic(f_coh)

    end_time = time.time()
    training_time = end_time - start_time
    my_print(1, f"{pat} Rank{rank} Total training time: {training_time:.2f} seconds")

    # save traning process
    # Convert the timestamp to a datetime object
    if save_time is not None:
        dt_object = datetime.datetime.fromtimestamp(save_time)
        # Format the datetime object
        save_time = dt_object.strftime('%Y-%m-%d %H:%M')
    n_para_total, n_para_train = count_model_parameters(model)
    trn_info = {
        'n_segs_trn': n_segs_trn,
        'n_segs_val': n_segs_val,
        'computer_name': COMPUTER,
        'trained_epochs': epoch + 1,
        'best_epochs': best_epoch + 1,
        'train_time_seconds': training_time,
        'train_time_per_epoch': training_time / (epoch + 1),
        'n_model_parameters_total': n_para_total,
        'n_model_parameters_train': n_para_train,
        'stop_trn_loss': train_losses[-1],
        'stop_val_loss': val_losses[-1],
        'best_val_trn_loss': best_trn_loss,
        'best_val_loss': best_val_loss,
        'save_time': save_time
    }
    save_result(model_ids, trn_info=trn_info, config=CONFIG)
    # Plotting the training process
    save_training_plot(model_id, round_id, train_losses, val_losses)

    if Path(f_log).exists():
        shutil.move(f_log, MODEL_PATH / f"model{model_id}_round{round_id}_training.log")

    # free the GPU memory, avoid  Memory Fragmentation Error
    # torch.cuda.empty_cache()
    return model_ids


def get_model_output(data_key, data, label, model, optimizer, rank, data_type):
    # data has been pre-processed and normalized in EEGDataset.__get_items__
    if optimizer is not None: optimizer.zero_grad()
    data = data.to(rank)
    if data.ndim == 3:  # (B, H, W)
        data = data.unsqueeze(1)  # -> (B, 1, H, W)
    output = model(data).squeeze()  # shape: [B]
    return data, output


###############   Post Training   #####################
def save_training_plot(model_id, round_id, train_losses, val_losses):
    x_values = np.arange(1, len(train_losses) + 1)
    plt.plot(x_values, train_losses, label='Training loss')
    x_values = np.arange(1, len(val_losses) + 1)
    plt.plot(x_values, val_losses, label='Validation loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(MODEL_PATH / f'model{model_id}_round{round_id}_training.png')
    plt.close()


def compute_loss_train(criterion, label, output):
    label = label.to(output.device)
    loss = criterion(output, label.float())
    label_pred = torch.zeros_like(output)
    label_pred[torch.sigmoid(output) >= 0.5] = 1
    label_pred = label_pred.flatten()
    return loss, label_pred.to(output.device)
