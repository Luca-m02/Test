import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from metrics import get_results_summary
import logging
import time

def train_epoch(args, model, train_dataloader, val_dataloader, data_min, data_max, optimizer, epoch, device):
    ##### Training #####
    model.train(True)
    train_loss_kl = []
    train_loss_mse = []
    train_loss = []

    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        #loss_kl = F.kl_div(torch.log(F.softmax(logits.transpose(1,2) / 0.1, dim=-1) + 1e-9), F.softmax(targets.transpose(1,2) / 0.1, dim=-1), reduction='batchmean')
        loss_mse = F.mse_loss(logits, targets)
        loss_total = loss_mse   # loss_kl +
        loss_total.backward()
        optimizer.step()
        #train_loss_kl.append(loss_kl.detach())
        train_loss_mse.append(loss_mse.detach())
        train_loss.append(loss_total.detach())

    ##### Validation #####
    model.eval()
    val_mae = []
    val_mre = []

    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            mae_score = F.l1_loss(logits, targets)
            mre_score = mean_relative_error(logits, targets)
            val_mae.append(mae_score.detach())
            val_mre.append(mre_score.detach())

    #train_loss_kl = torch.stack(train_loss_kl).mean().item()
    train_loss_mse = torch.stack(train_loss_mse).mean().item()
    train_loss = torch.stack(train_loss).mean().item()
    val_mae = torch.stack(val_mae).mean().item()
    val_mre = torch.stack(val_mre).mean().item()

    ##### Metrics #####
    logging.debug(
        f'Epoch: {epoch+1}/{args.n_epochs} - Train Loss: {train_loss:.4f} - Train MSE Loss: {train_loss_mse:.4f} - Validation MAE: {val_mae:.4f} - Validation MRE: {val_mre:.4f}',
    )

    return val_mae, val_mre


def train(args, model, train_dataloader, val_dataloader, data_min, data_max, optimizer, path_ckpts, device):
    best_mae_score = np.inf
    best_mre_score = np.inf
    print('Training full model ...')

    n_epochs_without_improvements = 0
    for epoch in tqdm(range(args.n_epochs), total=args.n_epochs, leave=False):
        mae_score, mre_score = train_epoch(args, model, train_dataloader, val_dataloader, data_min, data_max, optimizer, epoch, device)
        if mae_score < best_mae_score:
            best_mae_score = mae_score
            best_mre_score = mre_score
            #print('New best epoch: {}/{}, Validation MAE: {:.4f}, Validation MRE: {:.4f}.'.format(epoch+1, args.n_epochs, best_mae_score, best_mre_score))
            torch.save(model.state_dict(), path_ckpts)
            #print('Model saved to: {}'.format(path_ckpts))
            n_epochs_without_improvements = 0
        else:
            n_epochs_without_improvements += 1
        if n_epochs_without_improvements == args.early_stopping:
            print(f'No improvements for {args.early_stopping} epochs: end training')
            break

    return


def test(args, model, test_dataloader, data_min, data_max, device):
    ##### Testing #####
    model.eval()
    power_pred = []
    power_true = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_dataloader, total=len(test_dataloader), leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            logits = denormalize(logits, data_min, data_max)
            # targets = denormalize(targets, data_min, data_max) i targets non vengono normalizzati per niente nel test
            power_pred.append(logits)
            power_true.append(targets)

        power_pred = torch.cat(power_pred, dim=0).cpu().numpy()
        power_true = torch.cat(power_true, dim=0).cpu().numpy()
        print(f"predicted shape: {power_pred.shape}, label shae: {power_true.shape}")
        power_pred = np.where(power_pred < 0, 0, power_pred)
        res = get_results_summary(power_true, power_pred)

        return res, power_pred, power_true


def mean_relative_error(pred, label, eps=1e-9):
    temp = torch.full_like(label, eps)
    maximum, _ = torch.max(torch.stack([label, pred, temp], dim=-1), dim=-1)
    return torch.mean(torch.nan_to_num(torch.abs(label - pred) / maximum))


def denormalize(data, data_min, data_max):
    return (data_max - data_min) * data + data_min
