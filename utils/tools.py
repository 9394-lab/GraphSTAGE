import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns


plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.learning_rate * 0.5 * (1 + math.cos(math.pi * epoch / 10))}

    elif args.lradj == 'type3':
        lr_adjust = {
            2: 1e-3, 5: 5e-4, 7: 1e-5,
            8: 5e-6, 9: 1e-7, 10: 5e-8
        }
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** epoch)}

    elif args.lradj == 'type4traffic':
        lr_adjust = {
            2: 2e-3, 3: 1e-3, 5: 5e-4
        }

    elif args.lradj == 'type4ECL':
        lr_adjust = {
            1: 5e-4, 3:2e-4, 5:1e-4
        }

    elif args.lradj == 'type4solar':
        lr_adjust = {
            1: 5e-4, 3:2e-4, 5:1e-4
        }

    elif args.lradj == 'type4ETTh':
        lr_adjust = {
            3: 5e-6
        }

    elif args.lradj == 'type4ETTm':
        lr_adjust = {
            1: 1e-5, 3:2e-4, 5:1e-4
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf', setting=None):
    """
    Results visualization
    """
    fontsize = 12
    plt.figure()
    plt.plot(preds, label='Prediction', linewidth=3, color='tomato')
    plt.plot(true, label='GroundTruth', linewidth=2, color='royalblue')

    model_name = setting.split('_')[3]
    plt.title(model_name, fontsize=fontsize+6)
    plt.legend(fontsize=fontsize, loc='upper left')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)

    plt.savefig(name, bbox_inches='tight')


def heatmap_visual(Tem_graph, Spa_graph, tem_graph_name, spa_graph_name):
    if isinstance(Tem_graph, torch.Tensor):
        Tem_graph = Tem_graph.detach().cpu().numpy()
    if isinstance(Spa_graph, torch.Tensor):
        Spa_graph = Spa_graph.detach().cpu().numpy()
    fontsize = 12

    print("Tem_graph min:", Tem_graph.min(), "max:", Tem_graph.max(), "mean:", Tem_graph.mean())
    print("Spa_graph min:", Spa_graph.min(), "max:", Spa_graph.max(), "mean:", Spa_graph.mean())

    #  Temporal Graph Heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(Tem_graph, cmap='viridis')
    plt.title('Temporal Learnable Graph', fontsize=16)
    plt.xlabel('Patch Index', fontsize=16)
    plt.ylabel('Patch Index', fontsize=16)
    plt.gca().invert_yaxis()
    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(tem_graph_name, bbox_inches='tight')
    plt.close()

    #  Spatial Graph Heatmap
    plt.figure(figsize=(8, 8))
    sns.heatmap(Spa_graph, cmap='viridis')
    plt.title('Spatial Learnable Graph', fontsize=16)
    plt.xlabel('Node Index', fontsize=16)
    plt.ylabel('Node Index', fontsize=16)
    plt.gca().invert_yaxis()
    plt.tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(spa_graph_name, bbox_inches='tight')
    plt.close()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
