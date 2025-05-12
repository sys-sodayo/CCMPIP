
import csv
import os
import random
import statistics

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, \
    roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from utils import roc, print_eva, fold_roc, list_to_csv, evaluate_model_performance
'''from  getdatalist import train_x_tensor, train_y_tensor, train_aaindex, test_aaindex, test_x_tensor, test_y_tensor

train_t5,train_aaindex,train_y =    train_x_tensor,train_aaindex, train_y_tensor
test_t5, test_aaindex, test_y = test_x_tensor,test_aaindex, test_y_tensor'''
from  getdatalist import test_aaindex, test_x_tensor, test_y_tensor

test_t5, test_aaindex, test_y = test_x_tensor,test_aaindex, test_y_tensor


class Squash(nn.Module):

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, s: torch.Tensor):  # s: [batch_size, n_capsules, n_features]
        s2 = (s ** 2).sum(dim=-1, keepdims=True)
        return (s2 / (1 + s2)) * (s / torch.sqrt(s2 + self.epsilon))



class Router(nn.Module):
    def __init__(self, in_caps: int, out_caps: int, in_d: int, out_d: int, iterations: int):  # int_d: 前一层胶囊的特征数目
        super().__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.iterations = iterations
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()

        self.weight = nn.Parameter(torch.randn(in_caps, out_caps, in_d, out_d), requires_grad=True)

    def forward(self, u: torch.Tensor):

        u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)
        b = u.new_zeros(u.shape[0], self.in_caps, self.out_caps)
        v = None
        for i in range(self.iterations):
            c = self.softmax(b)
            s = torch.einsum('bij,bijm->bjm', c, u_hat)
            v = self.squash(s)
            a = torch.einsum('bjm,bijm->bij', v, u_hat)
            b = b + a
        return v


class cross_attention(nn.Module):
    def __init__(self):
        super(cross_attention, self).__init__()

        self.linearK = nn.Sequential(nn.Linear(1024, 531))
        self.linearQ = nn.Sequential(nn.Linear(531, 531))
        self.linearV = nn.Sequential(nn.Linear(1024, 1024))

        self.softmax = nn.Softmax()

        self.linear_mid = nn.Linear(1024, 1024)
        self.linear_final = nn.Linear(1024, 1024)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.2)
        self.cnn = nn.Sequential(
            nn.Conv1d(
            in_channels= 1024,
            out_channels= 512,
            kernel_size=3,
            stride=1,
            padding=1,
        ),
            nn.BatchNorm1d(512),
            nn.ELU(),  # activation
            nn.Conv1d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm1d(256),
            nn.ELU(),  # activation
        )

        self.capsule_layer = Router(in_caps=256, out_caps=2,
                                    in_d=25, out_d=8,iterations=2)


        self.fc = nn.Sequential(nn.Linear(16, 64),
                                nn.ELU(),
                                nn.Dropout(0.3),
                                nn.Linear(64, 2))



    def forward(self, x_t5, x_aaindex):

        Q, K, V = x_aaindex, x_t5, x_t5
        K = self.linearK(K)
        Q = self.linearQ(Q)
        V = self.linearV(V)
        attn = torch.einsum('bij,bkj->bik', Q, K)
        attn_sum = torch.einsum('bij->bi', attn)
        attn = torch.einsum('bij,bi->bij', attn, 1 / attn_sum)
        x_t5 = torch.einsum('bij,bjd->bid', attn, V)
        x_t5 = self.dropout(self.elu(self.linear_mid(x_t5)))

        x_t5 = x_t5 + V
        att_out = self.linear_final(x_t5)

        att_out = att_out.permute(0, 2, 1)

        cnn_out = self.cnn(att_out)
        cap_out = self.capsule_layer(cnn_out)
        view = cap_out.view(cap_out.size(0),-1)
        out = self.fc(view)

        return out, attn


def train_model(x_t5, x_aaindex, y):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 40
    batch_size = 32
    fold = 1
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    y = torch.Tensor(y)

    accs, mccs, pres, recs, spes, f1s, aucs, fprs, tprs, recalls, precisions  = [], [], [], [], [], [], [], [], [], [], []
    models = []



    for train_index, val_index in skf.split(x_t5, y):
        print(f"Fold: {fold}")

        model = cross_attention().to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr= 5e-5)

        train_fold_t5, val_fold_t5 = x_t5[train_index], x_t5[val_index]
        train_fold_aaindex, val_fold_aaindex = x_aaindex[train_index], x_aaindex[val_index]
        train_fold_labels, val_fold_labels = y[train_index], y[val_index]

        best_mcc = 0
        best_model_state = None
        fold_best_metrics = {}
        best_epoch = 0
        for epoch in range(num_epochs):
            model.train()
            front = 1
            while front <= len(train_fold_t5):
                behind = min(front + batch_size - 1, len(train_fold_t5))
                inputs_t5 = train_fold_t5[front - 1:behind].to(device)
                inputs_aaindex = train_fold_aaindex[front - 1:behind].to(device)
                labels = train_fold_labels[front - 1:behind].to(device)

                optimizer.zero_grad()
                outputs,_ = model(inputs_t5, inputs_aaindex)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

                front = behind + 1


            model.eval()
            val_predictions = []
            pr_list = []
            with torch.no_grad():
                front = 1
                while front <= len(val_fold_t5):
                    behind = min(front + batch_size - 1, len(val_fold_t5))
                    inputs_t5 = val_fold_t5[front - 1:behind].to(device)
                    inputs_aaindex = val_fold_aaindex[front - 1:behind].to(device)

                    outputs,_ = model(inputs_t5, inputs_aaindex)
                    pred = torch.argmax(outputs, dim=-1).tolist()
                    Y_pred_1 = [y[1] for y in outputs]
                    pr_list.extend(Y_pred_1)
                    val_predictions.extend(pred)
                    front = behind + 1

            pr_list = torch.tensor(pr_list).numpy()
            auc = roc_auc_score(val_fold_labels.numpy(), pr_list)
            acc = accuracy_score(val_fold_labels, val_predictions)
            pre = precision_score(val_fold_labels, val_predictions)
            rec = recall_score(val_fold_labels, val_predictions)
            spe = recall_score(val_fold_labels, val_predictions, pos_label=0)
            f1 = f1_score(val_fold_labels, val_predictions)
            mcc = matthews_corrcoef(val_fold_labels, val_predictions)
            fpr, tpr, _ = roc_curve(val_fold_labels, val_predictions)
            precision, recall, _ = precision_recall_curve(val_fold_labels.numpy(), pr_list)

            num_samples = 100
            fpr_sampled = np.linspace(0, 1, num_samples)
            tpr_sampled = np.interp(fpr_sampled, fpr, tpr)
            precision_sampled = np.linspace(0, 1, num_samples)
            recall_sampled = np.interp(precision_sampled, precision, recall)


            if mcc > best_mcc:
                best_mcc = mcc
                best_model_state = model.state_dict()
                fold_best_metrics = {
                    "acc": acc, "mcc": mcc, "pre": pre, "rec": rec,
                    "spe": spe, "f1": f1, "auc": auc, "fpr_sampled": fpr_sampled, "tpr_sampled": tpr_sampled,
                    "precision_sampled": precision_sampled, "recall_sampled": recall_sampled
                }
                best_epoch = epoch
        print_eva([fold_best_metrics["acc"]], [fold_best_metrics["mcc"]], [fold_best_metrics["pre"]],
                  [fold_best_metrics["rec"]], [fold_best_metrics["spe"]], [fold_best_metrics["f1"]],
                  [fold_best_metrics["auc"]])
        print("best_epoch: ", best_epoch)
        accs.append(fold_best_metrics["acc"])
        mccs.append(fold_best_metrics["mcc"])
        pres.append(fold_best_metrics["pre"])
        recs.append(fold_best_metrics["rec"])
        spes.append(fold_best_metrics["spe"])
        f1s.append(fold_best_metrics["f1"])
        aucs.append(fold_best_metrics["auc"])
        fprs.append(fold_best_metrics["fpr_sampled"])
        tprs.append(fold_best_metrics["tpr_sampled"])
        recalls.append(fold_best_metrics["recall_sampled"])
        precisions.append((fold_best_metrics["precision_sampled"]))

        fold += 1
        models.append(best_model_state)


    best_model = models[mccs.index(max(mccs))]
    torch.save(best_model, "../../model/cross_att_cnn_qaa_cap.pt")



    print("k_fold mean:")
    print_eva(accs, mccs, pres, recs, spes, f1s, aucs)

    fold_roc(fprs, tprs)

    violin_data = {'SN': recs, 'SP': spes, 'ACC': accs, 'AUC': aucs, 'MCC': mccs
                   }
    violin_data = pd.DataFrame(violin_data)
    violin_data.to_csv('../../ablosion_data/cross_att_cnn_qaa_cap.csv', index=False)


    mean_precision = np.mean(precisions, axis=0)
    mean_recall = np.mean(recalls, axis=0)
    mean_fpr = np.mean(fprs, axis=0)
    mean_tpr = np.mean(tprs, axis=0)
    pr_curve_data = pd.DataFrame({'Precision': mean_precision, 'Recall': mean_recall})
    pr_curve_data.to_csv('../../pr_data/cross_att_cnn_qaa_cap.csv', index=False)
    roc_curve_data = pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr})
    roc_curve_data.to_csv('../../roc_data/cross_att_cnn_qaa_cap.csv', index=False)


def  to_test_model (x_t5, x_aaindex, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cross_attention().to(device)
    model.load_state_dict(torch.load("../model/model.pt"))
    model.eval()
    y = torch.tensor(y)


    pr_list,test_predictions,outs_list = [],[],[]
    with torch.no_grad():
        for start in range(0, len(x_t5), 40):
            inputs_t5 = x_t5[start:start + 40].to(device)
            inputs_aaindex = x_aaindex[start:start + 40].to(device)
            outputs,_ = model(inputs_t5, inputs_aaindex)
            outs_list.extend(outputs)
            test_predictions.extend(torch.argmax(outputs, dim=-1).tolist())
            pr_list.extend(outputs[:, 1].cpu().numpy())

    evaluate_model_performance(y, test_predictions)

    roc(y,pr_list)
    return outs_list


if __name__ == "__main__":

    #train_model(train_t5, train_aaindex, train_y)
    outs = to_test_model(test_t5, test_aaindex, test_y)





