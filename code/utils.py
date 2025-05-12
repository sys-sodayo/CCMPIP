import csv
import random
import re
import statistics
import numpy as np
import pandas as pd
import torch
import esm
from transformers import T5Tokenizer, T5EncoderModel
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, \
    confusion_matrix, roc_curve, auc


def file2list(filename):
    sequences = []
    with open(filename, 'r') as file:
        name, sequence = None, []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if name:
                    sequences.append((name, ''.join(sequence)))
                name, sequence = line, []
            else:
                sequence.append(line)
        if name:
            sequences.append((name, ''.join(sequence)))
    return sequences


def file2T5(inputpath, outputpath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载标记器
    tokenizer = T5Tokenizer.from_pretrained('E:\pretrained\prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    # 加载模型
    model = T5EncoderModel.from_pretrained('E:\pretrained\prot_t5_xl_half_uniref50-enc').to(device)

    model.eval()

    data = file2list(inputpath)
    sequence_representations = []
    data = [i[1] for i in data]

    batch_size = 1

    for i in range(0, len(data), batch_size):
        batch_seq = []
        print(data[i])
        print(i)
        batch = data[i:i + batch_size]
        batch_len = [len(i) for i in batch]
        batch = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in batch]
        ids = tokenizer(batch, add_special_tokens=True, padding='longest')
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)


            for i in range(len(embedding_repr)):
                seq_em = embedding_repr.last_hidden_state[i, :batch_len[i]]  # shape (len(seq) x 1024)
                seq_em_per_protein = seq_em.mean(dim=0)  # shape (1024)
                sequence_representations.append(seq_em_per_protein.cpu().tolist())
            # print(sequence_representations)


            torch.cuda.empty_cache()

        torch.save(torch.tensor(sequence_representations), outputpath)


def file2T5_tensor(inputpath, outputpath, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained('E:\pretrained\prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    model = T5EncoderModel.from_pretrained('E:\pretrained\prot_t5_xl_half_uniref50-enc').to(device)

    model.eval()

    data = file2list(inputpath)
    sequence_representations = []
    data = [i[1] for i in data]

    batch_size = 1

    for i in range(0, len(data), batch_size):
        batch_seq = []
        print(data[i])
        print(i)
        batch = data[i:i + batch_size]
        batch_len = [len(i) for i in batch]
        batch = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in batch]

        ids = tokenizer(batch, add_special_tokens=True, padding='longest')
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)


        for i in range(len(embedding_repr)):
            seq_em = embedding_repr.last_hidden_state[i, :]  # shape (len(seq) x 1024)
            batch_seq.append(seq_em.cpu().tolist())


        for j in range(batch_size):
            one_seq_data = batch_seq[j]
            if len(one_seq_data) < max_length:
                one_seq_data = torch.cat(
                    (torch.tensor(one_seq_data), (torch.tensor([[0] * 1024] * (max_length - len(one_seq_data))))),
                    dim=0)
                sequence_representations.append(one_seq_data)
            else:
                one_seq_data = one_seq_data[:max_length]
                sequence_representations.append(torch.tensor(one_seq_data))

            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    sequence_representations = torch.stack(sequence_representations, dim=0)
    torch.save(sequence_representations, outputpath)


def evaluate_model_performance(true_labels, predictions):
    """Evaluate model performance on classification metrics and plot confusion matrix."""
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    specificity = recall_score(true_labels, predictions, pos_label=0)
    f1 = f1_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)

    print(f"Acc: {accuracy:.4f}, "
          f"MCC: {mcc:.4f}, "
          f"Pr: {precision:.4f}, "
          f"Sn: {recall:.4f}, "
          f"Sp: {specificity:.4f}, "
          f"F1-score: {f1:.4f}, ", end="")

    plot_confusion_matrix(true_labels, predictions)


def plot_confusion_matrix(true_labels, predictions):
    """Plot the confusion matrix for given true and predicted labels."""
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Class 0', 'Class 1'])
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Display counts in each cell
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()



def roc(test_y, pr_list):
    """Plot ROC curve and calculate AUC score."""
    fpr, tpr, thresholds = roc_curve(test_y, pr_list)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xticks([(i / 10) for i in range(11)])
    plt.yticks([(i / 10) for i in range(11)])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    print(f"AUC: {roc_auc:.4f} ")


def fold_roc(fprs, tprs):
    aucs = []
    for i in range(len(fprs)):
        roc_auc = np.trapz(tprs[i], fprs[i])
        aucs.append(roc_auc)

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = 0.0
    for i in range(len(fprs)):
        mean_tpr += np.interp(mean_fpr, fprs[i], tprs[i])
    mean_tpr /= len(fprs)
    mean_tpr += 0.0
    mean_auc = np.trapz(mean_tpr, mean_fpr)


    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(mean_fpr, mean_tpr, color='navy', lw=2, label='Mean ROC (AUC = %0.2f)' % mean_auc)

    for i in range(len(fprs)):
        plt.plot(fprs[i], tprs[i], lw=1, alpha=0.3, label='Fold %d Roc(AUC = %0.2f)' % (i + 1, aucs[i]))

    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.xticks([(i / 10) for i in range(11)])
    plt.yticks([(i / 10) for i in range(11)])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of ten-fold cross validation')
    plt.legend(loc="lower right")
    plt.show()


def print_eva(accs, mccs, pres, recs, spes, f1s, aucs):
    print(f"Train Accuracy: {statistics.mean(accs):.4f}, "
          f"Train MCC: {statistics.mean(mccs):.4f}, "
          f"Train Precision: {statistics.mean(pres):.4f}, "
          f"Train Recall: {statistics.mean(recs):.4f}, "
          f"Train Specificity: {statistics.mean(spes):.4f}, "
          f"Train AUC: {statistics.mean(aucs):.4f}, "
          f"Train f1_score: {statistics.mean(f1s):.4f}, ")



def file2aaindex(file_path, outputpath, aaindex_file, max_len):

    aaindex_df = pd.read_excel(aaindex_file, index_col=0)
    aaindex_dict = aaindex_df.to_dict(orient='index')

    def get_features_for_sequence(sequence):
        features = np.zeros((max_len, 531))
        for i, aa in enumerate(sequence):
            if i >= max_len:
                break
            if aa in aaindex_dict:
                features[i, :] = np.array(list(aaindex_dict[aa].values()))
        return features

    data = file2list(file_path)
    data = [i[1] for i in data]
    #处理每条序列
    all_features = []
    for sequence in data:
        features = get_features_for_sequence(sequence)
        all_features.append(features.tolist())

    #return torch.tensor(all_features)
    print(len(all_features), len(all_features[0]), len(all_features[0][0]))
    torch.save(torch.tensor(all_features), outputpath)

def file2aaindex_avg(file_path, outputpath, aaindex_file):
    aaindex_df = pd.read_excel(aaindex_file, index_col=0)
    aaindex_dict = aaindex_df.to_dict(orient='index')

    def get_avg_features_for_sequence(sequence):
        features = []
        for aa in sequence:
            if aa in aaindex_dict:
                features.append(np.array(list(aaindex_dict[aa].values())))

        if len(features) == 0:
            return np.zeros(531)

        return np.mean(features, axis=0)

    data = file2list(file_path)
    data = [i[1] for i in data]

    all_features = []
    for sequence in data:
        avg_features = get_avg_features_for_sequence(sequence)
        all_features.append(avg_features)

    all_features_tensor = torch.tensor(all_features, dtype=torch.float32)
    print(f"Feature tensor shape: {all_features_tensor.shape}")

    torch.save(all_features_tensor, outputpath)

def list_to_csv(data, csv_file_path):

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i, group in enumerate(data):
            if isinstance(group, (list, np.ndarray)):
                row = []
                for j, seq in enumerate(group):
                    if isinstance(seq, (list, np.ndarray)):
                        seq_str = ",".join(map(str, seq))
                    else:
                        seq_str = str(seq)
                    row.append(seq_str)
                writer.writerow(row)
            else:

                writer.writerow([str(group)])
    print(f"数据已保存到 {csv_file_path}")


def csv_to_list(csv_file_path):

    nested_list = []
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            group = []
            for seq_str in row:
                if seq_str:
                    try:
                        seq = [float(x) for x in seq_str.split(',')]
                    except ValueError as e:
                        print(f"解析错误: {e} 在序列 {seq_str}")
                        seq = []
                else:
                    seq = []
                group.append(seq)
            nested_list.append(group)
    return nested_list



#标准化三维数据
def standardize_features_with_train_rules(train_tensor, test_tensor):
    """
    Standardize the features in train and test tensors based on the statistics of the training set.

    Args:
        train_tensor (torch.Tensor): Training tensor with shape (train_size, sequence_length, num_features).
        test_tensor (torch.Tensor): Testing tensor with shape (test_size, sequence_length, num_features).

    Returns:
        tuple: (standardized_train_tensor, standardized_test_tensor)
    """

    train_size, sequence_length, num_features = train_tensor.shape
    test_size, _, _ = test_tensor.shape


    standardized_train_tensor = torch.zeros_like(train_tensor)
    standardized_test_tensor = torch.zeros_like(test_tensor)


    for feature_idx in range(num_features):
        train_feature_values = train_tensor[:, :, feature_idx]

        flattened_train_values = train_feature_values.flatten()

        train_median = flattened_train_values.median()
        train_q1 = flattened_train_values.quantile(0.25)
        train_q3 = flattened_train_values.quantile(0.75)
        train_iqr = train_q3 - train_q1

        standardized_train_values = (train_feature_values - train_median) / (train_iqr + 1e-8)
        standardized_train_tensor[:, :, feature_idx] = standardized_train_values

        test_feature_values = test_tensor[:, :, feature_idx]

        standardized_test_values = (test_feature_values - train_median) / (train_iqr + 1e-8)
        standardized_test_tensor[:, :, feature_idx] = standardized_test_values

    return standardized_train_tensor, standardized_test_tensor



class EarlyStopping:

    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.Inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop








