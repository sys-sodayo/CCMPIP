import numpy as np
import torch
import random

from sklearn.model_selection import train_test_split

#from Python_project.utils import standardize_features
from utils import standardize_features_with_train_rules

'''''''''''''''''''''''''t5+aaindex'''''''''''''''''''''''''
'''train_x_tensor = torch.load('../dataset/ProtT5/train_t5.pt')
test_x_tensor = torch.load('../dataset/ProtT5/test_t5.pt')
train_aaindex = torch.load('../dataset/train_aaindex.pt')
test_aaindex = torch.load('../dataset/test_aaindex.pt ')
train_aaindex_avg = torch.load('../dataset/train_aaindex_avg.pt')

train_aaindex, test_aaindex = standardize_features_with_train_rules(train_aaindex, test_aaindex)

# 保存标准化后的结果
#torch.save(train_aaindex, '../dataset/train_aaindex_std.pt')
#torch.save(test_aaindex, '../dataset/test_aaindex_std.pt')

train_y_tensor = [1]*1245 + [0]*1627
test_y_tensor = [1]*171 + [0]*171
data = list(zip(train_x_tensor, train_aaindex, train_aaindex_avg, train_y_tensor))

random.seed(1)
random.shuffle(data)
train_x_tensor, train_aaindex, train_aaindex_avg, train_y_tensor = zip(*data)
train_x_tensor = torch.stack(train_x_tensor, dim=0)
train_aaindex = torch.stack(train_aaindex, dim=0)
train_aaindex_avg = torch.stack(train_aaindex_avg, dim=0)'''

'''''''''''''''''''''''''test_data'''''''''''''''''''''''''
test_x_tensor = torch.load('../test_data/test_t5.pt')
test_aaindex = torch.load('../test_data/test_aaindex_std.pt ')
test_y_tensor = [1]*171 + [0]*171
