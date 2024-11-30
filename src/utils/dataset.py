import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from models.reshape_data import ReshapeDataSet
import torch

# 加载原始数据集
def dataset_loader(raw_data_dir,file_name):
    data_path = os.path.join(raw_data_dir, file_name)
    data = pd.read_csv(data_path)
    return data # 返回数据集为 DataFrame 对象

def preprocess_data(data):
    x = data.drop(columns=['Class']) # 返回 x 为特征数据(去除 Class 列)
    y = data['Class'] # 返回 y 为标签数据(Class)
    return x,y

def find_known_class(N,y): # N 为已知类别的数量,且 N 的取值范围为 {7,9,11,13}
    unique_labels = y.unique()
    known_classes = np.random.choice(unique_labels, N, replace=True)
    return known_classes # 返回 N 个已知类别 根据论文的说法作为已知类别的同时，也是旧类别
        

def split_data(x,y,N,test_size = 1/6):
    
    known_classes = find_known_class(N,y) 
    known_mask = y.isin(known_classes) # 同理
    unknown_mask = ~known_mask # 创建布尔掩码用于表示未知类别和已知样本
    
    x_known = x[known_mask]
    y_known = y[known_mask]
    
    x_unknown = x[unknown_mask]
    y_unknown = y[unknown_mask]
    
   # 将未知类别的标签统一设置为 num_classes
    unknown_label = len(known_classes)
    y_unknown = pd.Series([unknown_label] * len(y_unknown), index=y_unknown.index)

    # 对已知类别的标签进行编码（从 0 到 N-1）
    y_known = y_known.replace(dict(zip(known_classes, range(len(known_classes)))))

    # 按照一定比例划分已知类别的数据 并且训练集只包含已知类别 未知类别用于加入测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x_known, y_known, test_size=test_size, stratify=y_known, random_state=42
    )
    
    # 将未知类别数据添加到测试集中
    x_test = pd.concat([x_test, x_unknown])
    y_test = pd.concat([y_test, y_unknown])
    return x_train, x_test, y_train, y_test,known_classes
    

def create_episode(X_train, y_train, num_episode=100):
    episodes = []
    for _ in range(num_episode):
        support_set, query_set, support_labels, query_labels = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
        support_set = torch.tensor(support_set.values, dtype=torch.float32)
        query_set = torch.tensor(query_set.values, dtype=torch.float32)
        support_labels = torch.tensor(support_labels.values, dtype=torch.long)
        query_labels = torch.tensor(query_labels.values, dtype=torch.long)
        episodes.append((support_set, query_set, support_labels, query_labels))
    return episodes

def make_processed_dataset(x_train,x_test,y_train,y_test,processed_data_dir):
    os.makedirs(processed_data_dir, exist_ok=True)
    
    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)
    
    train_data.to_csv(os.path.join(processed_data_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(processed_data_dir, 'test.csv'), index=False)
    
def load_processed_data(file_name,raw_data_dir,processed_data_dir,N):
    data = dataset_loader(raw_data_dir,file_name)
    x,y = preprocess_data(data)
    x_train, x_test, y_train, y_test,known_classes = split_data(x,y,N)
    make_processed_dataset(x_train,x_test,y_train,y_test,processed_data_dir)
    
    return (x_train, x_test, y_train, y_test,known_classes)

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size):
    train_dataset = ReshapeDataSet(X_train, y_train)
    test_dataset = ReshapeDataSet(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

