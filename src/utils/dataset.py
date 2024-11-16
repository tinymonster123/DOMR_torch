import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch

def dataset_loader(raw_data_dir,file_name):
    data_path = os.path.join(raw_data_dir, file_name)
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data):
    x = data.drop(columns=['Class'])
    y = data['Class']
    return x,y


def split_data(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

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
    
def load_processed_data(file_name,raw_data_dir,processed_data_dir):
    data = dataset_loader(raw_data_dir,file_name)
    x,y = preprocess_data(data)
    x_train, x_test, y_train, y_test = split_data(x,y)
    make_processed_dataset(x_train,x_test,y_train,y_test,processed_data_dir)
    
    return (x_train, x_test, y_train, y_test)

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size):
    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32),
                                 torch.tensor(y_test.values, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

