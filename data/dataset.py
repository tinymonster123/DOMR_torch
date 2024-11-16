import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

def create_episode(x_train,y_train,num_episodes=100):
    episodes = []
    for _ in range(num_episodes):
        support_set, query_set, support_labels, query_labels = train_test_split(x_train, y_train, test_size=0.5)
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

if __name__ == '__main__':
    raw_data_dir = 'data/raw'
    processed_data_dir = 'data/processed'
    file_name = 'cicmaldroid2020.csv'
    
    x_train,x_test,y_train,y_test =load_processed_data(file_name,raw_data_dir,processed_data_dir)
    
    episodes = create_episode(x_train,y_train)
    
    print(f'shape of train set: {x_train.shape}, {y_train.shape}')
    print(f'shape of test set: {x_test.shape}, {y_test.shape}')    
    print(f'number of episodes: {len(episodes)}')