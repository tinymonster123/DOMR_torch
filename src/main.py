import torch
import pandas as pd
import numpy as np
from utils.dataset import load_processed_data, create_dataloaders, create_episode
from models.representation import Representation
from models.recognizer import aggregation
from models.meta_learning import MAML, train_maml_model
from models.classifier import Classifier, train_classifier
from utils.evaluation import evaluate_model
from utils.feature_extraction import extract_features
from utils.incremental_update import incremental_update
from utils.expand import expand_classifier

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置数据集文件路径
    raw_data_dir = '/root/DOMR_torch/data/raw'
    processed_data_dir = '/root/DOMR_torch/data/processed'
    file_name = 'cicmaldroid2020.csv'

    # 设置超参数
    batch_size = 32
    inner_lr = 0.01
    outer_lr = 0.001
    inner_steps = 5
    meta_iterations = 10
    classifier_epochs = 10
    N = 7

    # 加载并处理数据
    X_train, X_test, y_train, y_test,known_classes = load_processed_data(file_name, raw_data_dir, processed_data_dir,N)

    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=batch_size)
    
    # 创建特征提取器
    feature_dim = 128
    feature_extractor = Representation().to(device)
    
    # 创建 Episodes
    episodes = create_episode(X_train, y_train, num_episode=100)
    
    # 初始化 MAML 模型
    maml_model = MAML( inner_lr, outer_lr, inner_steps, device)
    train_maml_model(maml_model, episodes, num_iterations=meta_iterations,known_classes=known_classes)
    torch.save(feature_extractor.state_dict(), '/root/DOMR_torch/experiment/logs/feature_extractor.pth')

    feature_extractor.load_state_dict(torch.load('/root/DOMR_torch/experiment/logs/feature_extractor.pth',weights_only=True))
    feature_extractor.eval()

    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    torch.save((train_features, train_labels), '/root/DOMR_torch/experiment/logs/train_features.pth')

    test_features, test_labels = extract_features(feature_extractor, test_loader, device)
    torch.save((test_features, test_labels), '/root/DOMR_torch/experiment/logs/test_features.pth')

    num_classes = len(set(y_train))
    classifier = Classifier(input_dim=feature_dim, num_classes=num_classes).to(device)
    train_classifier(classifier, train_features.numpy(), train_labels.numpy(), device, epochs=classifier_epochs)
    torch.save(classifier.state_dict(), '/root/DOMR_torch/experiment/logs/classifier.pth')

       # 使用聚合策略进行预测
    preds = aggregation(classifier, feature_extractor, X_test, device, num_classes)

    # 将 y_test 转换为 numpy 数组
    if isinstance(y_test, torch.Tensor):
        y_true = y_test.cpu().numpy()
    elif isinstance(y_test, pd.Series):
        y_true = y_test.values
    else:
        y_true = np.array(y_test)

    # 评估模型
    evaluate_model(preds, y_true)
    
    unknown_label = num_classes  # 未知类别的标签
    unknown_indices = preds == unknown_label
    X_new = X_test[unknown_indices]
    y_new = y_test[unknown_indices]

    # 如果没有未知类别的样本，增量更新无法进行
    if len(X_new) == 0:
        print("No unknown samples detected. Incremental update is not performed.")
        return

    # 重新编码新类别标签（假设新类别的标签从 num_classes 开始）
    unique_new_labels = np.unique(y_new)
    label_mapping = {label: idx + num_classes for idx, label in enumerate(unique_new_labels)}
    y_new_mapped = y_new.map(label_mapping)

    num_new_classes = len(unique_new_labels)
    total_classes = num_classes + num_new_classes

    # 创建新数据加载器
    new_dataset = torch.utils.data.TensorDataset(torch.tensor(X_new.values, dtype=torch.float32),
                                                 torch.tensor(y_new_mapped.values, dtype=torch.long))
    new_data_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True)

    # 旧数据加载器（用于知识蒸馏）
    old_dataset = torch.utils.data.TensorDataset(torch.tensor(train_features.numpy(), dtype=torch.float32),
                                                 torch.tensor(train_labels.numpy(), dtype=torch.long))
    old_data_loader = torch.utils.data.DataLoader(old_dataset, batch_size=batch_size, shuffle=True)

    # 扩展分类器以适应新类别
    classifier = expand_classifier(classifier, num_new_classes).to(device)

    # 进行增量更新
    incremental_update(classifier, feature_extractor, new_data_loader, old_data_loader, device, epochs=5)

    # 保存更新后的模型
    torch.save(classifier.state_dict(), '/root/DOMR_torch/experiment/logs/classifier_incremental.pth')

    # 使用更新后的模型进行预测
    preds_incremental = aggregation(classifier, feature_extractor, X_test, device, total_classes)

    # 评估增量更新后的模型
    evaluate_model(preds_incremental, y_true)


if __name__ == '__main__':
    main()
