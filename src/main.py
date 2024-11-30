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
    input_dim = X_train.shape[1]
    hidden_dim = 256
    feature_dim = 128
    feature_extractor = Representation(input_dim, hidden_dim, feature_dim).to(device)
    
    # 创建 Episodes
    episodes = create_episode(X_train, y_train, num_episode=100)
    
    # 初始化 MAML 模型
    maml_model = MAML(feature_extractor, inner_lr=inner_lr, outer_lr=outer_lr, inner_steps=inner_steps, device=device)
    train_maml_model(maml_model, episodes, num_iterations=meta_iterations)
    torch.save(feature_extractor.state_dict(), '/root/DOMR_torch/src/models/feature_extractor.pth')

    feature_extractor.load_state_dict(torch.load('/root/DOMR_torch/src/models/feature_extractor.pth',weights_only=True))
    feature_extractor.eval()

    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    torch.save((train_features, train_labels), '/root/DOMR_torch/data/processed/train_features.pth')

    test_features, test_labels = extract_features(feature_extractor, test_loader, device)
    torch.save((test_features, test_labels), '/root/DOMR_torch/data/processed/test_features.pth')

    num_classes = len(set(y_train))
    classifier = Classifier(input_dim=feature_dim, num_classes=num_classes).to(device)
    train_classifier(classifier, train_features.numpy(), train_labels.numpy(), device, epochs=classifier_epochs)
    torch.save(classifier.state_dict(), '/root/DOMR_torch/src/models/classifier.pth')

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


if __name__ == '__main__':
    main()
