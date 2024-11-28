import torch
import pandas as pd
from utils.dataset import load_processed_data, create_dataloaders, create_episode
from models.representation import Representation
from models.meta_learning import MAML, train_maml_model
from models.classifier import Classifier, train_classifier
from utils.evaluation import evaluate_model
from utils.feature_extraction import extract_features


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_data_dir = 'data/raw'
    processed_data_dir = 'data/processed'
    file_name = 'cicmaldroid2020.csv'

    batch_size = 32
    inner_lr = 0.01
    outer_lr = 0.001
    inner_steps = 5
    meta_iterations = 10
    classifier_epochs = 10
    N = 7

    X_train, X_test, y_train, y_test,known_classes = load_processed_data(file_name, raw_data_dir, processed_data_dir,N)

    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=batch_size)

    input_dim = X_train.shape[1]
    hidden_dim = 256
    feature_dim = 128
    feature_extractor = Representation(input_dim, hidden_dim, feature_dim).to(device)

    episodes = create_episode(X_train, y_train, num_episode=100)

    maml_model = MAML(feature_extractor, inner_lr=inner_lr, outer_lr=outer_lr, inner_steps=inner_steps, device=device)
    train_maml_model(maml_model, episodes, num_iterations=meta_iterations)
    torch.save(feature_extractor.state_dict(), 'src/models/feature_extractor.pth')

    feature_extractor.load_state_dict(torch.load('src/models/feature_extractor.pth',weights_only=True))
    feature_extractor.eval()

    train_features, train_labels = extract_features(feature_extractor, train_loader, device)
    torch.save((train_features, train_labels), 'data/processed/train_features.pth')

    test_features, test_labels = extract_features(feature_extractor, test_loader, device)
    torch.save((test_features, test_labels), '/data/processed/test_features.pth')

    num_classes = len(set(y_train))
    classifier = Classifier(input_dim=feature_dim, num_classes=num_classes).to(device)
    train_classifier(classifier, train_features.numpy(), train_labels.numpy(), device, epochs=classifier_epochs)
    torch.save(classifier.state_dict(), 'src/models/classifier.pth')

    evaluate_model(classifier, test_features, test_labels, device)


if __name__ == '__main__':
    main()
