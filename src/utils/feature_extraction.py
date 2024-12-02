import torch

def extract_features(feature_extractor, data_loader, device):
    feature_extractor.eval() 
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            features = feature_extractor.extract_features(data)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_features, all_labels