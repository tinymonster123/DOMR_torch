import torch
import torch.nn as nn
import torch.optim as optim

class Representation(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim):
        super(Representation, self).__init__()
        self.layer1 = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim,feature_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

    def functional_forward(self,x,params):
        x = nn.functional.linear(x,params[0],params[1])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x,params[2],params[3])
        return x

def center_distance_loss(features, labels):
    unique_labels = torch.unique(labels)
    loss = 0.0
    for label in unique_labels:
        class_features = features[labels == label]
        center = class_features.mean(dim=0)
        intra_class_loss = ((class_features - center) ** 2).sum() / class_features.size(0)
        loss += intra_class_loss
    return loss / len(unique_labels)

def meta_train(feature_extractor, episodes, inner_lr, outer_lr, inner_steps, iterations, device):
    feature_extractor.to(device)
    optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=outer_lr)

    for iteration in range(iterations):
        total_loss = 0.0
        for episode in episodes:
            support_set, support_labels, query_set, query_labels = episode
            support_set, support_labels = support_set.to(device), support_labels.to(device)
            query_set, query_labels = query_set.to(device), query_labels.to(device)

            fast_weights = list(feature_extractor.parameters())
            for _ in range(inner_steps):
                support_features = feature_extractor(support_set)
                loss = center_distance_loss(support_features, support_labels)
                grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = [w - inner_lr * g for w, g in zip(fast_weights, grads)]

            query_features = feature_extractor(query_set)
            query_loss = center_distance_loss(query_features, query_labels)
            optimizer.zero_grad()
            query_loss.backward()
            optimizer.step()

            total_loss += query_loss.item()
        average_loss = total_loss / len(episodes)
        print(f"Iteration [{iteration+1}/{iterations}], Loss: {average_loss:.4f}")
