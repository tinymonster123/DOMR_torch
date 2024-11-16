import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_classes)
        ])

    def forward(self, x):
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(x))
        outputs = torch.cat(outputs, dim=1)
        return outputs

def train_classifier(model, X_train, y_train, device, epochs):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(epochs):
        data = torch.tensor(X_train, dtype=torch.float32).to(device)
        labels = torch.tensor(y_train, dtype=torch.long).to(device)
        
        features = data

        binary_labels = torch.zeros(labels.size(0), model.num_classes).to(device)
        for index in range(model.num_classes):
            binary_labels[:, index] = (labels == index).float()
        outputs = model(features)
        loss = criterion(outputs, binary_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
