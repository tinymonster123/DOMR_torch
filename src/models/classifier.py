import torch
import torch.nn as nn
from main import logger

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        # 通过循环创建多个分类器，实现 one-vs-rest 分类器
        self.classifiers = nn.ModuleList([ # 包含多个二类分类器
            nn.Sequential(
                nn.Linear(input_dim, 128), 
                nn.ReLU(),
                nn.Linear(128, 16),
                nn.ReLU(),
                nn.Linear(16, 2), # 修改输出维度为 2 以适应 LogSoftmax 输入参数要求
                nn.LogSoftmax(dim=1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x):
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(x).unsqueeze(1)) # 将输出结果添加到 outputs 列表中
        outputs = torch.cat(outputs, dim=1) # 将 outputs 列表中的 tensor 按列拼接为一个 tensor
        return outputs

def train_classifier(model, X_train, y_train, device, epochs):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train() # train 是一个 nn.Module 类的方法，用于将模型设置为训练模式

    for epoch in range(epochs):
        data = torch.tensor(X_train, dtype=torch.float32).to(device)
        labels = torch.tensor(y_train, dtype=torch.long).to(device)
        
        features = data

        binary_labels = torch.zeros(labels.size(0), model.num_classes).to(device)
        
        total_loss = 0.0
        for index in range(model.num_classes):
            binary_labels = (labels == index).long()
            output = model.classifiers[index](features)    
            loss = criterion(output, binary_labels)
            total_loss += loss # 对 loss 进行累加(值得一提的是 loss 是 tensor 类型而不是一个简单的 float)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        logger.info(f'Epoch: {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}')
