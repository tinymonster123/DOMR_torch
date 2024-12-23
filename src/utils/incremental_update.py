import torch
import torch.nn as nn
from main import logger
import torch.nn.functional as F

def incremental_update(classifier, feature_extractor, new_data_loader, old_data_loader, device, epochs=5):
    # 设置模型为训练模式
    classifier.train()
    feature_extractor.eval()  # 冻结特征提取器

    # 定义损失函数和优化器
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=0.001)

    # 准备知识蒸馏所需的旧模型输出
    old_outputs = []
    for data, _ in old_data_loader:
        data = data.to(device)
        with torch.no_grad():
            output = classifier(data)
            old_outputs.append(output.detach())
    old_outputs = torch.cat(old_outputs, dim=0)

    # 训练循环
    for epoch in range(epochs):
        total_loss = 0.0
        # 将新旧数据加载器组合起来
        for (new_data, new_labels), (old_data, _) in zip(new_data_loader, old_data_loader):
            new_data, new_labels = new_data.to(device), new_labels.to(device)
            old_data = old_data.to(device)

            optimizer.zero_grad()
            
            # 处理新数据(其中数据已经是处理好的)
            new_output = classifier(new_data)
            # 计算新数据的损失
            loss_new = criterion(new_output, new_labels)

            # 处理旧数据
            old_output = classifier(old_data)
            # 使用知识蒸馏损失（KL 散度）
            loss_kd = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(old_output, dim=1),
                F.softmax(old_outputs[:old_output.size(0)], dim=1)
                )

            # 总损失
            loss = loss_new + loss_kd
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(new_data_loader)
        logger.info(f'Incremental Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')