import torch.nn as nn

def expand_classifier(classifier, num_new_classes):
    # 冻结原有的分类器参数，避免对旧类别的影响
    for param in classifier.parameters():
        param.requires_grad = False

    # 获取特征维度
    input_dim = classifier.input_dim

    # 创建新的分类器模块（针对新类别）
    new_classifiers = nn.ModuleList([
        nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.LogSoftmax(dim=1)
        ) for _ in range(num_new_classes)
    ])

    # 将新分类器添加到原有的分类器列表中
    classifier.classifiers.extend(new_classifiers)

    # 解除新分类器的参数冻结，使其可训练
    for param in new_classifiers.parameters():
        param.requires_grad = True

    # 更新分类器的类别数量
    classifier.num_classes += num_new_classes

    return classifier