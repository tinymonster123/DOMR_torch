import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict # 有序字典，用于按插入顺序存储字典

# The representation network is built on the classic convolutional neural network (CNN) LeNet5 .
# 使用 CNN LeNet5 搭建一个 Representation NetWork
class Representation(nn.Module):
    def __init__(self):
        super(Representation, self).__init__()
        
        # 将多个层组合成一个顺序容器
        self.conv_net = nn.Sequential(OrderedDict([              
            ('C1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('Tanh1', nn.Tanh()), # 每个 Tanh 激活函数都是为了增加模型的非线性，用于帮助模型捕捉输入数据的复杂性
            ('S2', nn.AvgPool2d(kernel_size=(2, 2), stride=2)),
            
            ('C3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('Tanh3', nn.Tanh()),
            ('S4', nn.AvgPool2d(kernel_size=(2, 2), stride=2)),
            
            ('C5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('Tanh5', nn.Tanh()),
        ]))
        
        # 定义全连接层
        self.fully_connected = nn.Sequential(OrderedDict([
            ('F6', nn.Linear(120, 84)),
            ('Tanh6', nn.Tanh()),
            ('F7', nn.Linear(84, 10)),
            ('LogSoftmax', nn.LogSoftmax(dim=-1)) # 表示在最后一个维度进行 softmax
        ]))

        
    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1) # 将多维张量展平为一维张量
        x = self.fully_connected(x)
        return x # 返回的模型输出结果，是一个对数概率
    
    # 使用给定的参数进行前向传播，用于元学习中的快速权重更新
    def get_feature_params(self,x,params):
        
        x = F.conv2d(x,params['conv_net.0.weight'],params['conv_net.0.bias'])
        x = F.tanh(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2), stride=2)
        
        x = F.conv2d(x,params['conv_net.3.weight'],params['conv_net.3.bias'])
        x = F.tanh(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2), stride=2)
        
        x = F.conv2d(x,params['conv_net.6.weight'],params['conv_net.6.bias'])
        x = F.tanh(x)
        
        x = x.view(x.size(0), -1)
        return x # 返回的是卷积层提取的特征向量
        

    # 使用给定的参数进行前向传播，使用于元学习中的快速权重更新
    def functional_forward(self,x,params):
        
        x = self.get_feature_params(x,params)
        
        x = F.linear(x,params['fully_connected.0.weight'],params['fully_connected.0.bias'])
        x = F.tanh(x)
        x = F.linear(x,params['fully_connected.2.weight'],params['fully_connected.2.bias'])
        x = F.log_softmax(x,dim=-1)
        return x #
        
        
# 定义中心损失函数
def center_distance_loss(features, labels):
    unique_labels = torch.unique(labels)
    loss = 0.0
    for label in unique_labels:
        class_features = features[labels == label]
        center = class_features.mean(dim=0)
        intra_class_loss = ((class_features - center) ** 2).sum() / class_features.size(0)
        loss += intra_class_loss
    return loss / len(unique_labels)


