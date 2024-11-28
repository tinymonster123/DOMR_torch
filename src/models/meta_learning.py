import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.representation import Representation,center_distance_loss

representation = Representation()

# 定义 MAML 算法 其中模型使用为 Representation
class MAML(nn.Module):
    def __init__(self, inner_lr,outer_lr,inner_steps,device):
        super(MAML, self).__init__()
        self.model = representation.to(device) # 特征提取器模型，使用了 LeNet5 (也就是 Representation)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.device = device
        self.outer_optimizer = optim.Adam(self.model.parameters(), lr=self.outer_lr) # 使用 Adam 优化器 学习率需按照论文中为 0.001
        
        def forward(self, x):
            return self.model(x.to(self.device))
    
    # 定义内部更新函数 对应 MAML 算法中的内部优化器    
    def inner_update(self,support_set,support_labels,known_classes,old_params=None):
            # 使用交叉熵损失函数
            criterion = nn.NLLLoss() # 使用负对数似然损失函数 适应模型最后一层 LogSoftmax
            
            # 根据论文中将 loss weights 都设置为 1
            lambda1 = 1.0
            lambda2 = 1.0
            
            new_index = [i for i,label in enumerate(support_labels) if label not in known_classes] # 新类别的索引
            old_index = [i for i,label in enumerate(support_labels) if label in known_classes] # 旧类别的索引    
                   
            # 对模型参数进行复制 用于内部更新参数不影响全局参数 以初始化快速权重
            fast_weights = [p.clone() for p in self.model.parameters()]
            
            support_set,support_labels = support_set.to(self.device),support_labels.to(self.device) # 将支持集和标签移动到指定设备
            
            # 进行多次更新 内部循环 模拟元学习在新任务上的快速学习
            for _ in range(self.inner_steps):
                output = self.model.functional_forward(support_set, fast_weights)
                
                # 根据论文中 l2 和 l1 都是针对于新类别
                if new_index: # 如果没有新类别则损失为 0 
                    output_new = output[new_index]
                    support_new = support_set[new_index]
                    labels_new = support_labels[new_index]
                    # 获取新类别的特征向量
                    features_new = self.model.get_feature_params(support_new,fast_weights)
                    # 对新类别进行中心损失计算作为 l1 函数(计算欧式距离)
                    loss_l1 = center_distance_loss(features_new,labels_new)
                    # 对新类别进行交叉熵损失计算作为 l2 函数
                    loss_l2 = criterion(output_new, labels_new)
                else:
                    loss_l2 = 0.0
                    loss_l1 = 0.0
                    
                # 根据论文中 l3 是针对于旧类别(这一步可以将旧的样本传递给模型，防止灾难性遗忘)
                if old_index and old_params is not None:
                    # 禁用梯度计算
                    with torch.no_grad():
                        old_output = self.model.functional_forward(support_set[old_index], old_params)
                        old_output_oldchosen = old_output[old_index]
                        soft_target = F.softmax(old_output_oldchosen, dim=1)
                        
                    output_old = output[old_index]
                    # 使用 KL 散度作为蒸馏损失 作为 l3 函数
                    loss_l3 = F.kl_div(F.log_softmax(output_old, dim=1), soft_target, reduction='batchmean')
                else:
                    loss_l3 = 0.0
                    
                loss = lambda1 * loss_l1 + lambda2 * loss_l2 + loss_l3
                
                # 使用自动微分来进行梯度计算，用来进行快速权重更新
                grads = torch.autograd.grad(loss, fast_weights, create_graph=True, allow_unused=True)
                fast_weights = [w - self.inner_lr * g if g is not None else w for w, g in zip(fast_weights, grads)]
            
            return fast_weights
    
    # 定义外部更新函数 对应 MAML 算法中的外部优化器    
    def outer_update(self,support_set,support_labels,query_set,query_labels):
            criterion = nn.CrossEntropyLoss()
            self.outer_optimizer.zero_grad()
            fast_weights = self.inner_update(support_set,support_labels)
            query_set,query_labels = query_set.to(self.device),query_labels.to(self.device)
            output = self.model.functional_forward(query_set, fast_weights)
            loss = criterion(output, query_labels)
            loss.backward()
            self.outer_optimizer.step()
            return loss.item()
        
    def meta_train(self, episodes):
            for episode in episodes:
                support_set, query_set, support_labels, query_labels = episode
                # fast_weights = self.inner_update(support_set,support_labels)
                loss = self.outer_update(support_set, support_labels, query_set, query_labels)
            return loss
        
def train_maml_model(model,episodes,num_iterations):
    for iteration in range(num_iterations):
        loss = model.meta_train(episodes)
        print(f'iteration: {iteration+1}/{num_iterations}, loss: {loss}')
        

    
    
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