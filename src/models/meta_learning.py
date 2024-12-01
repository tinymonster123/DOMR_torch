import torch
import torch.nn as nn
import torch.optim as optim
from main import logger
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
        self.criterion = nn.CrossEntropyLoss() # 使用交叉熵损失函数
        
        def forward(self, x):
            return self.model(x.to(self.device))
    
    # 定义内部更新函数 对应 MAML 算法中的内部优化器    
    def inner_update(self,support_set,support_labels,known_classes,old_params=None):
            # 使用交叉熵损失函数
            criterion = nn.NLLLoss() # 使用负对数似然损失函数 适应模型最后一层 LogSoftmax
            
            # 根据论文中将 loss weights 都设置为 1
            lambda1 = 1.0
            lambda2 = 1.0
            
            if support_set.dim() == 2:
                support_set = support_set.view(-1,1,10,47)
            
            new_index = [i for i,label in enumerate(support_labels) if label not in known_classes] # 新类别的索引
            old_index = [i for i,label in enumerate(support_labels) if label in known_classes] # 旧类别的索引    
                   
            # 对模型参数进行复制 用于内部更新参数不影响全局参数 以初始化快速权重
            fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
            
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
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=False, allow_unused=True)
                fast_weights = {name: param - self.inner_lr * grad if grad is not None else param
                            for (name, param), grad in zip(fast_weights.items(), grads)}
            return fast_weights
    
    # 定义外部更新函数 对应 MAML 算法中的外部优化器 
    # 将 outer_update 函数进行修改 直接在内部进行多个 episode 的训练 以及区分对已知类别和未知类别的操作
    '''
    输入 batch_episodes : 为含有一个 batch 的 episode 数据
    输入 known_classes : 为已知类别数据
    在外部优化器中针对的数据集为查询集
    '''
    def outer_update(self,batch_episodes,known_classes):
        total_loss = 0.0
        self.outer_optimizer.zero_grad()
        
        if isinstance(known_classes, torch.Tensor):
            known_classes = known_classes.tolist()
        else:
            known_classes = [kc.item() if isinstance(kc, torch.Tensor) else kc for kc in known_classes]
        
        for episode in batch_episodes:
            support_set, query_set, support_labels, query_labels = episode
            fast_weights = self.inner_update(support_set,support_labels,known_classes)
            
            # 对 query_set 进行维度变换
            if query_set.dim() == 2:
                query_set = query_set.view(-1,1,10,47)
                 
            
            # 外部优化器中需要计算已知类别和未知类别的损失
            query_set,query_labels = query_set.to(self.device),query_labels.to(self.device)
            
            # 获得查询集的输出结果
            output = self.model.functional_forward(query_set,fast_weights) # 使用的是为 log_softmax 输出
            output_odds = torch.exp(output) # log_softmax 输出转换为 softmax 输出
            
            # 使用 torch.isin 获取已知和未知类别索引
            known_classes_tensor = torch.tensor(known_classes, device=self.device)
            is_known = torch.isin(query_labels, known_classes_tensor)
        
            known_index = torch.nonzero(is_known, as_tuple=False).squeeze()
            unknown_index = torch.nonzero(~is_known, as_tuple=False).squeeze()
            
            if known_classes.numel() > 0:
                output_known = output[known_index]
                query_labels_known = query_labels[known_index]
                loss_lk = self.criterion(output_known, query_labels_known)
            else:
                loss_lk = 0.0
                
            # 此步骤是计算未知类别的输出在已知类别上的概率分布的熵，从而来帮助模型在未知概率上进行低概率的输出来增强模型拒绝未知类别的能力
            if unknown_index.numel() > 0:
                unknown_odds = output_odds[unknown_index]
                # 
                known_odds = unknown_odds[:,:len(known_classes)]
                
                loss_lu = -torch.sum(known_odds * torch.log(known_odds + 1e-10),dim=1).mean()
            else:
                loss_lu = 0.0
                
            total_loss += loss_lk + loss_lu
                
        loss_average = total_loss / len(batch_episodes)
        
        # 对平均损失进行反向传播，以更新模型参数
        loss_average.backward()
        self.outer_optimizer.step()
        
        torch.cuda.empty_cache()
        
        return loss_average.item()        
        
    # 定义元训练，使用外部更新函数进行多次迭代
    def meta_train(self, episodes,known_classes):
            loss = self.outer_update(episodes,known_classes)
            return loss
        
def train_maml_model(model,episodes,num_iterations,known_classes):
    for iteration in range(num_iterations):
        loss = model.meta_train(episodes,known_classes)
        logger.info(f'iteration: {iteration+1}/{num_iterations}, loss: {loss}')
        torch.cuda.empty_cache()
        
