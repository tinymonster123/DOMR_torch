import torch
import pandas as pd
import numpy as np

def aggregation(test,representation_network,classifiers,device,num_classes):
    classifiers.eval()
    representation_network.eval() # 将分类器模型和 representation 网络模型设置为评估模式
    unknown_label = num_classes
    
    with torch.no_grad(): # 禁用梯度计算，来提高推理速度
        '''
        这部分代码是为了将 test 数据转化为 torch tensor 类型的数据，并将其移动到 GPU 上
        所以需要对 test 数据的类型进行判断，来进行合适的处理
        '''
        
        if isinstance(test,pd.DataFrame):
            test = torch.tensor(test.values, dtype=torch.float32).to(device)
        elif isinstance(test,np.ndarray):
            test = torch.tensor(test, dtype=torch.float32).to(device)
        elif isinstance(test,pd.Series):
            test = torch.tensor(test.values, dtype=torch.float32).to(device)
        elif isinstance(test,torch.Tensor):
            test = test.to(device)
        else:
            raise ValueError('Unsupported data type') 
        
        # 使用 representation 网络对 test 数据进行特征提取
        features = representation_network(test)
        
        outputs = classifiers(features) # 因为经过 classifier 处理后 outputs 的输出形状为 [batch_size,num_classes,2] 其中 2 代表分类器的输出维度
        outputs = torch.exp(outputs) # 将 LogSoftmax 处理的输出进行转换为 softmax
        
        # 将分类器进行正类预测
        positive_odds = outputs[:,:,1] # 提取每个分类器的正类概率
        
        # 获取每个分类器的预测类别
        _, predicted_labels = torch.max(outputs, dim=2)
        
        # 将样本进行判断是否为负类
        negative_judgement = torch.all(predicted_labels == 0, dim=1) # 类型为布尔张量
        # 对预测标签进行初始化
        preds = torch.zeros(outputs.size(0),dtype=torch.long).to(device)
        # 对判定为负类的样本进行标签为未知类型
        preds[negative_judgement] = unknown_label
        
        known_index = (~negative_judgement).nonzero(as_tuple=True)[0] #获取所有被至少一个分类器判定为正类的样本的索引
        if known_index.numel() > 0: # 返回张量中的元素个数，检测是否存在至少一个正类样本
            known_positive_odds = positive_odds[known_index] 
            max_odds,max_classes = torch.max(known_positive_odds,dim=1) # 选择最大概率的类别
            preds[known_index] = max_classes
            
        preds = preds.cpu().numpy()
        
    return preds
            
        
            
            
            
    
    