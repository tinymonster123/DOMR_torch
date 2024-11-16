import torch
import torch.nn as nn
import torch.optim as optim

class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,stride=1)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self,x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        
        x = x.view(-1, 64*5*5)
        x = self.fc1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = torch.log_softmax(x, dim=1)
        
        return x


class MAML(nn.Module):
    def __init__(self, model,inner_lr,outer_lr,inner_steps,device):
        super(MAML, self).__init__()
        self.model = model.to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.device = device
        self.outer_optimizer = optim.Adam(self.model.parameters(), lr=self.outer_lr)
        
        def forward(self, x):
            return self.model(x.to(self.device))
        
        # def inner_update(self,support_set,support_labels):