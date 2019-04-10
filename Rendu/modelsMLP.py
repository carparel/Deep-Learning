import torch
from torch import nn
from torch.nn import functional as F


""" MLP MODELS """
    
    
"""Model generation for MLP model (2 hidden layers) without weight sharing and without auxiliary loss."""        
class MLP_NOsharing_NOaux(nn.Module):
    def __init__(self, hidden, act_fun):
        super(MLP_NOsharing_NOaux, self).__init__()
        self.act_fun = act_fun
        self.fc1_1 = nn.Linear(196, hidden)
        self.fc1_2 = nn.Linear(196, hidden)
        self.fc2_1 = nn.Linear(hidden,hidden)
        self.fc2_2 = nn.Linear(hidden,hidden)
        # After concatenation of the features from image 1 and image 2
        self.fc3 = nn.Linear(hidden*2,2)

    def forward(self, x):
        x1_1 = self.act_fun(self.fc1_1(x[:,0,:,:].view(-1,196)))
        x1_2 = self.act_fun(self.fc1_2(x[:,1,:,:].view(-1,196)))
        x2_1 = self.act_fun(self.fc2_1(x1_1))
        x2_2 = self.act_fun(self.fc2_2(x1_2))
        x = torch.cat([x2_1, x2_2],1)
        x = self.fc3(x)       
        return x

    
"""Model generation for MLP model (2 hidden layers) with weight sharing and without auxiliary loss.""" 
class MLP_sharing_NOaux(nn.Module):
    def __init__(self, hidden, act_fun):
        super(MLP_sharing_NOaux, self).__init__()
        self.act_fun = act_fun
        self.fc1 = nn.Linear(196, hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        # After concatenation of the features from image 1 and image 2
        self.fc3 = nn.Linear(hidden*2,2)

    def forward(self, x):
        fc_image = []
        for image in range(2):
            x1 = self.act_fun(self.fc1(x[:,image,:,:].view(-1,196)))
            x2 = self.act_fun(self.fc2(x1))
            fc_image.append(x2)
        x = torch.cat([fc_image[0],fc_image[1]],1)
        x = self.fc3(x)       
        return x

    
"""Model generation for MLP model (2 hidden layers) without weight sharing and with auxiliary loss.""" 
class MLP_NOsharing_aux(nn.Module):
    def __init__(self, hidden, act_fun):
        super(MLP_NOsharing_aux, self).__init__()
        self.act_fun = act_fun
        self.fc1_1 = nn.Linear(196, hidden)
        self.fc1_2 = nn.Linear(196, hidden)
        self.fc2_1 = nn.Linear(hidden,hidden)
        self.fc2_2 = nn.Linear(hidden,hidden)
        
        # For classification with classes
        self.fc_aux1 = nn.Linear(hidden, 10)
        self.fc_aux2 = nn.Linear(hidden, 10)
        
        # After concatenation of the features from image 1 and image 2
        self.fc3 = nn.Linear(hidden*2,2)

    def forward(self, x):
        x1_1 = self.act_fun(self.fc1_1(x[:,0,:,:].view(-1,196)))
        x1_2 = self.act_fun(self.fc1_2(x[:,1,:,:].view(-1,196)))
        x2_1 = self.act_fun(self.fc2_1(x1_1))
        x2_2 = self.act_fun(self.fc2_2(x1_2))
        
        aux1 = F.softmax(self.fc_aux1(x2_1),1)
        aux2 = F.softmax(self.fc_aux2(x2_2),1)
        
        x = torch.cat([x2_1, x2_2],1)
        x = self.fc3(x)       
        return x, aux1, aux2

    
"""Model generation for MLP model (2 hidden layers) with weight sharing and with auxiliary loss.""" 
class MLP_sharing_aux(nn.Module):
    def __init__(self, hidden, act_fun):
        super(MLP_sharing_aux, self).__init__()
        self.act_fun = act_fun
        self.fc1 = nn.Linear(196, hidden)
        self.fc2 = nn.Linear(hidden,hidden)
        
        # For classification with classes
        self.fc_aux1 = nn.Linear(hidden, 10)
        self.fc_aux2 = nn.Linear(hidden, 10)
        
        # After concatenation of the features from image 1 and image 2
        self.fc3 = nn.Linear(hidden*2,2)

    def forward(self, x):
        fc_image = []
        for image in range(2):
            x1 = self.act_fun(self.fc1(x[:,image,:,:].view(-1,196)))
            x2 = self.act_fun(self.fc2(x1))
            fc_image.append(x2)
            
        aux1 = F.softmax(self.fc_aux1(fc_image[0]),1)
        aux2 = F.softmax(self.fc_aux2(fc_image[1]),1)
        
        x = torch.cat([fc_image[0],fc_image[1]],1)
        x = self.fc3(x)       
        return x, aux1, aux2