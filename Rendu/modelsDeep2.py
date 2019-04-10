import torch
from torch import nn
from torch.nn import functional as F


""" DEEP 2 MODELS """
    

"""Model generation for deep model 2 (4 convolutional layers and 1 max pooling) without weight sharing and without auxiliary loss."""    
class Deep_NOsharing_NOaux2(nn.Module):
    def __init__(self, hidden, act_fun):
        super(Deep_NOsharing_NOaux2, self).__init__()
        self.act_fun = act_fun
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv1_2 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3_2 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=2)
        self.conv4_2 = nn.Conv2d(64, 128, kernel_size=2)
        
        # After concatenation of the features from image 1 and image 2
        self.fc1 = nn.Linear(1024, hidden)
        self.fc2 = nn.Linear(hidden,2)

    def forward(self, x):
        x1_1 = self.act_fun(self.conv1_1(x[:,0,:,:].view(100,1,14,14)))
        x2_1 = self.act_fun(self.conv2_1(x1_1))
        x3_1 = self.act_fun(self.conv3_2(x2_1))
        x4_1 = self.act_fun(F.max_pool2d(self.conv4_1(x3_1), kernel_size=2, stride=2))
        
        x1_2 = self.act_fun(self.conv1_2(x[:,1,:,:].view(100,1,14,14)))
        x2_2 = self.act_fun(self.conv2_2(x1_2))
        x3_2 = self.act_fun(self.conv3_2(x2_2))
        x4_2 = self.act_fun(F.max_pool2d(self.conv4_2(x3_2), kernel_size=2, stride=2))

        x = torch.cat([x4_1, x4_2],1)
        x = self.act_fun(self.fc1(x.view(-1, 1024)))
        x = self.fc2(x)
        return x

    
"""Model generation for deep model 2 (4 convolutional layers and 1 max pooling) with weight sharing and without auxiliary loss."""  
class Deep_sharing_NOaux2(nn.Module):
    def __init__(self, hidden, act_fun):
        super(Deep_sharing_NOaux2, self).__init__()
        self.act_fun = act_fun
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        
        # After concatenation of the features from image 1 and image 2
        self.fc1 = nn.Linear(1024, hidden)
        self.fc2 = nn.Linear(hidden,2)

    def forward(self, x):
        conv_images = []
        for image in range(2):
            x1 = self.act_fun(self.conv1(x[:,image,:,:].view(100,1,14,14)))
            x2 = self.act_fun(self.conv2(x1))
            x3 = self.act_fun(self.conv3(x2))
            x4 = self.act_fun(F.max_pool2d(self.conv4(x3), kernel_size=2, stride=2))
            conv_images.append(x4)
        
        x = torch.cat([conv_images[0], conv_images[1]],1)
        x = self.act_fun(self.fc1(x.view(-1, 1024)))
        x = self.fc2(x)
        return x

    
"""Model generation for deep model 2 (4 convolutional layers and 1 max pooling) without weight sharing and with auxiliary loss."""  
class Deep_NOsharing_aux2(nn.Module):
    def __init__(self, hidden, act_fun):
        super(Deep_NOsharing_aux2, self).__init__()
        self.act_fun = act_fun
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv1_2 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3_2 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=2)
        self.conv4_2 = nn.Conv2d(64, 128, kernel_size=2)
        
        # For classification with classes
        self.fc_aux1 = nn.Linear(512, 10)
        self.fc_aux2 = nn.Linear(512, 10)
        
        # After concatenation of the features from image 1 and image 2
        self.fc1 = nn.Linear(1024, hidden)
        self.fc2 = nn.Linear(hidden,2)

    def forward(self, x):
        x1_1 = self.act_fun(self.conv1_1(x[:,0,:,:].view(100,1,14,14)))
        x2_1 = self.act_fun(self.conv2_1(x1_1))
        x3_1 = self.act_fun(self.conv3_2(x2_1))
        x4_1 = self.act_fun(F.max_pool2d(self.conv4_1(x3_1), kernel_size=2, stride=2))
        
        x1_2 = self.act_fun(self.conv1_2(x[:,1,:,:].view(100,1,14,14)))
        x2_2 = self.act_fun(self.conv2_2(x1_2))
        x3_2 = self.act_fun(self.conv3_2(x2_2))
        x4_2 = self.act_fun(F.max_pool2d(self.conv4_2(x3_2), kernel_size=2, stride=2))

        aux1 = F.softmax(self.fc_aux1(x4_1.view(-1,512)),1)
        aux2 = F.softmax(self.fc_aux2(x4_2.view(-1,512)),1)
        
        x = torch.cat([x4_1, x4_2],1)
        x = self.act_fun(self.fc1(x.view(-1, 1024)))
        x = self.fc2(x)
        return x, aux1, aux2

    
"""Model generation for deep model 2 (4 convolutional layers and 1 max pooling) with weight sharing and with auxiliary loss."""  
class Deep_sharing_aux2(nn.Module):
    def __init__(self, hidden, act_fun):
        super(Deep_sharing_aux2, self).__init__()
        self.act_fun = act_fun
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        
        # For classification with classes
        self.fc_aux1 = nn.Linear(512, 10)
        self.fc_aux2 = nn.Linear(512, 10)
        
        # After concatenation of the features from image 1 and image 2
        self.fc1 = nn.Linear(1024, hidden)
        self.fc2 = nn.Linear(hidden,2)

    def forward(self, x):
        conv_images = []
        for image in range(2):
            x1 = self.act_fun(self.conv1(x[:,image,:,:].view(100,1,14,14)))
            x2 = self.act_fun(self.conv2(x1))
            x3 = self.act_fun(self.conv3(x2))
            x4 = self.act_fun(F.max_pool2d(self.conv4(x3), kernel_size=2, stride=2))
            conv_images.append(x4)
            
        aux1 = F.softmax(self.fc_aux1(conv_images[0].view(-1,512)),1)
        aux2 = F.softmax(self.fc_aux2(conv_images[1].view(-1,512)),1)
        
        x = torch.cat([conv_images[0], conv_images[1]],1)
        x = self.act_fun(self.fc1(x.view(-1, 1024)))
        x = self.fc2(x)
        return x, aux1, aux2