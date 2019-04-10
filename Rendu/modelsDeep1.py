import torch
from torch import nn
from torch.nn import functional as F


""" DEEP 1 MODELS """


"""Model generation for deep model 1 (2 convolutional layers and 2 max pooling) without weight sharing and without auxiliary loss."""    
class Deep_NOsharing_NOaux(nn.Module):
    def __init__(self, hidden, act_fun):
        super(Deep_NOsharing_NOaux, self).__init__()
        self.act_fun = act_fun
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv1_2 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # After concatenation of the features from image 1 and image 2
        self.fc1 = nn.Linear(512, hidden)
        self.fc2 = nn.Linear(hidden,2)

    def forward(self, x):
        x1_1 = self.act_fun(F.max_pool2d(self.conv1_1(x[:,0,:,:].view(100,1,14,14)), kernel_size=2, stride=2))
        x2_1 = self.act_fun(F.max_pool2d(self.conv2_1(x1_1), kernel_size=2, stride=2))
        
        x1_2 = self.act_fun(F.max_pool2d(self.conv1_2(x[:,1,:,:].view(100,1,14,14)), kernel_size=2, stride=2))
        x2_2 = self.act_fun(F.max_pool2d(self.conv2_2(x1_2), kernel_size=2, stride=2))
        
        x = torch.cat([x2_1, x2_2],1)
        x = self.act_fun(self.fc1(x.view(-1, 512)))
        x = self.fc2(x)
        return x

    
"""Model generation for deep model 1 (2 convolutional layers and 2 max pooling) with weight sharing and without auxiliary loss."""    
class Deep_sharing_NOaux(nn.Module):
    def __init__(self, hidden, act_fun):
        super(Deep_sharing_NOaux, self).__init__()
        self.act_fun = act_fun
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # After concatenation of the features from image 1 and image 2
        self.fc1 = nn.Linear(512, hidden)
        self.fc2 = nn.Linear(hidden,2)

    def forward(self, x):
        conv_images = []
        for image in range(2):
            first_conv = self.act_fun(F.max_pool2d(self.conv1(x[:,image,:,:].view(100,1,14,14)), kernel_size=2, stride=2))
            conv_images.append(self.act_fun(F.max_pool2d(self.conv2(first_conv), kernel_size=2, stride=2)))
        
        x = torch.cat([conv_images[0], conv_images[1]],1)
        x = self.act_fun(self.fc1(x.view(-1, 512)))
        x = self.fc2(x)
        return x

    
"""Model generation for deep model 1 (2 convolutional layers and 2 max pooling) without weight sharing and with auxiliary loss."""    
class Deep_NOsharing_aux(nn.Module):
    def __init__(self, hidden, act_fun):
        super(Deep_NOsharing_aux, self).__init__()
        self.act_fun = act_fun
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv1_2 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # For classification with classes
        self.fc_aux1 = nn.Linear(256, 10)
        self.fc_aux2 = nn.Linear(256, 10)
        
        # After concatenation of the features from image 1 and image 2
        self.fc1 = nn.Linear(512, hidden)
        self.fc2 = nn.Linear(hidden,2)

    def forward(self, x):
        x1_1 = self.act_fun(F.max_pool2d(self.conv1_1(x[:,0,:,:].view(100,1,14,14)), kernel_size=2, stride=2))
        x2_1 = self.act_fun(F.max_pool2d(self.conv2_1(x1_1), kernel_size=2, stride=2))
        
        x1_2 = self.act_fun(F.max_pool2d(self.conv1_2(x[:,1,:,:].view(100,1,14,14)), kernel_size=2, stride=2))
        x2_2 = self.act_fun(F.max_pool2d(self.conv2_2(x1_2), kernel_size=2, stride=2))

        aux1 = F.softmax(self.fc_aux1(x2_1.view(-1,256)),1)
        aux2 = F.softmax(self.fc_aux2(x2_2.view(-1,256)),1)
        
        x = torch.cat([x2_1, x2_2],1)
        x = self.act_fun(self.fc1(x.view(-1, 512)))
        x = self.fc2(x)
        return x, aux1, aux2

    
"""Model generation for deep model 1 (2 convolutional layers and 2 max pooling) with weight sharing and with auxiliary loss."""    
class Deep_sharing_aux(nn.Module):
    def __init__(self, hidden, act_fun):
        super(Deep_sharing_aux, self).__init__()
        self.act_fun = act_fun
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # For classification with classes
        self.fc_aux1 = nn.Linear(256, 10)
        self.fc_aux2 = nn.Linear(256, 10)
        
        # After concatenation of the features from image 1 and image 2
        self.fc1 = nn.Linear(512, hidden)
        self.fc2 = nn.Linear(hidden,2)

    def forward(self, x):
        conv_images = []
        for image in range(2):
            first_conv = self.act_fun(F.max_pool2d(self.conv1(x[:,image,:,:].view(100,1,14,14)), kernel_size=2, stride=2))
            conv_images.append(self.act_fun(F.max_pool2d(self.conv2(first_conv), kernel_size=2, stride=2)))
            
        aux1 = F.softmax(self.fc_aux1(conv_images[0].view(-1,256)),1)
        aux2 = F.softmax(self.fc_aux2(conv_images[1].view(-1,256)),1)
        
        x = torch.cat([conv_images[0], conv_images[1]],1)
        x = self.act_fun(self.fc1(x.view(-1, 512)))
        x = self.fc2(x)
        return x, aux1, aux2