import torch


"""Super class of a module from which the other will inherit."""
class Module(object):
    
    """Function to define the foward pass of the module."""
    def forward_pass(self, *_input_):
        raise NotImplementedError

    """Function to define the backward pass of the module."""
    def backward_pass(self, *gradwrtoutput):
        raise NotImplementedError
        
    """Function to acess the parameters of the module."""
    def param(self):
        return []
    
    """Function to update the module."""
    def update(self, eta):
        return [] 
    
    """Function to reset the module gradients."""
    def zerograd(self):
        return []    
    

"""Linear module of a neural network."""
class Linear(Module):
    
    """Function to initiate a linear module."""
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.Xavier_Normalization(in_features, out_features)
        
        self.dl_dw = torch.zeros(self.w.size())
        self.dl_db = torch.zeros(self.b.size())
        
        self.cache_input_act = None
    
    """Xavier Normalization used to initialize the weigths and bias."""
    def Xavier_Normalization(self, in_features, out_features):
        std_init_w = math.sqrt(2/(self.in_features + self.out_features))
        self.w = torch.empty(out_features,in_features).normal_(0,std_init_w)
        std_init_b = 1/math.sqrt(self.w.size(1))
        self.b = torch.empty(out_features).uniform_(-std_init_b,std_init_b)
    
    """Function to compute the forward pass of a linear module."""
    def forward_pass(self, _input_):
        self.cache_input_act = _input_
        s = _input_ @ self.w.t() + self.b 
        return s

    """Function to compute the backward pass of a linear module."""
    def backward_pass(self, gradwrtoutput):
        dl_dx =  gradwrtoutput @ self.w
        self.dl_dw =  gradwrtoutput.t() @ self.cache_input_act
        self.dl_db = torch.sum(gradwrtoutput, dim = 0)
        return dl_dx
    
    """Function to access the parameters of a linear module."""
    def param(self):
        return [(self.w, self.dl_dw), (self.b, self.dl_db)]
    
    """Function to update the weights and bias of a linear module."""
    def update(self, eta):
        self.w -= eta * self.dl_dw
        self.b -= eta * self.dl_db
        
    """Function to reset the gradients of a linear model."""
    def zerograd(self):
        self.dl_dw = torch.zeros(self.w.size())
        self.dl_db = torch.zeros(self.b.size())  


"""ReLU module of a neural network."""
class ReLU(Module):
    
    """Fonction to initiate a ReLU module."""
    def __init__(self):
        super(ReLU, self).__init__()
        self.cache_input_linear = None
    
    """ReLU function."""
    def ReLU_fun(self, _input_):
        return _input_ * (_input_ > 0).float()
    
    """Derivative of the ReLU function."""
    def d_ReLU(self, _input_):
        return 1. * (_input_ > 0).float()

    """Function to compute the forward pass of a ReLU module."""
    def forward_pass(self, _input_):
        self.cache_input_linear = _input_
        x = self.ReLU_fun(_input_) #100 x 25
        return x

    """Function to compute the forward pass of a ReLU module."""
    def backward_pass(self, gradwrtoutput):
        dl_ds =  self.d_ReLU(self.cache_input_linear) * gradwrtoutput # 100 x 2
        self.cache_input_linear = None
        return dl_ds
        
        
"""Tanh module of a neural network."""
class Tanh(Module):
    
    """Fonction to initiate a Tanh module."""
    def __init__(self):
        super(Tanh, self).__init__()
        self.cache_input_linear = None
        
    """Tanh function."""
    def Tanh_fun(self,_input_):
        return _input_.tanh()
    
    """Derivative of the Tanh function."""
    def d_Tanh(self,_input_):
        return (1 - torch.pow(self.Tanh_fun(_input_), 2))

    """Function to compute the forward pass of a Tanh module."""
    def forward_pass(self, _input_):
        self.cache_input_linear = _input_
        x = self.Tanh_fun(_input_) #100 x 25
        return x
   
    """Function to compute the backward pass of a Tanh module."""
    def backward_pass(self, gradwrtoutput):
        dl_ds =  self.d_Tanh(self.cache_input_linear) * gradwrtoutput # 100 x 2
        self.cache_input_linear = None
        return dl_ds


"""Sigmoid module of a neural network."""
class Sigmoid(Module):
    
    """Function to initiate a Sigmoid module."""
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.cache_input_linear = None
        
    """Sigmoid function."""
    def Sigmoid_fun(self,_input_):
        return 1/(1+torch.exp(-_input_))
    
    """Derivative of the Sigmoid function."""
    def d_Sigmoid(self,_input_):
        return torch.exp(-_input_)/torch.pow((1+torch.exp(-_input_)),2)

    """Function to compute the forward pass of a Sigmoid module."""
    def forward_pass(self, _input_):
        self.cache_input_linear = _input_
        x = self.Sigmoid_fun(_input_) #100 x 25
        return x
   
    """Function to compute the backward pass of a Sigmoid module."""
    def backward_pass(self, gradwrtoutput):
        dl_ds =  self.d_Sigmoid(self.cache_input_linear) * gradwrtoutput # 100 x 2
        self.cache_input_linear = None
        return dl_ds
    
    
"""Dropout module of a neural network."""
class Dropout(Module):
    
    "Function to initiate a Dropout module."
    def __init__(self, p = 0.5):
        super(Dropout, self).__init__()
        self.p = p
        self.prob = None
    
    "Function to turn off some units during the foward pass."
    def forward_pass(self, _input_):
        self.prob = torch.bernoulli((1-self.p) * torch.ones(_input_.size()[1]))/(1-self.p)
        x = _input_ * self.prob
        return x
   
    "Function to compute the gradient during the backward pass."
    def backward_pass(self, gradwrtoutput):
        return gradwrtoutput * self.prob 