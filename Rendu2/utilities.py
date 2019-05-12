import torch


"""Function to generate a circular dataset."""
def generate_disc_set(nb):
    input_ = torch.empty(nb, 2).uniform_(0,1)
    target_ = torch.empty(nb,2)
    
    for i in range(nb):
        if (torch.norm(input_[i]) < math.sqrt(1/(2*math.pi))):
            target_[i,0] = 0
            target_[i,1] = 1
        else : 
            target_[i,0] = 1
            target_[i,1] = 0
    return input_, target_


"""Function to compute the MSE loss between two matrices."""
def LossMSE(v, t):
    return torch.sum(torch.pow(t - v, 2)).item()


"""Function to compute the derivative of the MSE loss between two matrices."""
def d_LossMSE(v, t):
    return (2*(v-t))


"""Function to train a given model on a training using SGD."""
def train(model, train_input, train_target, eta, batch_size, epochs):
    losses = []
    for e in range(epochs):
        batch_loss = 0 
        for input_, target_ in zip(train_input.split(batch_size), train_target.split(batch_size)):
            output_ = model.forward_pass(input_)
            batch_loss += LossMSE(output_, target_)
            model.backward_pass(target_)
            model.update(eta)
            model.zerograd()
        losses.append(batch_loss)
    return losses


"""Function to compute the number of error of a classification output."""
def error(output, target):
    errors = 0
    indices = []
    for i in range(target.size()[0]):
        if (torch.argmax(output[i,:]) != torch.argmax(target[i,:])):
            errors += 1
            indices.append(i)
    return errors, indices