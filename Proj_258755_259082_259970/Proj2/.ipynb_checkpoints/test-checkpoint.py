from utilities import *
from classes import *

import torch
torch.set_grad_enabled(False)


# DATA GENERATION
print('Generating data...')
train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

print('Size of the input = ', train_input.size())
print('Size of the target = ', train_target.size())

# MODEL TRAINING
print('')
print('Training model...')
eta = 0.01
mini_batch_size = 100
epochs = 50

model = Sequential(Linear(2,25), ReLU(), Linear(25,25), Tanh(), Dropout(0.5), Linear(25,25), ReLU(), Linear(25,2), Sigmoid())
losses = train(model, train_input, train_target, eta, mini_batch_size, epochs)

# MODEL TESTING
print('')
print('Testing model...')
output_train = model.forward_pass(train_input, training = False)
output_test = model.forward_pass(test_input, training = False)

errors_train, indices_train = error(output_train, train_target)
print('Train error :', errors_train/10, '%')
errors_test, indices_test = error(output_test, test_target)
print('Test error :', errors_test/10, '%')

