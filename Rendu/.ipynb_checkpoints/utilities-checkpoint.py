import torch
from modelsShallow import *
from modelsMLP import *
from modelsConv1 import *
from modelsConv2 import *
import dlc_practical_prologue as prologue


"""Function to train a model without auxiliary loss."""
def train_model_NOaux(model, train_input, train_target, nb_epochs, batch_size, criterion, eta): 
    
    optimizer = torch.optim.Adam(model.parameters(), lr = eta)
    for e in range(nb_epochs):
        if (e % 10 == 0 and e > 0):
            eta = eta/10
            optimizer = torch.optim.Adam(model.parameters(), lr = eta)
        for step_ in range(0,train_input.size(0),batch_size):                              
            output = model(train_input[step_:step_+batch_size])
            loss = criterion(output, train_target[step_:step_+batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
"""Function to train a model with auxiliary loss."""
def train_model_aux(model, train_input, train_target, train_classes, nb_epochs, batch_size, criterion, eta, lambda_):   
    
    optimizer = torch.optim.Adam(model.parameters(), lr = eta)
    for e in range(nb_epochs):
        if (e % 10 == 0 and e > 0):
            eta = eta/10
            optimizer = torch.optim.Adam(model.parameters(), lr = eta)
        for step_ in range(0,train_input.size(0),batch_size):                              
            output_target, output_im1, output_im2 = model(train_input[step_:step_+batch_size])
            loss_target = criterion(output_target, train_target[step_:step_+batch_size])
            loss_im1 = criterion(output_im1, train_classes[step_:step_+batch_size,0])
            loss_im2 = criterion(output_im2, train_classes[step_:step_+batch_size,1])
            loss = loss_target + lambda_*(loss_im1 + loss_im2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
  

"""Function to compute the number of error for a model without auxiliary loss."""   
def compute_nb_errors_NOaux(model, data_input, data_target, mini_batch_size): 
    
    nb_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1
    return nb_errors


"""Function to compute the number of error for a model with auxiliary loss."""   
def compute_nb_errors_aux(model, data_input, data_target, mini_batch_size): 
    
    nb_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        output,_,_ = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_errors = nb_errors + 1
    return nb_errors


"""Function to to train a specific model."""   
def model_training(input_, target, classes, hidden_units, eta, lambda_, model_type = 'Shallow', sub_model = 'NOsharing_NOaux', nb_epochs = 25, mini_batch_size = 100, criterion = nn.CrossEntropyLoss()):
    
    if(model_type == 'Shallow'):
        if(sub_model == 'NOsharing_NOaux'):
            model = Shallow_NOsharing_NOaux(hidden = hidden_units, act_fun = F.relu)
        elif(sub_model == 'sharing_NOaux'):
            model = Shallow_sharing_NOaux(hidden = hidden_units, act_fun = F.relu)
        elif(sub_model == 'NOsharing_aux'):
            model = Shallow_NOsharing_aux(hidden = hidden_units, act_fun = F.relu)
        elif(sub_model == 'sharing_aux'):
            model = Shallow_sharing_aux(hidden = hidden_units, act_fun = F.relu)
    elif(model_type == 'MLP'):
        if(sub_model == 'NOsharing_NOaux'):
            model = MLP_NOsharing_NOaux(hidden = hidden_units, act_fun = F.relu)
        elif(sub_model == 'sharing_NOaux'):
            model = MLP_sharing_NOaux(hidden = hidden_units, act_fun = F.relu)
        elif(sub_model == 'NOsharing_aux'):
            model = MLP_NOsharing_aux(hidden = hidden_units, act_fun = F.relu)
        elif(sub_model == 'sharing_aux'):
            model = MLP_sharing_aux(hidden = hidden_units, act_fun = F.relu)
    elif(model_type == 'Conv1'):
        if(sub_model == 'NOsharing_NOaux'):
            model = Conv_NOsharing_NOaux(hidden = hidden_units, act_fun = F.relu)
        elif(sub_model == 'sharing_NOaux'):
            model = Conv_sharing_NOaux(hidden = hidden_units, act_fun = F.relu)
        elif(sub_model == 'NOsharing_aux'):
            model = Conv_NOsharing_aux(hidden = hidden_units, act_fun = F.relu)
        elif(sub_model == 'sharing_aux'):
            model = Conv_sharing_aux(hidden = hidden_units, act_fun = F.relu)
    elif(model_type == 'Conv2'):
        if(sub_model == 'NOsharing_NOaux'):
            model = Conv_NOsharing_NOaux2(hidden = hidden_units, act_fun = F.relu)
        elif(sub_model == 'sharing_NOaux'):
            model = Conv_sharing_NOaux2(hidden = hidden_units, act_fun = F.relu)
        elif(sub_model == 'NOsharing_aux'):
            model = Conv_NOsharing_aux2(hidden = hidden_units, act_fun = F.relu)
        elif(sub_model == 'sharing_aux'):
            model = Conv_sharing_aux2(hidden = hidden_units, act_fun = F.relu)
                
    if(sub_model == 'NOsharing_aux' or sub_model == 'sharing_aux'): 
        train_model_aux(model, input_[:700], target[:700], classes[:700], nb_epochs, mini_batch_size, criterion, eta, lambda_)
        accuracy = 1 - compute_nb_errors_aux(model, input_[700:], target[700:], mini_batch_size)/len(target[700:])
    else: 
        train_model_NOaux(model, input_[:700], target[:700], nb_epochs, mini_batch_size, criterion, eta)
        accuracy = 1 - compute_nb_errors_NOaux(model, input_[700:], target[700:], mini_batch_size)/len(target[700:])
    
    return accuracy, model


"""Function to generate dictionary where to stock the hyper parameters."""
def create_dict():
    results = {'Shallow':{'NOsharing_NOaux':{}, 'sharing_NOaux':{}, 'NOsharing_aux':{}, 'sharing_aux':{}},
               'MLP':{'NOsharing_NOaux':{}, 'sharing_NOaux':{}, 'NOsharing_aux':{}, 'sharing_aux':{}},  
               'Conv1':{'NOsharing_NOaux':{}, 'sharing_NOaux':{}, 'NOsharing_aux':{}, 'sharing_aux':{}},
               'Conv2':{'NOsharing_NOaux':{}, 'sharing_NOaux':{}, 'NOsharing_aux':{}, 'sharing_aux':{}}  
              }
    return results


"""Fuction to fill the dictionary with the results."""
def fill_results(results, type_model, sub_model, acc, eta, hidden, lambda_):
    results[type_model][sub_model]['Acc'] = acc
    results[type_model][sub_model]['eta'] = eta
    results[type_model][sub_model]['hidden'] = hidden
    if(sub_model == 'NOsharing_aux' or sub_model == 'sharing_aux'): 
        results[type_model][sub_model]['lambda'] = lambda_
    return results


"""Function to perform a grid search over hyper-parameters to find the combination that gives the best accuracy"""
def grid_search_(lambdas, etas, hidden_units, train_input, train_target, train_classes, test_input, test_target):
    type_models = ['Shallow', 'MLP', 'Conv1', 'Conv2']
    sub_models = ['NOsharing_NOaux', 'sharing_NOaux', 'NOsharing_aux', 'sharing_aux']
    acc_test = torch.zeros(len(type_models),len(sub_models))
    results = create_dict()
    
    i = 0

    for t, type_model in enumerate(type_models):
        for s, sub_model in enumerate(sub_models):
            i += 1
            print('Getting hyper-parameters for architecture', i, '/ 16...')
            performances = torch.zeros(len(lambdas),len(hidden_units),len(etas))
            for l, lambda_ in enumerate(lambdas):
                for h, hidden in enumerate(hidden_units):
                    for e, eta in enumerate(etas):
                        acc, _ = model_training(train_input, train_target, train_classes, hidden.item(), eta.item(), 
                                              lambda_.item(), model_type = type_model, 
                                              sub_model = sub_model)
                        performances[l,h,e] = acc
            best_performance = torch.max(performances)
            best_idx = (performances == best_performance).nonzero();
            
            best_eta = etas[best_idx[0,2]].item()
            best_hidden = hidden_units[best_idx[0,1]].item()
            best_lambda = lambdas[best_idx[0,0]].item()
                
            results = fill_results(results, type_model, sub_model, best_performance.item(), 
                                       best_eta, best_hidden, best_lambda)
    return results