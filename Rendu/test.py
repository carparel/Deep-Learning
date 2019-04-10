from utilities import *
import torch


# Instanciate the models to be tested and get the hyper-parameters
type_models = ['Shallow', 'MLP', 'Deep1', 'Deep2']
sub_models = ['NOsharing_NOaux', 'sharing_NOaux', 'NOsharing_aux', 'sharing_aux']
HP = eval(open('HP.txt', 'r').read())

n_iter = 30
nbr_pairs = 1000
all_values = torch.zeros(n_iter,len(type_models),len(sub_models))


# Test each model 30 times
for j in range(n_iter):
    print('Iteration', j+1, '/', n_iter, '...')
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nbr_pairs)
    acc_test = torch.zeros(len(type_models),len(sub_models))
    for t, type_model in enumerate(type_models):
        for s, sub_model in enumerate(sub_models):
            best_hidden = HP[type_model][sub_model]['hidden']
            best_eta = HP[type_model][sub_model]['eta']
            if(sub_model == 'NOsharing_aux' or sub_model == 'sharing_aux'): 
                best_lambda = HP[type_model][sub_model]['lambda']
            else: best_lambda = 0
            
            _, model = model_training(train_input, train_target, train_classes, best_hidden, best_eta, best_lambda, 
                                      model_type = type_model, sub_model = sub_model)
                
            if(sub_model == 'NOsharing_aux' or sub_model == 'sharing_aux'): 
                acc_test[t,s] = 1 - compute_nb_errors_aux(model, test_input, test_target, 100)/len(test_target)
            else: 
                acc_test[t,s] = 1 - compute_nb_errors_NOaux(model, test_input, test_target, 100)/len(test_target)
    all_values[j] = acc_test
    
print
    
# Save results
torch.save(all_values, 'results.pt')
print('Results are saved in the file results.pt')