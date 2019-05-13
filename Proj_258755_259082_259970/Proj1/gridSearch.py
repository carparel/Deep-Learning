from utilities import grid_search_
import dlc_practical_prologue as prologue
import torch


# Generate Dataset
nbr_pairs = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nbr_pairs)

# Choose hyper parameters to perform the grid search on
lambdas = torch.tensor([0.25, 0.5, 0.75, 1])
etas = torch.tensor([0.1, 0.01, 0.001])
hidden_units = torch.tensor([50, 100, 200, 300])

HP = grid_search_(lambdas, etas, hidden_units, train_input, train_target, train_classes, test_input, test_target)

f = open("HP.txt","w")
f.write( str(HP) )
f.close()
print('Results are saved in the file HP.txt')