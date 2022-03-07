import numpy as np
from ogb.lsc import PCQM4MDataset
from ogb.utils import smiles2graph

dataset = PCQM4MDataset(root = './pcqm4m/', only_smiles = True)

i = 1234
print(dataset[i])

max_num_atoms = 0
for i in range(len(dataset)):
    smiles, target = dataset[i]
    graph_obj = smiles2graph(smiles)
    num_atoms = graph_obj['num_nodes']
    if max_num_atoms < num_atoms:
        max_num_atoms = num_atoms
        print('Max num atoms:', max_num_atoms)
    if (i + 1) % 1000 == 0:
        print('Done', i + 1)

'''
from ogb.graphproppred import Evaluator

evaluator = Evaluator(name = 'ogbg-molhiv')
print(evaluator.expected_input_format) 
print(evaluator.expected_output_format)

y_true = np.reshape(np.array([1, 1, 0, 0]), (4, 1))
y_pred = np.reshape(np.array([1, 1, 1, 0]), (4, 1))
input_dict = {"y_true": y_true, "y_pred": y_pred}
result_dict = evaluator.eval(input_dict)
print(result_dict['rocauc'])
'''
