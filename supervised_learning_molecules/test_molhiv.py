import numpy as np
from ogb.graphproppred import GraphPropPredDataset

dataset = GraphPropPredDataset(name = 'ogbg-molhiv')

print(len(dataset))

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

### set i as an arbitrary index
for i in range(100):
    graph, label = dataset[i] # graph: library-agnostic graph object

    # print(graph)
    print(label)

from ogb.graphproppred import Evaluator

evaluator = Evaluator(name = 'ogbg-molhiv')
print(evaluator.expected_input_format) 
print(evaluator.expected_output_format)

y_true = np.reshape(np.array([1, 1, 0, 0]), (4, 1))
y_pred = np.reshape(np.array([1, 1, 1, 0]), (4, 1))
input_dict = {"y_true": y_true, "y_pred": y_pred}
result_dict = evaluator.eval(input_dict)
print(result_dict['rocauc'])
