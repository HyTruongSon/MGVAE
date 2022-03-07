import numpy as np

def read_index(file_name):
    file = open(file_name, 'r')
    index = [float(element) for element in file.readline().split(',')]
    file.close()
    return index

def save_index(file_name, index):
    file = open(file_name, 'w')
    for i in range(len(index)):
        if i > 0:
            file.write(',')
        file.write(str(int(index[i])))
    file.close()

def read_smiles(file_name):
    file = open(file_name, 'r')
    num_molecules = int(file.readline())
    res = []
    for sample in range(num_molecules):
        value = file.readline()
        res.append(value)
    file.close()
    res = np.array(res)
    return res

def read_target(file_name):
    file = open(file_name, 'r')
    num_molecules = int(file.readline())
    res = []
    for sample in range(num_molecules):
        value = float(file.readline())
        res.append(value)
    file.close()
    res = np.array(res)
    return res

train_index = read_index('train.index')
val_index = read_index('val.index')
test_index = read_index('test.index')

total_index = train_index + val_index + test_index
save_index('total.index', total_index)
print(len(read_index('total.index')))

num_train = len(train_index)
num_val = len(val_index)
num_test = len(test_index)
num_total = len(total_index)

new_train_index = range(0, num_train)
new_val_index = range(num_train, num_train + num_val)
new_test_index = range(num_train + num_val, num_total)

save_index('new_train.index', new_train_index)
save_index('new_val.index', new_val_index)
save_index('new_test.index', new_test_index)

smiles = read_smiles('../ZINC_smiles/smiles')
logp = read_target('../ZINC_smiles/logp')
qed = read_target('../ZINC_smiles/qed')
sas = read_target('../ZINC_smiles/sas')

smiles_file = open('smiles', 'w')
logp_file = open('logp', 'w')
qed_file = open('qed', 'w')
sas_file = open('sas', 'w')
smiles_file.write(str(num_total) + '\n')
logp_file.write(str(num_total) + '\n')
qed_file.write(str(num_total) + '\n')
sas_file.write(str(num_total) + '\n')
for i in range(num_total):
    index = int(total_index[i])
    smiles_file.write(smiles[index])
    logp_file.write(str(logp[index]) + '\n')
    qed_file.write(str(qed[index]) + '\n')
    sas_file.write(str(sas[index]) + '\n')
smiles_file.close()
logp_file.close()
qed_file.close()
sas_file.close()
print('Done')
