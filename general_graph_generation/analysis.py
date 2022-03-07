import numpy as np

def read_file(file_name):
    all_lines = []
    all_names = []
    all_values = []
    with open(file_name) as file:
        for line in file:
            all_lines.append(line)
            pos = -1
            for i in range(len(line)):
                if line[i] == ':':
                    pos = i
                    break
            name = line[:pos]
            value = float(line[pos + 2:])
            found = False
            for i in range(len(all_names)):
                if all_names[i] == name:
                    all_values[i].append(value)
                    found = True
                    break
            if found == False:
                all_names.append(name)
                all_values.append([value])
    return all_lines, all_names, all_values

all_lines, all_names, all_values = read_file('community_logs')

for i in range(len(all_names)):
    arr = np.array(all_values[i])
    mean = np.mean(arr)
    std = np.std(arr)
    print(all_names[i], ': mean = ', mean, ', std = ', std)
