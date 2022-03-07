import numpy as np

def get_content(file_name):
    res = []
    with open(file_name) as file:
        for line in file:
            pos = -1
            for i in range(len(line)):
                if line[i] == '=':
                    pos = i
                    break
            str = ''
            i = pos + 1
            while i < len(line):
                str += line[i]
                i += 1
            res.append(float(str))
    return res

arr = get_content('out')
arr = np.array(arr)
print(arr)
print('Mean:', np.mean(arr))
print('STD:', np.std(arr))
