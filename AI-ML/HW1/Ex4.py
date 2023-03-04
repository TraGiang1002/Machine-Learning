# Ex4: print all Possible Combinations from the three Digits
data4 = [1, 2, 3]

def get_combinations(data, n):
    result = []
    if n == 0:
        result.append('')
        return result
    smaller = get_combinations(data, n-1)
    for i in range(len(smaller)):
        for j in range(len(data)):
            if str(data[j]) not in smaller[i]:
                result.append(smaller[i] + str(data[j]))
    return result


data4 = [1, 2, 3]
combinations = get_combinations(data4, 3)
print(combinations)

