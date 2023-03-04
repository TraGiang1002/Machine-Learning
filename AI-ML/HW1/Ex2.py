# Ex2: Given a list, extract all elements whose frequency is greater than k.
data2 = [4, 6, 4, 3, 3, 4, 3, 4, 3, 8]
k = 3

list2 = []
n = len(data2)
for i in range(n):
    count = 0
    counter = data2[i]
    for i in range(n):
        if(counter == data2[i]):
            count = count + 1
    if(count >= 3):
        list2.append(counter)
answer = set(list2)
print(list(answer))