# Ex3: find the strongest neighbour. Given an array of N positive integers.
# The task is to find the maximum for every adjacent pair in the array.
data3 = [4, 5, 6, 7, 3, 9, 11, 2, 10]

list = []
n = len(data3)
for i in range(n):
    if (i == 0):
        if(data3[0] >= data3[1]):
            list.append(data3[0])
        else:
            list.append(data3[1])
    elif(i == (n-1)):
        if (data3[n-1] >= data3[n-2]):
            list.append(data3[n-1])
        else:
            list.append(data3[n-2])
    elif(i >= 1 & i <= (n-2)):
        if ((data3[i] > data3[i + 1]) & (data3[i] > data3[i-1])):
            list.append(data3[i])
        elif ((data3[i-1] > data3[i]) & (data3[i-1] > data3[i+1])):
            list.append(data3[i - 1])
        elif ((data3[i+1] > data3[i]) & (data3[i+1] > data3[i-1])):
            list.append(data3[i + 1])
print(list)
