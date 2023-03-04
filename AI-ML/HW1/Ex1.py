# Ex1: Write a program to count positive and negative numbers in a list
data1 = [-10, -21, -4, -45, -66, 93, 11, -4, -6, 12, 11, 4]

def Count(array):
    countP = 0
    countN = 0
    for i in array:
        if(i>=0):
            countP = countP + 1;
        else:
            countN = countN + 1;
    return countP, countN
count1, count2 = Count(data1)
print(count1, count2)