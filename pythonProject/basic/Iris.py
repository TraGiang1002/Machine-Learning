with open (file="Iris.csv", mode="r") as file:
    a = file.readlines()

x = float(input("Chieu dai: "))
y =  float(input("Chieu rộng: "))
total = []
name = ""
for line in a[1:]:
    text = line.split(sep=",")
    tinhChieuDai = abs(x - float(text[1]))
    tinhChieuRong = abs(x - float(text[2]))
    sum = tinhChieuDai + tinhChieuRong
    total.append(sum)
    if min(total) == sum:
        name = text[-1]
print(name)

