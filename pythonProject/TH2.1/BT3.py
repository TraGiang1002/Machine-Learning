import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Change to data path on your computer
data = pd.read_csv("SAT_GPA.csv")

# Separate features and labels
I = data.iloc[:,0]
J = data.iloc[:,1]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(I, J, train_size=60)

# Temp (F degree)
X = np.array([X_train]).T
# Press (Atm)
y = np.array([y_train]).T

# Visualize data
plt.scatter(X,y)
plt.axis([1663, 2050, 2.4, 3.81])
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

#Building Xbar
one = np.ones((X.shape[0], 1)) #tao ma trận cột các ptu = 1 độ dài
Xbar = np.concatenate((one, X), axis = 1) #thêm cột các ptu = 1 vào trước ma trận X

# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar) # Xbar bình phương
b = np.dot(Xbar.T, y) # nhân Xbar với y
w = np.dot(np.linalg.pinv(A), b) # nhân b với ma trận nghịch đảo cua A
print('w = ', w)
# Preparing the fitting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(1663, 2050, 2) # tạo mang 2 ptu 193 và 213
y0 = w_0 + w_1*x0
# Drawing the fitting line
plt.scatter(X,y)
plt.plot(x0, y0) # the fitting line
plt.axis([1663, 2050, 2.4, 3.81])
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

