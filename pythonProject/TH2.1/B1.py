from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
# Temp (F degree)
X = np.array([[194.5, 194.3, 197.9, 198.4, 199.4, 199.9, 200.9, 201.1, 201.4, 201.3, 203.6, 204.6, 209.5, 208.6, 210.7, 211.9, 212.2]]).T
# Press (Atm)
y = np.array([[20.79,20.79,22.4,22.67,23.15,23.35,23.89,23.99,24.02,24.01,25.14,26.57,28.49,27.76,29.04,29.88,30.06]]).T

# Visualize data
#vẽ biểu đồ đường
plt.plot(X, y, 'ro') #r là màu đỏ (red) và o là hình tròn biểu thị cho các điểm dữ liệu
#thiết lập giới hạn trục của biểu đồ
plt.axis([193, 213, 19, 31]) #[193, 213] là giới hạn trục x và [19, 31] là giới hạn trục y
plt.xlabel('Temperature (F)')
plt.ylabel('Pressure (Atm)')
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
x0 = np.linspace(193, 213, 2) # tạo mang 2 ptu 193 và 213
y0 = w_0 + w_1*x0
# Drawing the fitting line
plt.plot(X.T, y.T, 'ro') # data
plt.plot(x0, y0) # the fitting line
plt.axis([193, 213, 19, 31])
plt.xlabel('Temperature (F)')
plt.ylabel('Pressure (Atm)')
plt.show()
