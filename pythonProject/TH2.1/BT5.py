# Khai báo các thư viện
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error
# Đọc dữ liệu từ tập tin CSV
data = pd.read_csv("real_estate.csv")
print(data)

y_data = data.iloc[:, -1]
x_data = data.iloc[:, 1:7]
x_data['X1 transaction date'] = data['X1 transaction date'].apply(int)
lregr = linear_model.LinearRegression()

# Chia dữ liệu thành phần training với 350 mẫu đầu tiên, phần validation với số mẫu còn lại
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=350, shuffle=False)
lregr.fit(x_train, y_train)
# Chạy dự đoán cho phần dữ liệu validation và đưa ra tổng bình phương sai số của dự đoán
y_pred = lregr.predict(x_test)
sse = ((y_test - y_pred) ** 2).sum()
print('Tổng bình phương sai số của dự đoán:', sse)