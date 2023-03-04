import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Đọc dữ liệu từ tập tin CSV
data = pd.read_csv('vidu4_lin_reg.txt', sep=" ", header=0)
print(data)
regr = linear_model.LinearRegression()
y_data = data.iloc[:, -1]
x_data = data.iloc[:, 1:6]
regr.fit(x_data, y_data)
print('β_0 : ', regr.intercept_)
print('β_1(TUOI): ', regr.coef_[0])
print('β_2(CHOLESTEROL): ', regr.coef_[1])
print('β_3(GLUCOSE): ', regr.coef_[2])
print('β_4(HA): ', regr.coef_[3])
print('β_5(BMI): ', regr.coef_[4])
data = pd.read_csv('vidu4_lin_reg.txt', sep=" ", header=0)
print(data)
regr = linear_model.LinearRegression()
y_data = data.iloc[:, -1]
x_data = data.iloc[:, 1:6]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=False)
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)
print(mean_absolute_error(y_pred, y_test))
print(mean_squared_error(y_pred, y_test))