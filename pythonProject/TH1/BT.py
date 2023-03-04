import pandas as pd
from sklearn.model_selection import train_test_split # thư viện chia dữ liệu thành tập huấn luyện và tập kiểm tra
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score # tính các chỉ số đánh giá kết quả của mô hình
# accuracy_score (độ chính xác), precision_score (độ chính xác dương tính), recall_score (độ nhạy).

# Load data
df = pd.read_csv('Cancer.csv', header=None)
# Separate features and labels
X = df.iloc[:, 1:]
y = df.iloc[:, 1]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=120, stratify=y)

# Select 80 benign and 40 malignant samples for testing
test_samples = pd.concat([X_test[y_test == 2].head(80), X_test[y_test == 4].head(40)])
X_test = X_test.drop(test_samples.index)
y_test = y_test.drop(test_samples.index)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict test set labels
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=4)
recall = recall_score(y_test, y_pred, pos_label=4)

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))