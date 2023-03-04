## packages
from __future__ import division, print_function, unicode_literals
import numpy as np # thư viện tính toán ma trận và mảng
from scipy.sparse import coo_matrix # thư viện tạo và xử lý ma trận thưa dưới dạng  COO
from sklearn.naive_bayes import MultinomialNB, BernoulliNB # thư viện cài đặt thuật toán Naive Bayes cho bài toán phân loại multinomial hoặc Bernoulli
from sklearn.metrics import accuracy_score #  thư viện tính toán độ chính xác của mô hình phân loại so với kết quả thực tế.

# data path and file name
path = 'ex6DataPrepared/'
train_data_fn = 'train-features.txt'
test_data_fn = 'test-features.txt'
train_label_fn = 'train-labels.txt'
test_label_fn = 'test-labels.txt'
nwords = 2500

def read_data(data_fn, label_fn):
    ## read label_fn
    with open(path + label_fn) as f: # đọc dữ liệu từ tệp ghi vào f
        content = f.readlines() # từ f dữ liêu trả về danh sách theo chuỗi dòng lưu vào content
    label = [int(x.strip()) for x in content] # mỗi chuỗi đươợc loại bỏ khoảng trắng bằng strip và chuyển về int
    # print(label)
    ## read data_fn
    with open(path + data_fn) as f:
        content = f.readlines()
    # remove '\n' at the end of each line
    content = [x.strip() for x in content]
    print(content)
    dat = np.zeros((len(content), 3), dtype=int) # trả về ma trận len(content)x3 kiểu dữ liệu int
    print(dat)
    for i, line in enumerate(content): # đọc từng donng dữ liệu
        a = line.split(' ') # tách thành mảng các chuỗi con từ khoảng trắng lưu vào a
        print(a)
        dat[i, :] = np.array([int(a[0]), int(a[1]), int(a[2])]) # mảng a chuyển sang kiểu dữ liệu int và được lưu trữ trong một hàng của ma trận dat, theo thứ tự [int(a[0]), int(a[1]), int(a[2])]
        print(dat)
    # remember to -1 at coordinate since we're in Python
    # check this: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo
    #for more information about coo_matrix
    data = coo_matrix((dat[:, 2], (dat[:, 0] - 1, dat[:, 1] - 1)), shape=(len(label), nwords)) # ma trận dat và biến label được sử dụng để tạo ra một ma trận thưa kích thước len(label)xnwords
    print(data)
    print(label)
    return (data, label)

(train_data, train_label) = read_data(train_data_fn, train_label_fn)
(test_data, test_label) = read_data(test_data_fn, test_label_fn)
clf = MultinomialNB()
clf.fit(train_data, train_label)
y_pred = clf.predict(test_data)
print(y_pred)