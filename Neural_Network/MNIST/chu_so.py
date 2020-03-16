# Import các thư viện cần thiết
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Import MNIST data
# TF có hỗ trợ giúp chúng ta đọc dữ liệu mnist
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
mnist = tf.keras.datasets.mnist
# print(mnist)

# Khai báo các tham số của mô hình
learning_rate = 0.1 # tốc độ học
num_steps = 500 # tổng số lần học/huấn luyện
batch_size = 128 # Số điểm dữ liệu đưa vào học mỗi lần huấn luyến
display_step = 100 # Cứ sau 100 lần học, hiện thị các thay đổi thông số của mô hình
 
# Các tham số của mạng
n_hidden_1 = 256 # 1st layer number of neurons - số nơ ron của layer 1
n_hidden_2 = 256 # 2nd layer number of neurons - số nơ ron của layer 2
input_shape = 784 # MNIST data input (img shape: 28*28) kích thước của 1 input(vector 784 chiều).
num_classes = 10 # MNIST total classes (0-9 digits) - label vector dạng one-hot
 
# tf Graph input

X = tf.placeholder("float", shape=input_shape, name=None)
Y = tf.placeholder("float", shape=num_classes, name=None)