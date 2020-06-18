import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import autokeras as ak
import cv2
from tensorflow.keras.datasets import mnist
from tqdm import tqdm
import numpy as np

 
path_data = 'data/'
 
x_data = []
y_data = []
 
for i_name in tqdm(os.listdir(os.path.join(path_data, '1'))):
    image = cv2.imread((os.path.join(path_data, '1', i_name)))
    resized_image = cv2.resize(image, (64, 64))
    x_data.append(resized_image)
    y_data.append([1])
    
for i_name in tqdm(os.listdir(os.path.join(path_data, '0'))):
    image = cv2.imread((os.path.join(path_data, '0', i_name)))
    resized_image = cv2.resize(image, (64, 64))
    x_data.append(resized_image)
    y_data.append([0])
    
x_data = np.array(x_data)
y_data = np.array(y_data)

print(x_data.shape, y_data.shape)

#Thực hiện chia dữ liệu thành 2 tập ngẫu nhiên để train và validation theo tỉ lệ 9:1
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1)

#Tiến hành huấn luyện với Auto Keras
# from autokeras.tasks.image import ImageClassifier
clf = ak.ImageClassifier(max_trials=1)
clf.fit(x_train, y_train, epochs=100)
# from autokeras.image_supervised import ImageClassifier
# clf = ImageClassifier(verbose=True)
# clf.fit(x_train, y_train, time_limit=12 * 60 * 60)

# clf.final_fit(x_train, y_train, x_val, y_val, retrain=True)
score = clf.evaluate(x_val, y_val)
print(score)

# clf.load_searcher().load_best_model().produce_keras_model().save('model.h5')
# from autokeras.utils import pickle_from_file
 
# model = pickle_from_file("best_auto_keras_model.h5")
# clf.export_keras_model("hezd.h5")
clf.export_model()