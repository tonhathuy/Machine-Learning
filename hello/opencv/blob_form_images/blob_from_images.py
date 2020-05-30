# import the necessary packages
from imutils import paths
import numpy as np
from cv2 import cv2 as cv 

rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# print('rows:', rows[:10])
print(classes[:10])

# load our serialized model from disk
net = cv.dnn.readNetFromCaffe("bvlc_googlenet.prototxt",
	"bvlc_googlenet.caffemodel")

# grab the paths to the input images
imagePaths = sorted(list(paths.list_images("images/")))
# print(imagePaths)
# (1) load the first image from disk, (2) pre-process it by resizing
# it to 224x224 pixels, and (3) construct a blob that can be passed
# through the pre-trained network
image = cv.imread(imagePaths[2])
resized = cv.resize(image, (224,224))
blob = cv.dnn.blobFromImage(resized, 1, (224, 224), (104, 117, 123))
print("first Blob: {}".format(blob.shape))

# set the input to the pre-trained deep learning network and obtain
# the output predicted probabilities for each of the 1,000 ImageNet
# classes
net.setInput(blob)
preds = net.forward()
# print(preds[0])
# sort the probabilities (in descending) order, grab the index of the
# top predicted label, and draw it on the input image
# https://www.geeksforgeeks.org/numpy-argsort-in-python/
idx = np.argsort(preds[0])[::-1][0] #print ra index của so lon nhat , nếu ::1 thì in ra bé nhất
print(preds[0][idx]) # in ra preds[0] lon nhat
text = "Label: {}, {:.2f}%".format(classes[idx],
	preds[0][idx] * 100)
cv.putText(image, text, (5, 25),  cv.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 0, 255), 2)

# show the output image
cv.imshow("Image", image)
cv.waitKey(0)
cv.destroyAllWindows()




# import numpy as geek 
  
# # input array 
# in_arr = geek.array([ 2, 0,  1, 5, 4, 1, 9]) 
# print ("Input unsorted array : ", in_arr)  
  
# out_arr = geek.argsort(in_arr)[::1][0] 
# print ("Output sorted array indices : ", out_arr) 
# print("Output sorted array : ", in_arr[out_arr]) 