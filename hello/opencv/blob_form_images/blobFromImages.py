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

images = []

for p in imagePaths[1:]:
    image = cv.imread(p)
    image = cv.resize(image, (224, 224))
    images.append(image)

blob = cv.dnn.blobFromImages(images, 1, (224, 224), (104, 117, 123))
print("Second blob: {}".format(blob.shape))

net.setInput(blob)
preds = net.forward()

for (i, p) in enumerate(imagePaths[1:]):

    image = cv.imread(p)

    idx = np.argsort(preds[i])[::-1][0]

    text = "Label: {}, {:.2f}%".format(classes[idx], preds[i][idx]*100)
    cv.putText(image, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.imshow("Image", image)
    cv.waitKey(0)
# cv.destroyAllWindows()