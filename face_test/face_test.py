from time import sleep
import cv2
import numpy as np

modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile) #readNetFromCaffe
im = cv2.imread('test1.jpg')
h, w = im.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)), 1.0,
(300, 300), (104.0, 117.0, 123.0))
# ALT: To achieve the best accuracy run the model on BGR images resized to 
# 300x300 applying mean subtraction of values (104, 177, 123) for each blue, green and red channels correspondingly.
net.setInput(blob)
faces = net.forward()
# print(faces)
#to draw faces on image
for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            im = cv2.rectangle(im, (x, y), (x1, y1), (0, 0, 255), 2)

cv2.imshow('face_test_boxes', im)