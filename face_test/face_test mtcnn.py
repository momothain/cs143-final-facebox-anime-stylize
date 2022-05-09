import cv2
from mtcnn.mtcnn import MTCNN
from cv2 import waitKey
from cv2 import destroyAllWindows
import os

detector = MTCNN()
img = cv2.imread('cs143-final-facebox-anime-stylize/face_test/aot.jpg')
faces = detector.detect_faces(img)  # result
directory = r'C:/Users/lizak\Desktop\Brown\Spring 2022/CSCI 1430/cs143-final-facebox-anime-stylize/face_test/results'
os.chdir(directory)
# to draw faces on image
for result in faces:
    x, y, w, h = result['box']
    x1, y1 = x + w, y + h
    cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

filename = 'facebox_mtcnn_aot.jpg'
cv2.imwrite(filename, img)

cv2.imshow('face_test_boxes', img)
# keep the window open until we press a key
waitKey(0)
# close the window
destroyAllWindows()
