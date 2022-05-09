import cv2
import dlib
from mtcnn.mtcnn import MTCNN
import numpy as np
from cv2 import waitKey
from cv2 import destroyAllWindows

detector1 = MTCNN()
modelFile = "cs143-final-facebox-anime-stylize/face_test/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "cs143-final-facebox-anime-stylize/face_test\deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret, img = cap.read()
    if ret == True:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        height, width = img.shape[:2]
        img1 = img.copy()
        img2 = img.copy()
        img3 = img.copy()
        # detect faces in the image
        faces1 = detector1.detect_faces(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                     1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces3 = net.forward()

        # display faces on the original image
        for result in faces1:
            x, y, w, h = result['box']
            x1, y1 = x + w, y + h
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img, 'mtcnn', (30, 30), font, 1,
                    (255, 255, 0), 2, cv2.LINE_AA)

        for i in range(faces3.shape[2]):
            confidence = faces3[0, 0, i, 2]
            if confidence > 0.5:
                box = faces3[0, 0, i, 3:7] * \
                    np.array([width, height, width, height])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img2, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img2, 'dnn', (30, 30), font, 1,
                    (255, 255, 0), 2, cv2.LINE_AA)

        h2 = cv2.hconcat([img, img2])

        cv2.imshow("mtcnn", img)
        cv2.imshow("dnn", img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
