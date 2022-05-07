import numpy as np
from skimage.io import imread
import dlib
import cv2


def NN = 0
def CNN = 0

# read image
# detect face(s) using NN
# create bounding box(es) using top/left/right/bot-most values
# run stylization on just face boxes
# cut and paste the stylized faces into the original target image

# pre-process?


def get_im(image_path):
   image = imread(image_path, as_gray=False)
   image = np.flatten(image)
   return image


def get_faces(image):
   faces = NN(image)
   # learn how this is represented from paper
   # convert to 2d array?
   return faces


def bound_box(face):
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread('test.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)  # result
    # to draw faces on image
    for result in faces:
        x = result.left()
        y = result.top()
        x1 = result.right()
        y1 = result.bottom()
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

   top = Positive_Infinity
   bot = Negative_Infinity
   left = pos
   right = neg
   # NEED FACE INFO IN CONTEXT OF OG IMAGE -> find loc
   for i in range(len(face)):
       for j in range(np.shape(face)[2]):
           3
 
   return (top,bot,left,right)
 
# TODO: choose style transfer or stylize
def stylify(style_image, target_image):
   return CNN(style_image, target_image)
 
def bb_style(style_image, target_image, bounds):
   # for i in range(bounds[0], bounds[1]):
   #     for j in range(bounds[2], bounds[3]):
   box_cut = target_image[bounds[0]:bounds[1], bounds[2]:bounds[3]]
   styled_box = stylify(style_image, box_cut)
   new_image = target_image
   new_image[bounds[0]:bounds[1], bounds[2]:bounds[3]] = styled_box
   return new_image