import cv2
import skimage.io as io
import urllib.request
import numpy as np
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

st = "https://cdn.cnn.com/cnnnext/dam/assets/190227090506-20190227-ai-faces-split-exlarge-169.jpg"
img_array = urllib.request.urlopen(st)
print (img_array)
test1 = cv2.imdecode(np.array(bytearray(img_array.read()), dtype=np.uint8), -1)
# test1 = cv2.imdecode(img_array, -1)
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)        
print(len(faces))