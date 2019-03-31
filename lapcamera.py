#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 15:01:14 2019

@author: jay
"""

import cv2
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

import face_recognition
face_image = face_recognition.load_image_file(img_name)
face_locations = face_recognition.face_locations(face_image)
from PIL import Image
print("I found {} face(s) in this photograph.".format(len(face_locations)))
for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = face_image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
    pil_image.save("dataset/prediction/temp.jpg")

import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.utils import get_file
classifier = load_model('models/GDmodel2.h5')
test_image = image.load_img("dataset/prediction/temp.jpg", target_size = (128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
print(result)
if result[0][0] == 0:
    prediction = 'men'
else:    prediction = 'women'
print(prediction)


