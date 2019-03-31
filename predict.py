#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:35:45 2019

@author: jay
"""
import face_recognition
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.utils import get_file
classifier = load_model('models/GDmodel2.h5')
face_image = face_recognition.load_image_file("dataset/prediction/test3.jpg")
face_locations = face_recognition.face_locations(face_image)
from PIL import Image
print("I found {} face(s) in this photograph.".format(len(face_locations)))
i=0
for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    name = 'temp_' + str(i)
    i+=1
    # You can access the actual face itself like this:
    face_image = face_recognition.load_image_file("dataset/prediction/test3.jpg")
    face_image = face_image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
    pil_image.save("dataset/prediction/"+name+".jpg")
    
    

test_image = image.load_img('dataset/prediction/temp_1.jpg', target_size = (128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
print(result)
if result[0][0] == 0:
    prediction = 'men'
else:    prediction = 'women'
print(prediction)
