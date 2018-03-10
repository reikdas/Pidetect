#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 19:13:37 2018

@author: pratyush
"""

#https://stackoverflow.com/questions/34026097/using-a-pi-camera-module-with-opencv-python

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import os
import sys
import RPi.GPIO as GPIO

#Setup Pins
pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT, initial=0)

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(320, 240))

display_window = cv2.namedWindow("Faces")

path = os.path.join(sys.path[0], 'face_classifier.xml')
face_cascade = cv2.CascadeClassifier(path)

time.sleep(1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = frame.array

    #FACE DETECTION STUFF
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        GPIO.output(pin, GPIO.HIGH)

    #DISPLAY TO WINDOW
    cv2.imshow("Faces", image)
    key = cv2.waitKey(1)

    rawCapture.truncate(0)

    if key == ord("q"):
        camera.close()
        cv2.destroyAllWindows()
        break