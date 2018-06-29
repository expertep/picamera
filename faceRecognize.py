import sys, time
import cv2
import os
import numpy as np
import imutils
from imutils.video import VideoStream
import datetime
import argparse

def get_images(path, size):
    '''
    path: path to a folder which contains subfolders of for each subject/person
        which in turn cotains pictures of subjects/persons.

    size: a tuple to resize images.
        Ex- (256, 256)
    '''
    sub= 0
    images, labels= [], []
    people= []

    for subdir in os.listdir(path):
        for image in os.listdir(path+ "/"+ subdir):
            #print(subdir, images)
            img= cv2.imread(path+os.path.sep+subdir+os.path.sep+image, cv2.IMREAD_GRAYSCALE)
            img= cv2.resize(img, size)

            images.append(np.asarray(img, dtype= np.uint8))
            labels.append(sub)

            #cv2.imshow("win", img)
            #cv2.waitKey(10)

        people.append(subdir)
        sub+= 1

    return [images, labels, people]

def detect_faces(image):
    '''
    Takes an image as input and returns an array of bounding box(es).
    '''
    frontal_face= cv2.CascadeClassifier("/home/pi/opencv-2.4.13.6/data/haarcascades/haarcascade_frontalface_default.xml")
    #frontal_face= cv2.CascadeClassifier("haarcascade_upperbody.xml")
    bBoxes= frontal_face.detectMultiScale(image, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

    return bBoxes

def train_model(path):
    '''
    Takes path to images and train a face recognition model
    Returns trained model and people
    '''
    [images, labels, people]= get_images(path, (256, 256))
    #print([images, labels])

    labels= np.asarray(labels, dtype= np.int32)

    # initializing eigen_model and training
    print("Initializing eigen FaceRecognizer and training...")
    sttime= time.clock()
    eigen_model= cv2.face.FisherFaceRecognizer_create()
    eigen_model.train(images, labels)
    print("\tSuccessfully completed training in "+ str(time.clock()- sttime)+ " Secs!")

    return [eigen_model, people]

def majority(mylist):
    '''
    Takes a list and returns an element which has highest frequency in the given list.
    '''
    myset= set(mylist)
    ans= mylist[0]
    ans_f= mylist.count(ans)

    for i in myset:
        if mylist.count(i)> ans_f:
            ans= i
            ans_f= mylist.count(i)

    return ans