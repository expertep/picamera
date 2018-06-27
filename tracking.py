#!/usr/bin/env python
import sys, time
import cv2
import os
import numpy as np
import imutils
from imutils.video import VideoStream
import datetime
import argparse
import faceRecognize as faceRec

def stalable (sumAreaDetect):
    dif = 0
    lenSum = len(sumAreaDetect)
    for i in range (lenSum-1) :
        if i == lenSum-1:
            dif += abs(sumAreaDetect[0] - sumAreaDetect[i])
            break
        dif += abs(sumAreaDetect[i] - sumAreaDetect[i+1])
    if dif/lenSum <=0:
        return True
    else: return False

# Are we using the Pi Camera?
usingPiCamera = True
# Set initial frame size.
frameSize = (600, 400)

countPeople = 0

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size")
args = vars(ap.parse_args())

# Initialize mutithreading the video stream.

if args["video"] is None :
    vs = VideoStream(src=0, usePiCamera=usingPiCamera, resolution=frameSize, framerate=32).start()
else :
    vs = cv2.VideoCapture(args["video"])
    usingPiCamera = False
#vs = cv2.VideoCapture('/home/pi/Desktop/videoplayback.avi')
# Allow the camera to warm up.
time.sleep(2.0)
 
timeCheck = time.time()

firstFrame = None
firstTrack = None
status = 'motion'
# How long have we been tracking
idle_time = 30
reset = False

grid = frameSize[0]/5
doorsize = 100
center = frameSize[0]/2
bbox = None
sumAreaDetect  = [0,0,0]
pointMove = []
lencnts = 0

pathToDB = "/home/pi/FaceRecognizer/sorted_output_images"
#train
eigen_model, people= faceRec.train_model(pathToDB)
last_20 = [0 for i in range(20)]
final_5 = []
box_text= "Subject: "
while True:
    # Get the next frame.
    
    if args["video"] is None :
        frame = vs.read()
        frame = imutils.rotate(frame, 180)
    else:
        _,frame = vs.read()
    text = "Unoccupied"
        # If using a webcam instead of the Pi Camera,
    # we take the extra step to change frame size.
    if not usingPiCamera:
        frame = imutils.resize(frame, width=frameSize[0])    
    if (frame is None):
        print("not connect camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayFace = cv2.equalizeHist(gray)
    """
    faces = faceRec.detect_faces(grayFace)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        crop_gray_frame = grayFace[y:y+h, x:x+w]
        crop_gray_frame = cv2.resize(crop_gray_frame, (256, 256))
        crop_color_frame = frame[y:y+h, x:x+w]

        [predicted_label, predicted_conf]= eigen_model.predict(np.asarray(crop_gray_frame))
        last_20.append(predicted_label)
        last_20= last_20[1:]

        if idle_time%5== 0:
            max_label= faceRec.majority(last_20)
            box_text= format("Subject: "+ people[max_label])
            #box_text= format("Subject: "+ people[predicted_label])

        cv2.putText(frame, box_text, (x-20, y-5), cv2.FONT_HERSHEY_PLAIN, 1.3, (25,0,225), 2)
    """
    gray = cv2.GaussianBlur(grayFace, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None and idle_time > 10:
        firstFrame = gray
        continue
    #if status == "motion":
    if idle_time > 10:
        # Difference between current frame and background
        frame_delta = cv2.absdiff(firstFrame,gray)
        # Create a threshold to exclude minute movements
        thresh = cv2.threshold(frame_delta,100,255,cv2.THRESH_BINARY)[1]

        #Dialate threshold to further reduce error
        thresh = cv2.dilate(thresh,None,iterations=2)
        # Check for contours in our threshold
        _,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        lencnts = int(len(cnts))
        # Check each contour
        sumArea= 0
        
        if lencnts != 0:
            # If the contour is big enough

            # Set largest contour to first contour
            largest = 0
            
            # For each contour
            for i in range(lencnts):
                (px, py, pw, ph) = cv2.boundingRect(cnts[i])
                if idle_time%2 == 0:
                    sumArea += cv2.contourArea(cnts[i])
                
                # If this contour is larger than the largest
                if i != 0 and int(cv2.contourArea(cnts[i])) > int(cv2.contourArea(cnts[largest])):
                    # This contour is the largest
                    largest = i
                #(x,y,w,h) = cv2.boundingRect(cnts[i])
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)   
            
            if cv2.contourArea(cnts[largest]) > args["min_area"] :
                # Create a bounding box for our contour
                (x,y,w,h) = cv2.boundingRect(cnts[largest])
                # Convert from float to int, and scale up our boudning box
                (x,y,w,h) = (int(x),int(y),int(w),int(h))
                # Initialize tracker
                bbox = (x,y,w,h)
                #crop_gray_frame = grayFace[y:y+h, x:x+w]
                #faces = faceRec.detect_faces(crop_gray_frame)
                #for (x,y,w,h) in faces:
                    #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                # Switch from finding motion to tracking
                #cv2.imshow("Camera1",crop_gray_frame)
                status = 'tracking'
        if idle_time%2 == 0:
            sumAreaDetect[0] = sumArea
    # If we are tracking
    if status == 'tracking':
        # Update our tracker
        # Create a visible rectangle for our viewing pleasure
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame,p1,p2,(0,0,255),2)
        if firstTrack is None:
            firstTrack = p1
            continue
        #check line movement

        if not pointMove:
            pointMove.append(p1[0])
        if abs(p1[0]-pointMove[len(pointMove)-1]) > grid + bbox[2]:
            reset = True
        else : pointMove.append(p1[0])
        if len(pointMove) > 10 :
            pointMove.pop(0)
            #เข้า
        if not reset :
            if (firstTrack[0] > center-doorsize and firstTrack[0] < center + doorsize):
                if (p1[0] <= grid or p1[0] >= frameSize[0] - grid):
                    #if stalable(sumAreaDetect) or p1[0]<=0 or p2[0] >= frameSize[0]:
                    countPeople += 1
                    reset = True
                #if p1[0] <= grid*4 and p1[0] >= grid*2:
                    #filename = "/home/pi/in/"+str(time.time())+".jpg"
                    #cv2.imwrite(filename, frame)
            elif firstTrack[0] <= grid or firstTrack[0] >= frameSize[0]-grid:
                if p1[0] > center - doorsize and p1[0] < center + doorsize:
                    if stalable(sumAreaDetect) :
                        if countPeople > 0:
                            countPeople -= 1
                            reset = True
    cv2.putText(frame, "People: "+str(countPeople), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    tmp = 0
    cv2.putText(frame, "|" ,(int(center-doorsize), int(frameSize[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, "|" ,(int(center+doorsize), int(frameSize[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #while tmp <= frameSize[0]:
    #    cv2.putText(frame, "|" ,(int(tmp), int(frameSize[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #    tmp += grid

    if ((sumAreaDetect[0]>frameSize[0]*frameSize[1]*0.9) or (idle_time%55==0 and stalable(sumAreaDetect))):
        firstTrack = None
        firstFrame = None
        status = 'motion'
        bbox = None
        pointMove = []
    # Show our webcam
    #cv2.imshow("Thresh", thresh)
    cv2.imshow("Camera",frame)

    # If we have been tracking for more than a few seconds
    if reset:
        # Reset to motion
        status = 'motion'
        # Reset timer
        idle_time = 0
        print ("reset")
        reset = False
        # Reset background, frame, and tracker
        firstTrack = None
        #firstFrame = None
        bbox = None
        sumAreaDetect = [0,0,0]
        pointMove = []

    # Incriment timer
    idle_time += 1
    if idle_time%2 == 0:
        for i in range (len(sumAreaDetect)-1) :
            sumAreaDetect[i+1] = sumAreaDetect[i]
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup before exit.
vs.release()
cv2.destroyAllWindows()
vs.stop()
