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
from random import randint

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

def findTrack(bbox,pointMove):
    index = -1
    if pointMove:
        # set dif of last
        diffMinX = abs(pointMove[len(pointMove)-1][len(pointMove[len(pointMove)-1])-1][0] - bbox[0])
        diffMinY = abs(pointMove[len(pointMove)-1][len(pointMove[len(pointMove)-1])-1][1] - bbox[1])
        diffMin = (pow(diffMinX,2) + pow(diffMinY,2))**(1/2)
        index = len(pointMove)-1
        for i in range(len(pointMove)-1):
            diffX = abs(pointMove[i][len(pointMove[i])-1][0] - bbox[0])
            diffY = abs(pointMove[i][len(pointMove[i])-1][1] - bbox[1])
            diff = (pow(diffX,2) + pow(diffY,2))**(1/2)
            if diff < diffMin:
                index = i
                diffMin = diff
        if diffMin > 50 + bbox[2]:
            index = -1
    return index

def tooClose(bboxs,rec,frameSize):
    (x,y,w,h) = rec
    print(rec)
    rec = setCenter(rec)
    for bbox in bboxs:
        bbox = setCenter(bbox)
        difX = abs(bbox[0] - rec[0])
        difY = abs(bbox[1] - rec[1])
        if difX < bbox[0]-rec[0]+10:
            return False
        if difY < bbox[1]-rec[1]+30:
            return False
    return True

def setCenter(box):
    return (int(box[0]+box[2]/2),int(box[1]+box[3]/2))
        
# Are we using the Pi Camera?
usingPiCamera = True
# Set initial frame size.
frameSize = (600, 400)

countPeople = 0

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
#ap.add_argument("-a", "--min-area", type=int, default=frameSize[0]*frameSize[1]*0.004167, help="minimum area size")
ap.add_argument("-a", "--min-area", type=int, default=frameSize[0]*frameSize[1]*0.004, help="minimum area size")
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
firstTrack = []
status = 'motion'
# How long have we been tracking
idle_time = 30
reset = False

grid = frameSize[0]/5
doorsize = frameSize[0]/6
center = frameSize[0]/2
bbox = []
sumAreaDetect  = [0,0,0]
pointMove = []
lencnts = 0
trackCount = 0

pathToDB = "/home/pi/FaceRecognizer/sorted_output_images"
#train
#igen_model, people= faceRec.train_model(pathToDB)
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
    gray = cv2.equalizeHist(gray)
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
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None and idle_time > 8:
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
        cntsArea = []
        if lencnts != 0:
            # Set largest contour to first contour
            largest = 0
            # For each contour
            for i in range(lencnts):
                (x,y,w,h) = cv2.boundingRect(cnts[i])
                recArea =  cv2.contourArea(cnts[i])
                if idle_time%2 == 0:
                    sumArea += recArea
                cntsArea.append(recArea)
                # If this contour is larger than the largest
                #if i != 0 and int(cv2.contourArea(cnts[i])) > int(cv2.contourArea(cnts[largest])):
                    # This contour is the largest
                    #largest = i
                    #larger = largest
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #sort array
            for passnum in range(lencnts-1,0,-1):
                for i in range(passnum):
                    if cntsArea[i] < cntsArea[i+1]:
                        temp = cntsArea[i]
                        temp1 = cnts[i]
                        cntsArea[i] = cntsArea[i+1]
                        cnts[i] = cnts[i+1]
                        cntsArea[i+1] = temp
                        cnts[i+1] = temp1
            trackCount=int(lencnts/6)
            #select
            bbox = []
            for i in range(lencnts) :
                if cntsArea[i] > args["min_area"]:
                    # Create a bounding box for our contour
                    (x,y,w,h) = cv2.boundingRect(cnts[i])
                    # Convert from float to int, and scale up our boudning box
                    (x,y,w,h) = (int(x),int(y),int(w),int(h))
                    rec=(x,y,w,h)
                    if bbox==[] or tooClose(bbox,rec,frameSize):

                        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                        # Initialize tacker
                        status = 'tracking'
                        if len(bbox) < trackCount:
                            bbox.append((x,y,w,h))
                            #pointMove.append([])
                        else :
                            break
        if idle_time%2 == 0:
            sumAreaDetect[0] = sumArea
    # If we are tracking
    if status == 'tracking':
        for i in range(len(bbox)):
            # Create a visible rectangle for our viewing pleasure
            p0 = (int(bbox[i][0]), int(bbox[i][1]))
            p2 = (int(bbox[i][0] + bbox[i][2]), int(bbox[i][1] + bbox[i][3]))

            #set center
            p1 = setCenter(bbox[i])
            #find continous track
            indexTrack = findTrack(bbox[i],pointMove)
            #set new track
            if indexTrack == -1:
                indexTrack = len(firstTrack)
                firstTrack.append(p1[0])
                pointMove.append([])
                pointMove[indexTrack].append(p1)
                continue
            cv2.rectangle(frame,p0,p2,(255,int(255/(indexTrack+1)),50*indexTrack),2)
            
            #check line movement
            if abs(p1[0]-pointMove[indexTrack][len(pointMove[indexTrack])-1][0]) > grid + bbox[i][2]:
                reset = True
            else : pointMove[indexTrack].append(p1)
            if len(pointMove[indexTrack]) > 3 :
                pointMove[indexTrack].pop(0)

            #come in
            if not reset :
                if (firstTrack[indexTrack] > center-doorsize and firstTrack[indexTrack] < center + doorsize):
                    if (p1[0] <= grid or p1[0] >= frameSize[0] - grid):
                        #if stalable(sumAreaDetect) or p1[0]<=0 or p2[0] >= frameSize[0]:
                        countPeople += 1
                        reset = True
                    #if p1[0] <= grid*4 and p1[0] >= grid*2:
                        #filename = "/home/pi/in/"+str(time.time())+".jpg"
                        #cv2.imwrite(filename, frame)
                elif firstTrack[indexTrack] <= grid or firstTrack[indexTrack] >= frameSize[0]-grid:
                    if p1[0] > center - doorsize and p1[0] < center + doorsize:
                        if stalable(sumAreaDetect) :
                            if countPeople > 0:
                                countPeople -= 1
                                reset = True
        if len(firstTrack)>trackCount:
            for i in range(len(firstTrack)):
                if not ((firstTrack[i] > center-doorsize and firstTrack[i] < center + doorsize) or (firstTrack[i] < grid or firstTrack[i]>frameSize[0]-grid)):
                    #if abs(firstTrack[i]-pointMove[i][len(pointMove)-1]):
                    firstTrack.pop(i)
                    pointMove.pop(i)
                    break
        print(pointMove)
        print(firstTrack)
    #draw
    '''
    for line in pointMove :
        i = pointMove.index(line)+1
        y = 10*i
        color = (i*60,int(255/i),200)
        for j in range(len(line)-1):
            cv2.line(frame, (line[j], y), (line[j+1], y), color, 2)
    '''
    cv2.putText(frame, "People: "+str(countPeople), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    tmp = 0
    cv2.putText(frame, "|" ,(int(center-doorsize), int(frameSize[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, "|" ,(int(center+doorsize), int(frameSize[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #while tmp <= frameSize[0]:
    #    cv2.putText(frame, "|" ,(int(tmp), int(frameSize[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #    tmp += grid

    if ((sumAreaDetect[0]>frameSize[0]*frameSize[1]*0.9) or (idle_time%55==0 and stalable(sumAreaDetect))):
        firstTrack = []
        firstFrame = None
        status = 'motion'
        bbox = []
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
        print("reset")
        reset = False
        # Reset background, frame, and tracker
        firstTrack = []
        #firstFrame = []
        bbox = []
        sumAreaDetect = [0,0,0]
        pointMove = []

    # Incriment timer
    idle_time += 1
    if idle_time%2 == 0:
        for i in range (len(sumAreaDetect)-1) :
            sumAreaDetect[i+1] = sumAreaDetect[i]
    cv2.waitKey(0)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

# Cleanup before exit.
vs.release()
cv2.destroyAllWindows()
vs.stop()
