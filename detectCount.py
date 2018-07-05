# USAGE
# python detect.py --images images

# import the necessary packages
from __future__ import print_function
import imutils
from imutils.video import VideoStream
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import cv2
import sys, time
import os

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
def pedestrian (image, frame):
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
    orig = image.copy()
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs = None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 100, 100), 2)
    return image
def calArea(rec):
    return abs(rec[0]-rec[2]) * abs(rec[1]-rec[3])
def defineWH(rec):
    return (rec[0],rec[1],rec[2]-rec[0],rec[3]-rec[1])

usingPiCamera = True
frameSize = (300, 200)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-i", "--images", help="path to images directory")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

if args["video"] is None :
    vs = VideoStream(src=0, usePiCamera=usingPiCamera, resolution=frameSize, framerate=32).start()
else :
    vs = cv2.VideoCapture(args["video"])
    usingPiCamera = False
time.sleep(2.0)
firstFrame = None
firstTrack = None
status = 'motion'
# How long have we been tracking
idle_time = 30
reset = False

grid = frameSize[0]/5
doorsize = frameSize[0]/5
center = frameSize[0]/2
bboxs = []
sumAreaDetect  = [0,0,0]
pointMove = []
lencnts = 0
countPeople = 0
last_20 = [0 for i in range(20)]
final_5 = []
box_text= "Subject: "

while True:
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
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

    orig = frame.copy()
    
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
	    padding=(8, 8), scale=1.05)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayFace = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(grayFace, (21, 21), 0)
    # if the first frame is None, initialize it
    if firstFrame is None and idle_time >= 0:
        firstFrame = gray
        continue
    if idle_time >= 0:
        # Difference between current frame and background
        frame_delta = cv2.absdiff(firstFrame,gray)
        # Create a threshold to exclude minute movements
        thresh = cv2.threshold(frame_delta,100,255,cv2.THRESH_BINARY)[1]

        #Dialate threshold to further reduce error
        thresh = cv2.dilate(thresh,None,iterations=2)
        # Check for contours in our threshold
        _,motion,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # Check each contour
        sumArea = 0
        largest = 0
        for i in range(len(motion)):
            (x,y,w,h) = cv2.boundingRect(motion[i])
            recArea =  cv2.contourArea(motion[i])
            if idle_time%2 == 0:
                sumArea += recArea
            # If this contour is larger than the largest
            if i != 0 and int(cv2.contourArea(motion[i])) > int(cv2.contourArea(motion[largest])):
                #This contour is the largest
                largest = i

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255),2)
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    lencnts = len(pick)
    sumArea= 0
    cntsArea = []
    bboxs = []
    if lencnts != 0:
        # draw the final bounding boxes
        for i in range(len(pick)):
            recArea =  calArea(pick[i])
            if recArea > 10 :
                (xA, yA, xB, yB) = pick[i]
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                rec = defineWH(pick[i])
                bboxs.append(rec)
                status = 'tracking'

    # show the output images
    if status == 'tracking':
        for bbox in bboxs:
        # Create a visible rectangle for our viewing pleasure
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame,p1,p2,(0,0,255),2)
            #set center
            p1 = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))

            if firstTrack is None:
                firstTrack = p1[0]
                print(firstTrack)
                continue
            #check line movement
            if not pointMove:
                pointMove.append(p1[0])
            #if abs(p1[0]-pointMove[len(pointMove)-1]) > grid + bbox[2]:
            #    reset = True
            else : pointMove.append(p1[0])
            if len(pointMove) > 3 :
                pointMove.pop(0)
            print(pointMove)
            #come in
            if not reset :
                if (firstTrack > center-doorsize and firstTrack < center + doorsize):
                    if (p1[0] <= center-doorsize or p1[0] >= center+doorsize):
                        #if stalable(sumAreaDetect) or p1[0]<=0 or p2[0] >= frameSize[0]:
                        countPeople += 1
                        reset = True
                    #if p1[0] <= grid*4 and p1[0] >= grid*2:
                        #filename = "/home/pi/in/"+str(time.time())+".jpg"
                        #cv2.imwrite(filename, frame)
                elif firstTrack <= grid or firstTrack >= frameSize[0]-grid:
                    if p1[0] > center - doorsize and p1[0] < center + doorsize:
                        if stalable(sumAreaDetect) :
                            if countPeople > 0:
                                countPeople -= 1
                                reset = True
    cv2.putText(frame, "People: "+str(countPeople), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "|" ,(int(center-doorsize), int(frameSize[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, "|" ,(int(center+doorsize), int(frameSize[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #while tmp <= frameSize[0]:
    #    cv2.putText(frame, "|" ,(int(tmp), int(frameSize[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #    tmp += grid
    #reset frame
    if ((sumAreaDetect[0]>frameSize[0]*frameSize[1]*0.9) or (idle_time%55==0 and stalable(sumAreaDetect))):
        firstTrack = None
        firstFrame = None
        status = 'motion'
        bboxs = []
        pointMove = []

    # Show our webcam
    #cv2.imshow("frame_delta", frame_delta)
    #cv2.imshow("Thresh", thresh)

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
        bboxs = []
        sumAreaDetect = [0,0,0]
        pointMove = []

    # Incriment timer
    idle_time += 1
    if idle_time%2 == 0:
        for i in range (len(sumAreaDetect)-1) :
            sumAreaDetect[i+1] = sumAreaDetect[i]

    cv2.imshow("countPeople", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vs.release()
cv2.destroyAllWindows()
vs.stop()