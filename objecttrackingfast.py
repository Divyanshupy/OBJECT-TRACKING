# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:24:28 2018

@author: Divyanshu
"""

from imutils.video import FPS
import multiprocessing
import numpy as np
import imutils 
import dlib
import cv2
def start_tracker(box,label,rgb,inputQueue,outputQueue):
    t=dlib.correlation_tracker()
    rect=dlib.rectangle(box[0],box[1],box[2],box[3])
    t.start_track(rgb,rect)
    while True:
        rgb=inputQueue.get()
        if rgb is not None:
            t.update(rgb)
            pos=t.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            outputQueue((label,(startX,startY,endX,endY)))
inputQueues = []
outputQueues = []
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
net=cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt","mobilenet_ssd/MobileNetSSD_deploy.caffemodel")
vs=cv2.VideoCapture("race.mp4")
writer=None
fps=FPS().start()
while True:
    (grabbed,frame)=vs.read()
    if frame is None:
        break
    frame=imutils.resize(frame,width=600)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if writer is None:
        fourcc=cv2.VideoWriter_fourcc(*"MJPG")
        writer=cv2.VideoWriter("fast.avi",fourcc,30,(frame.shape[1],frame.shape[1]),True)
    if len(inputQueues)==0:
        (h,w)=frame.shape[:2]
        blob=cv2.dnn.blobFromImage(frame,0.007843,(w,h),127.5)
        net.setInput(blob)
        detections=net.forward()
        for i in np.arange(0,detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>0.2:
                idx=int(detections[0,0,i,1])
                label=CLASSES[idx]
                if CLASSES[idx]!="person":
                    continue
                box=detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX,startY,endX,endY)=box.astype("int")
                bb=(startX,startY,endX,endY)
                iq=multiprocessing.Queue()
                oq=multiprocessing.Queue()
                inputQueues.append(iq)
                outputQueues.append(oq)
                p=multiprocessing.Process(
                        target=start_tracker,
                        args=(bb,label,rgb,iq,oq))
                p.daemon=True
                p.start()
                cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)
                cv2.putText(frame,label,(startX,startY-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
                if writer is not None:
                    writer.write(frame)
                cv2.imshow("Frame",frame)
                key=cv2.waitKey(1)&0xFF
                if key==ord('q'):
                    break
                fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
vs.release()

                


