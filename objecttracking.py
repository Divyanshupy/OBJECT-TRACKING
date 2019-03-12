import cv2
import dlib
from imutils.video import FPS
import imutils
import numpy as np

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
net=cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt","mobilenet_ssd/MobileNetSSD_deploy.caffemodel")
vs=cv2.VideoCapture(0)
writer=None 
trackers=[]
labels=[]
fps=FPS().start()
    
while True:
    (grabbed,frame)=vs.read()
    if frame is  None:
        break
    frame=imutils.resize(frame,width=600)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    """if writer is None:
        fourcc=cv2.VideoWriter_fourcc(*"MJPG")
        writer=cv2.VideoWriter("output.avi",fourcc,30,(frame.shape[1],frame.shape[0]),True)"""
    if len(trackers)==0:
        (h,w)=frame.shape[:2]
        blob=cv2.dnn.blobFromImage(frame,0.007843,(w,h),127.5)
        net.setInput(blob)
        detections=net.forward()
        for i in np.arange(0,detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>0.2:
                idx=int(detections[0,0,i,1])
                label=CLASSES[idx]
                if(label!='person'):
                    continue
                box=detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX,startY,endX,endY)=box.astype('int')
                t = dlib.correlation_tracker()
                rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                t.start_track(rgb, rect)
                labels.append(label)
                trackers.append(t)
                cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)
                cv2.putText(frame,label,(startX,startY-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
                print(trackers)
                
                
				    
                
				    
    else:
        for (t,l) in zip(trackers,labels):
            t.update(rgb)
            pos=t.get_position()
            startX=int(pos.left())
            startY=int(pos.top())
            endX=int(pos.right())
            endY=int(pos.bottom())
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0,255,0),2)
            cv2.putText(frame,l,(startX,startY-15),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0))
            
    """if writer is not None:
        writer.write(frame)"""
    cv2.imshow("Frame:",frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
    fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
            
            
                    
                
                    