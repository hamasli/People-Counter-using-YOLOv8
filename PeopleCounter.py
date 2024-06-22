import numpy as np;
import cv2;
from ultralytics import YOLO
import cvzone
import math;
from sort import*

#for video
cap=cv2.VideoCapture('people.mp4');

#loading model
model=YOLO('../yolov8-weights/yolov8n.pt');

classfile='coco.names';
classNames=[];
with open(classfile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n');
print(classNames);

#import mask
mask=cv2.imread('peoplearea.png')

#crating sort instance
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3);
#for just total count use single one line
# limitsDown=[150,297,673,297];
#but for seperate seperate for both up and down
limitsUp=[103,161,296,161];
limitsDown=[527,489,760,489]



totalCountUp=[];
totalCountDown=[];
while True:
    success, image = cap.read()
    imgRegion = cv2.bitwise_and(image, mask)

    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    image = cvzone.overlayPNG(image, imgGraphics, (730, 260))
    results = model(imgRegion, stream=True)
    detection=np.empty((0,5));
    for r in results:
        boxes=r.boxes;
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0];
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2);
            w,h=x2-x1,y2-y1;
            bbox=int(x1),int(y1),int(w),int(h);
            #confidence
            conf=math.ceil((box.conf[0]*100))/100;
            # print(conf);
            cls=int(box.cls[0]);
            currentClass=classNames[cls];
            if currentClass=="person" and conf>0.3:
                currentArray=np.array(([x1,y1,x2,y2,conf]));
                detection=np.vstack((detection,currentArray));


    resultsTracker=tracker.update(detection);
    cv2.line(image,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,0,255),5);
    cv2.line(image, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5);
    for result in resultsTracker:
        x1,y1,x2,y2,Id=result;
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2);
        print(result);
        w, h = x2 - x1, y2 - y1;

        cvzone.cornerRect(image,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,255));
        cvzone.putTextRect(image, f'{int(Id)}', (max(0, x1), max(35, y1)), offset=10, thickness=3,
                           scale=2);
        #formula for getting the center of object using x,y,w,h of bounding box to detect center of object.
        cx,cy=x1+w//2,y1+h//2;
        cv2.circle(image,(cx,cy),5,(255,0,255),cv2.FILLED);

        #now we are checking that each object crossed the line or not. using limits up and down x1,x2,y1,y1
        if limitsUp[0]<cx<limitsUp[2] and limitsUp[1]-15<cy<limitsUp[1]+15:
            if totalCountUp.count(Id)==0:
                totalCountUp.append(Id)
                cv2.line(image, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5);

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(Id) == 0:
                totalCountDown.append(Id)
                cv2.line(image, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5);

    cv2.putText(image, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(image, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)
    cv2.imshow("image",image);
    # cv2.imshow("ImageRegion",imgRegion);
    cv2.waitKey(1);
