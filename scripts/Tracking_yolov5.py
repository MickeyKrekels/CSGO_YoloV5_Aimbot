import cv2
from numpy.lib.function_base import append
import torch
import numpy as np
import matplotlib.pyplot as plt

#extra parameters
videoPath = './video/CSGO_original_training_vid.mp4' 

input_size = 416
color_box = (60,179,113) # mediumseagreen
color_circle = (0,0,255) # red

class_names = ['Person','Head']

#load video
cap = cv2.VideoCapture(videoPath)
# background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)

#load yolo 
model = torch.hub.load('./yolov5', 'custom', path='./weights/YoloV5-CSGO-Weights-Trained-V1_Nano_Best.pt', source='local') 
model.conf = 0.40  # NMS confidence threshold

while True:
    red, frame = cap.read()

    # mask = background_subtractor.apply(frame)
    
    #---if there are no more frames aviable break loop---#
    if not red:
        break
    
    img = cv2.resize(frame, (input_size, input_size))  
    
     #---predict objects----#
    result = model(img)
    prediction = result.xyxy[0]

    boxes       = []
    centerpoint = []
    scores      = []
    classes     = []

     #---formating the predicted values----#
    for i in range(len(prediction)):
        x = int(prediction[i][0]) #box values
        y = int(prediction[i][1])
        w = int(prediction[i][2])
        h = int(prediction[i][3])

        cx = int((x + w) /2) #caculate the center point
        cy = int((y + h) /2)

        score = prediction[i][4]
        predicted_class = int(prediction[i][5])

        #---saving the values----#
        box = [x,y,w,h]
        boxes.append(box)

        centerpoint.append((cx,cy))
        scores.append(score)
        classes.append(predicted_class)

    for i in range(len(boxes)):
        box = boxes[i]
        info_to_display = class_names[classes[i]]

        cv2.circle(img,centerpoint[i],2, color_circle ,-1)
        cv2.rectangle(img,(box[0],box[1]),(box[2], box[3]),color_box,1)

        # cv2.rectangle(img, (int(box[0]), int(box[1]-15)), (int(box[0])+(len(info_to_display))*8,
        #                                                       int(box[1])), color_box, -1)     

        # cv2.putText(img, info_to_display, (int(box[0]), int(box[1]-5)), 0, 0.4,
        #               (255, 255, 255), 1)

    #---display the predicted image----#
    cv2.imshow("frame",img)
    key = cv2.waitKey(1)
    if(key == 24):
        break

cap.release()
cv2.destroyAllWindows()

