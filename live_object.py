import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

config_file = ('ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
frozen_model = ('frozen_inference_graph.pb')

model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

classlabels = []
file_name = 'coco.names'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')

# print (classlabels)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap=cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Camera not Opened")


font_scale = 1
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)
    print(ClassIndex)
    if (len(ClassIndex!=0)):
        for classInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if conf > 0.5:
                cv2.rectangle(frame, box, color= (0, 255, 0), thickness=1)
                cv2.putText(frame, classlabels[classInd - 1], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,  fontScale= 1, color=(255, 0, 0), thickness=2)
    cv2.imshow("object detection" , frame)
    if cv2.waitKey(25)& 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
