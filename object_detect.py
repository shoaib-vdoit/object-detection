import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

config_file = ('ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
frozen_model = ('frozen_inference_graph.pb')

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classlabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')

# print (classlabels)

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

img1=cv2.imread('img.jpg')



ClassIndex, confidence, bbox = model.detect(img1, confThreshold=0.5)
print(ClassIndex)

if len(ClassIndex) > 0:
    for classInd, conf, box in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        if conf > 0.5:
            cv2.rectangle(img1, box, color= (0, 255, 0), thickness=1)
            cv2.putText(img1, classlabels[classInd - 1], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,  fontScale= 1, color=(255, 0, 0), thickness=2)
        
cv2.imshow('Detected Objects', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
