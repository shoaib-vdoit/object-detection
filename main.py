import cv2

#reading frames from the objects
cap = cv2.VideoCapture("27260-362770008_medium.mp4")

#object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)


#we extracting frames one after another
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (620, 380),fx=0,fy=0,interpolation=cv2.CHAIN_APPROX_SIMPLE)
    height, width, _ = frame.shape
    

    #extract region of interest
    roi = frame[110:360, 0:620]

    #object detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        #calculate the area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 265, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    


    key = cv2.waitKey(20)
    if key == 15:    #for closing the video through S key
        break

cap.release()
cv2.destroyAllWindows()