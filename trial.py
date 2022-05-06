
#OpenCV 4.3.0, Raspberry Pi 3/B/4B-w/4/8GB RAM, Buster,v10.
#Date: 3rd, June, 2020

import cv2
import numpy as np
import random as rng
font = cv2.FONT_HERSHEY_SIMPLEX    

frame = cv2.imread('pictures/test.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (25, 25), 1.5)
edge = cv2.Canny(blur, 100,200)
contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
drawing = np.zeros((edge.shape[0], edge.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
# Show in a window
cv2.imshow('Contours', drawing)
cv2.waitKey(0)

for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, ('width = {}, height = {}'.format(w, h)),
                (x+30, y+30),
                font,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA)

# cv2.imshow('Measure Size', frame)
# cv2.waitKey(0)