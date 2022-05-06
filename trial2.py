import numpy as np
import cv2
import scipy.ndimage as ndi

img = cv2.imread('pictures/02.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth = ndi.filters.median_filter(gray, size=2)
edges = smooth > 200

lines = cv2.HoughLines(edges.astype(np.uint8), 0.5, np.pi/180, 120)

for line in lines:
    for rho,theta in line:
        print(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# Show the result

cv2.imshow('Measure Size', img)
cv2.waitKey(0)