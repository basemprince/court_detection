import cv2
import numpy as np

# =============================================================================
# img = cv2.imread('pictures/02.png')
# 
# pts = np.array([[105, 445], [1190, 445], [780, 150], [510, 150]], dtype=np.float32)
# for pt in pts:
#     cv2.circle(img, tuple(pt.astype(np.int)), 6, (0,0,255), -1)
# cv2.imshow('img', img)
# cv2.waitKey()
# # compute IPM matrix and apply it
# ipm_pts = np.array([[0,0], [0,720], [1280,720], [1280,0]], dtype=np.float32)
# ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)
# ipm = cv2.warpPerspective(img, ipm_matrix, img.shape[:2][::-1])
# 
# # display (or save) images
# cv2.imshow('ipm', ipm)
# cv2.waitKey()
# =============================================================================


img2 = cv2.imread('pictures/court_reference.png')
template_points = np.asarray([[147, 1839], [970, 1839], [10, 2388], [1107, 2388]])
for pt in template_points:
    cv2.circle(img2, tuple(pt.astype(np.int)), 6, (0,0,255), -1)
img2 = cv2.resize(img2, (400, 900))  
cv2.imshow('img', img2)
cv2.waitKey()