
#!/usr/bin/env python3

import cv2
import numpy as np
import image_processor
from hough_lines_prio import hough_lines_prio


img_loc = "pictures/05.jpg"
court_loc = "pictures/court_reference.png"
dialation = 5
kernel_size = 3
t = 0.94
perc_tp_remove = 0.1
s_factor = 1.1
court_factor = 1
ransac_iters = 1000
angle_threshold = 5 # degrees
percent_borders = 0.25

print("Processing {}".format(img_loc))

hl_p = hough_lines_prio()

im = image_processor.import_image(img_loc)
court_reference = image_processor.import_court(court_loc)
court_borders = image_processor.court_borders(court_reference,court_factor)
gray = image_processor.gray_scale(im,dialation,kernel_size)
image = hl_p.image_resize(im)
nlines, score = hl_p.extract_lines(image,im)



nlines = hl_p.score_filtering(im,t,perc_tp_remove)

inliers_list,angles_list = image_processor.get_vanishing_pts(nlines,ransac_iters, angle_threshold)
image_processor.visualize_inliers(im, nlines, inliers_list)

inliers_list = image_processor.filter_vps(inliers_list,angles_list)

section,section_ind,inter_pts = image_processor.section_form(nlines,inliers_list,s_factor)
target_lines = image_processor.object_form(section_ind,nlines,inliers_list,s_factor)
target_lines_span,target_thetas = image_processor.line_extend(target_lines,im)

borders = image_processor.border_order(target_lines_span,target_thetas,im)
borders_filtered = image_processor.border_filter(borders,percent_borders,im)

best_court_corners = image_processor.homography_builder(borders_filtered,court_borders,court_reference,im,gray)

image_processor.apply_homography(best_court_corners,court_borders,im)

