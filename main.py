
#!/usr/bin/env python3
import image_processor as ip
from hough_lines_prio import hough_lines_prio

##########parameters###############
img_loc = "pictures/06.webp"
court_loc = "pictures/court_reference.png"
dialation = 6
kernel_size = 3
t = 0.94
perc_tp_remove = 0.1
s_factor = 1.1
court_factor = 1
ransac_iters = 1000
angle_threshold = 5 # degrees
percent_borders = 0.3
##########parameters###############

print("Processing {}".format(img_loc))
hl_p = hough_lines_prio()

# image pre processing
im = ip.import_image(img_loc)
court_reference = ip.import_court(court_loc)
court_borders = ip.court_borders(court_reference,court_factor)
gray = ip.gray_scale(im,dialation,kernel_size)
image = hl_p.image_resize(im)

# find hough lines
nlines, score = hl_p.extract_lines(image,im)
nlines = hl_p.score_filtering(im,t,perc_tp_remove)

# find vanishing points and inliers
inliers_list,angles_list = ip.get_vanishing_pts(nlines,ransac_iters, angle_threshold)
ip.plot_inliers(im, nlines, inliers_list)
inliers_list = ip.filter_vps(inliers_list,angles_list)

# section and object forming
section,section_ind,inter_pts = ip.section_form(nlines,inliers_list,s_factor)
target_lines = ip.object_form(section_ind,nlines,inliers_list,s_factor,im)

# border filtering
target_lines_span,target_thetas = ip.line_extend(target_lines,im)
borders = ip.border_order(target_lines_span,target_thetas,im)
borders_filtered = ip.border_filter(borders,percent_borders,im)

#homography matrix calculations
best_court_corners = ip.homography_builder(borders_filtered,court_borders,court_reference,im,gray)
ip.apply_homography(best_court_corners,court_borders,im)

