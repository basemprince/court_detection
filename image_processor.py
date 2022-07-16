#!/usr/bin/env python3
import skimage.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from functools import cmp_to_key
import matplotlib.pyplot as plt
from scipy.spatial import distance
from itertools import combinations

def import_image(img_loc):
    return skimage.io.imread(img_loc) 

def import_court(court_loc):
    return cv2.imread(court_loc, 0)
    
def court_borders(court_reference,court_factor = 1):
    width = court_reference.shape[1]
    height = court_reference.shape[0]
    court_borders = np.asarray([[0, 0], [width/court_factor, 0], [width/court_factor, height*court_factor], [0, height*court_factor]])
    return  np.float32(court_borders).reshape(-1,1,2)

def gray_scale(image,dialation=3,kernel_size=3):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.dilate(gray,kernel,iterations=dialation)
    gray[gray > 0] = 1 
    return gray

def get_intersection_1(a1, a2, b1, b2):
    """ 
    returns intersection point between 2 lines
    a1,a2 points of first line
    a3,a4 points of second line
    """
    v_stacked = np.vstack([a1,a2,b1,b2])     
    h_stacked = np.hstack((v_stacked, np.ones((4, 1))))
    
    line_1 = np.cross(h_stacked[0], h_stacked[1])         
    line_2 = np.cross(h_stacked[2], h_stacked[3])           
    x, y, z = np.cross(line_1, line_2)
    if z == 0:  # no intersection
        return (float('inf'), float('inf'))
    x = x/z
    y = y/z
    return (x, y)


def get_intersection_2(a1, a2, b1, b2):
    x1,y1 = a1
    x2,y2 = a2
    x3,y3 = b1
    x4,y4 = b2
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return [x,y]

def get_angles(vp_pt, lines, ignored_pts, angle_threshold):
    vp_pt = vp_pt / vp_pt[-1]
    vp_direction = vp_pt[:2] - lines[:,0]
    lines_direction = lines[:,1] - lines[:,0]
    magnitude = np.linalg.norm(vp_direction, axis=1) * np.linalg.norm(lines_direction, axis=1)
    magnitude[magnitude == 0] = 1e-5
    cosine = np.clip((vp_direction*lines_direction).sum(axis=-1) / magnitude,-1,1)
    angle = np.arccos(np.abs(cosine))

    inliers = (angle < np.radians(angle_threshold))
    inliers[ignored_pts] = False
    vote_counter = inliers.sum()
    return inliers, vote_counter, angle

def ransac(lines, iter_num, angle_threshold, ignored_pts=None):
    best_votes = 0
    best_inliers = None
    best_vp = None
    best_angle = None
    if ignored_pts is None:
        ignored_pts = np.zeros((lines.shape[0])).astype('bool')
        lines_chosen = np.arange(lines.shape[0])
    else:
        lines_chosen = np.where(ignored_pts==0)[0]
    for i in range(iter_num):
        idx1, idx2 = np.random.choice(lines_chosen, 2, replace=False)
        l1 = np.cross(np.append(lines[idx1][1], 1), np.append(lines[idx1][0], 1))
        l2 = np.cross(np.append(lines[idx2][1], 1), np.append(lines[idx2][0], 1))

        vp_pt = np.cross(l1, l2)
        if vp_pt[-1] == 0:
            continue
        inliers, vote_counter,theta = get_angles(vp_pt, lines, ignored_pts, angle_threshold)
        if vote_counter > best_votes:
            best_votes = vote_counter
            best_vp = vp_pt
            best_inliers = inliers
            best_angle = theta
    return best_inliers, best_angle

def get_vanishing_pts(nlines, iterations, threshold):

    best_inliers_1, best_angle_1 = ransac(nlines, iterations, threshold)
    best_inliers_2,best_angle_2 = ransac(nlines, iterations, threshold, ignored_pts=best_inliers_1)
    ignored_pts = np.logical_or(best_inliers_1, best_inliers_2)
    best_inliers_3,best_angle_3 = ransac(nlines, iterations, threshold, ignored_pts=ignored_pts)
    inliers_list = [best_inliers_1, best_inliers_2, best_inliers_3]
    best_angle_list = [best_angle_1,best_angle_2,best_angle_3]
    return inliers_list,best_angle_list

def visualize_inliers(im, lines, inliers_list):
    colors = [(1.0, 22.0/255.0, 12.0/255.0), (102.0/255.0, 255.0/255.0, 0.0/255.0), 'b']
    subplot_count = len(inliers_list)

    fig, axes = plt.subplots(1, subplot_count, figsize=(15, 15), sharex=True, sharey=True)
    ax = axes.ravel()
    for i in range(len(inliers_list)):
        ax[i].imshow(im)
        for line in lines[inliers_list[i]]:
            p0, p1 = line
            ax[i].plot((p0[1], p1[1]),(p0[0], p1[0]), c=colors[i])
        ax[i].set_xlim((0, im.shape[1]))
        ax[i].set_ylim((im.shape[0], 0))
        ax[i].set_title('VP #{} Inliers'.format(str(i+1)),fontsize=26)

    for a in ax:
        a.set_axis_off()

    plt.tight_layout() 
    plt.show()
    plt.close()

def filter_vps(inliers_list,angles_list):
    average_anlges = np.degrees(np.average(angles_list,axis=1))
    inliers_list=np.delete(inliers_list,np.argmax(average_anlges),axis=0)
    return inliers_list

def section_form(nlines,inliers_list,s_factor):
    section = []
    section_ind = []
    inter_pts = []
    for i,line1 in enumerate(nlines[inliers_list[0]]):
        a0,a1 = line1
        a0 = [a0[0] *(1+s_factor)/2 + a1[0] *(1-s_factor)/2 , a0[1] *(1+s_factor)/2 + a1[1] *(1-s_factor)/2]
        a1 = [a1[0] *(1+s_factor)/2 + a0[0] *(1-s_factor)/2 , a1[1] *(1+s_factor)/2 + a0[1] *(1-s_factor)/2]
        plt.plot((a0[1], a1[1]),(a0[0], a1[0] ), 'r')
        counter = 0
        for j,line2 in enumerate(nlines[inliers_list[1]]):
            b0,b1 = line2
            b0 = [b0[0] *(1+s_factor)/2 + b1[0] *(1-s_factor)/2 , b0[1] *(1+s_factor)/2 + b1[1] *(1-s_factor)/2]
            b1 = [b1[0] *(1+s_factor)/2 + b0[0] *(1-s_factor)/2 , b1[1] *(1+s_factor)/2 + b0[1] *(1-s_factor)/2]
            plt.plot((b0[1], b1[1]),(b0[0], b1[0] ), 'b')
            intersect_pt = get_intersection_2(a0,a1,b0,b1)
            # print (intersect_pt)
            if intersect_pt is not None:
                plt.scatter(intersect_pt[1],intersect_pt[0],c='k',s=65)
                if(counter==0):
                    section.append([line1,line2])
                    section_ind.append([i,j+len(inliers_list[0])])
                    inter_pts.append([intersect_pt])
                else:
                    section[-1].append(line2)
                    section_ind[-1].append(j+len(inliers_list[0]))
                    inter_pts[-1].append(intersect_pt)
                counter +=1
                
    plt.gca().invert_yaxis()
    plt.tight_layout() 
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()
    plt.close()
    return section,section_ind,inter_pts

def object_form(section_ind,nlines,inliers_list,s_factor):
    merged_ind = section_ind
    obj_ind = []
    while len(merged_ind)>0:
        first, *rest = merged_ind
        first = set(first)

        lf = -1
        while len(first)>lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2

        obj_ind.append(first)
        merged_ind = rest
    
    for sec in obj_ind:
        r = np.random.rand(3,)
        for line1 in sec:
            if line1< len(inliers_list[0]):      
                a0,a1 = nlines[inliers_list[0]][line1]
            else:
                line1-= len(inliers_list[0])
                a0,a1 = nlines[inliers_list[1]][line1]
            a0 = [a0[0] *(1+s_factor)/2 + a1[0] *(1-s_factor)/2 , a0[1] *(1+s_factor)/2 + a1[1] *(1-s_factor)/2]
            a1 = [a1[0] *(1+s_factor)/2 + a0[0] *(1-s_factor)/2 , a1[1] *(1+s_factor)/2 + a0[1] *(1-s_factor)/2]
            plt.plot((a0[1], a1[1]),(a0[0], a1[0] ), c=r)
    plt.gca().invert_yaxis()
    plt.tight_layout() 
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()
    plt.close()

    biggest = -math.inf
    ind1= None
    for ind, obj in enumerate(obj_ind):
        length_obj = len(obj)
        if length_obj >= biggest:
            biggest = length_obj
            ind1 = ind

    main_obj = obj_ind[0]

    target_lines = [[],[]]

    for line1 in main_obj:
        if line1< len(inliers_list[0]):      
            a0,a1 = nlines[inliers_list[0]][line1]
            cc = [0.0,0.0,1.0]
            target_lines[0].append([a0,a1])
        else:
            line1-= len(inliers_list[0])
            a0,a1 = nlines[inliers_list[1]][line1]
            target_lines[1].append([a0,a1])
            cc = [0,1,0]
        plt.plot((a0[1], a1[1]),(a0[0], a1[0] ), c=cc)
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()
    return target_lines

def midpoint(line):
    return ((line[0][0]+line[1][0])/2, (line[0][1]+line[1][1])/2)

def find_line_eq (line,im):
    x_f = 0
    x_e = im.shape[1]
    y_f = 0
    y_e = im.shape[0]
    y_coords, x_coords = zip(*line)
   
   
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(A, y_coords,rcond=None)[0]
   
    theta = math.degrees(math.atan(m))
    
    x0 = (y_f-c)/m
    x1 = (y_e-c)/m

    if x0 > x_e or x0< x_f:
        x0 = np.clip(x0,0,x_e)
        y0 = (m * x0) + c
    else:
        y0 = y_f

    if x1 > x_e or x1< x_f:
        x1 = np.clip(x1,0,x_e)
        y1 = (m * x1) + c
    else:
        y1 = y_e

    return [[y0,x0],[y1,x1]] , theta

def line_extend(target_lines,im):
    target_lines_span = [[],[]]
    target_thetas = [[],[]]
    for i, parallel_lines in enumerate(target_lines):
        for line in parallel_lines:
            new_line, theta = find_line_eq(line,im)
            target_lines_span[i].append(new_line)
            target_thetas[i].append(theta)

    target_thetas[0] = sum([abs(ele) for ele in target_thetas[0]]) / len(target_thetas[0])
    target_thetas[1] = sum([abs(ele) for ele in target_thetas[1]]) / len(target_thetas[1])
    return target_lines_span,target_thetas


def check_side (line1,line2):
   
    inter = get_intersection_2(line1[0],line1[1],line2[0],line2[1])
    x_1 = line1[0][0]
    x_2 = line1[1][0]
    y_1 = line1[0][1]
    y_2 = line1[1][1]
    fx = x_2 - x_1
    fy = y_2 -y_1
    check = []

    if inter is not None:

        mid_x2, mid_y2 = midpoint(line2)
        torque = fx*(mid_y2-y_1)-fy*(mid_x2-x_1)
        if  torque>=0:
            # "point on left side"
            return 1
        else:
            return -1

    for pt in line2:
        torque = fx*(pt[1]-y_1)-fy*(pt[0]-x_1)
        if  torque>=0:
            # "point on left side"
            check.append(1)
        else:
            check.append(0)
    if all(check) == True:
        return 1
    else:
        return -1
 
def check_side_flipped (line1,line2):

    inter = get_intersection_2(line1[0],line1[1],line2[0],line2[1])
    x_1 = line1[0][0]
    x_2 = line1[1][0]
    y_1 = line1[0][1]
    y_2 = line1[1][1]
    fx = x_2 - x_1
    fy = y_2 -y_1
    check = []

    if inter is not None:
        mid_x2, mid_y2 = midpoint(line2)
        torque = fy*(mid_y2-y_1)-fx*(mid_x2-x_1)
       
        if  torque>=0:
            # "point on left side"
            return 1
        else:
            return -1

    for pt in line2:
        torque = fy*(pt[1]-y_1)-fx*(pt[0]-x_1)
        if  torque>=0:
            # "point on left side"
            check.append(1)
        else:
            check.append(0)
    if all(check) == True:
        return 1
    else:
        return -1
 
def border_order(target_lines_span,target_thetas,im):
    your_key = cmp_to_key(check_side)
    your_key_flipped = cmp_to_key(check_side_flipped)

    borders = []

    for i in range(len(target_lines_span)):
        if target_thetas[i] > 20:
        
            borders.append(sorted(target_lines_span[i], key=your_key))
        else:
            borders.append(sorted(target_lines_span[i], key=your_key_flipped))


    fig, axes = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)
    ax = axes.ravel()

    for j in range(len(borders)):
        if j == 0:
            num = 16
            num1 = 1
            st = -16
        else:
            num = 1
            num1 = 1
            st = 1
        ax[j].imshow(im)
        for i, vert in enumerate(borders[j]):
            ax[j].plot((vert[0][1], vert[1][1]),(vert[0][0], vert[1][0] ), c=[1,0,0])
            ax[j].text(st+(vert[0][1]+vert[1][1])/2+(i*num),(vert[0][0]+vert[1][0])/2+(i*num1),str(i),fontsize=20)
        ax[j].set_xlim((0, im.shape[1]))
        ax[j].set_ylim((im.shape[0], 0))
        ax[j].set_title('sorted lines {}'.format(str(j+1)),fontsize=26)

    for a in ax:
        a.set_axis_off()

    plt.tight_layout() 
    plt.show()
    plt.close()
    return borders

def border_filter(borders,percent_borders,im):
    borders_filtered = [[],[]]
    vert_len = math.ceil(math.ceil(percent_borders*len(borders[0])) / 2.) * 2
    hor_len = math.ceil(math.ceil(percent_borders*len(borders[1])) / 2.) * 2
    for j in range(vert_len):
        borders_filtered[0].append(borders[0][j])
        if j!=0:
            borders_filtered[0].append(borders[0][-j])
    for j in range(hor_len):
        borders_filtered[1].append(borders[1][j])
        if j!=0:
            borders_filtered[1].append(borders[1][-j])

    
    for i in borders_filtered:
        for j in range(len(i)):
            plt.plot((i[j][0][1], i[j][1][1]),(i[j][0][0], i[j][1][0] ), c=[1,0,0])
        plt.imshow(im)
    plt.xlim((0, im.shape[1]))
    plt.ylim((im.shape[0], 0))
    plt.tight_layout() 
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()
    plt.close()
    return borders_filtered


def order_points(pts):
	x_sort = pts[np.argsort(pts[:, 0]), :]
	left_x = x_sort[:2, :]
	right_x = x_sort[2:, :]
	left_x = left_x[np.argsort(left_x[:, 1]), :]
	(tl, bl) = left_x
	dist = distance.cdist(tl[np.newaxis], right_x, "euclidean")[0]
	(br, tr) = right_x[np.argsort(dist)[::-1], :]
	return np.array([tl,bl,br,tr], dtype="float32")


def homography_scorer(matrix,court_reference,im,gray):
    """
    Calculate score for homography matrix
    """
    court = cv2.warpPerspective(court_reference, matrix, (im.shape[1],im.shape[0]))
    

    court[court > 0] = 1    
    correct = court * gray
    incorrect = court - correct
    correct[correct >0] = 1

    total_correct = np.sum(correct)
    total_incorrect = np.sum(incorrect)
    return total_correct - (0.5* total_incorrect), correct, incorrect


def homography_builder(borders_filtered,court_borders,court_reference,im,gray):
    
    max_score = -np.inf
    k = 0
 
    # Loop over every pair of horizontal lines and every pair of vertical lines
    for horizontal_pair in list(combinations(borders_filtered[0], 2)):
        for vertical_pair in list(combinations(borders_filtered[1], 2)):
            h1, h2 = horizontal_pair
            v1, v2 = vertical_pair
            
            # getting 4 corner points [intersections]
            i1 = get_intersection_1(h1[0],h1[1], v1[0], v1[1])
            i2 = get_intersection_1(h1[0],h1[1], v2[0], v2[1])
            i3 = get_intersection_1(h2[0],h2[1], v1[0], v1[1])
            i4 = get_intersection_1(h2[0],h2[1], v2[0], v2[1])

            intersections = np.array([i1, i2, i3, i4])
            intersections = order_points(intersections)
            intersections[:,[0,1]]= intersections[:,[1,0]]
            
            
            homography_matrix = cv2.getPerspectiveTransform(court_borders,intersections)
            matrix_score,correct, incorrect = homography_scorer(homography_matrix,court_reference,im,gray)

            if max_score < matrix_score:
                k += 1
                fig, axes = plt.subplots(1, 2, figsize=(15, 15), sharex=True, sharey=True)
                ax = axes.ravel()
                max_score = matrix_score
                print("improved score: ", max_score)
                best_int_pts = intersections

                ax[0].plot((h1[0][1], h1[1][1]), (h1[0][0], h1[1][0]), color='r')
                ax[0].plot((h2[0][1], h2[1][1]), (h2[0][0], h2[1][0]),  color='r')
                ax[0].plot((v1[0][1], v1[1][1]), (v1[0][0], v1[1][0]),  color='r')
                ax[0].plot((v2[0][1], v2[1][1]), (v2[0][0], v2[1][0]),  color='r')
                for j, i in enumerate(intersections):
                    ax[0].scatter(i[0], i[1], color='r', s=30)
                ax[0].imshow(im)
                ax[0].set_title('candid court borders #{}'.format(k),fontsize=26)

                row,column = np.where(correct==1)
                row1,column1 = np.where(incorrect==1)
                ax[1].scatter(column,row,color='g' ,s=1)
                ax[1].scatter(column1,row1,color='r' ,s=1)
                ax[1].set_xlim(0,im.shape[1])
                ax[1].set_ylim(0,im.shape[0])
                ax[1].invert_yaxis()
                ax[1].imshow(im)
                ax[1].set_title('applied homography' ,fontsize=26)

                for a in ax:
                    a.set_axis_off()

                plt.tight_layout() 
                plt.show()
                plt.close()
            
    return best_int_pts


def get_corners(H, image_size):
    limit_y = float(image_size[0])
    limit_x = float(image_size[1])
    # Apply H to to find new image bounds
    tr  = np.dot(H, np.array([0.0,      limit_y, 1.0]).flatten()) # new top left
    br  = np.dot(H, np.array([limit_x,  limit_y, 1.0]).flatten()) # new bottom right
    bl  = np.dot(H, np.array([limit_x,      0.0, 1.0]).flatten()) # new bottom left
    matrix_corners = [tr,br,bl]
    
    for pt in matrix_corners:
        if pt[2] == 0:
            return None

    return matrix_corners

def scale_matrix(homography_matrix, image_size): 

    # Get new image corners
    matrix_corners = get_corners(homography_matrix, image_size)

    # don't scale if a point is at infinity
    if matrix_corners is None:
        print("scaling cannot be applied")
        return homography_matrix
        
    k = [max([matrix_corners[j][i] / matrix_corners[j][2] for j in range(len(matrix_corners))])/float(image_size[i]) for i in range(2)]
    k = max(k)
    k = min(k,5)
    print("Scaling factor:{}".format(k))
    scaling_matrix = np.array([[1.0/k,0.0,0.0],[0.0,1.0/k,0.0],[0.0,0.0,1.0]])

    return np.dot(scaling_matrix, homography_matrix)


def add_court_overlay(best_court_corners,court_borders,im):
    homography_matrix = cv2.getPerspectiveTransform(best_court_corners,court_borders)
    homography_matrix_scaled = scale_matrix(homography_matrix, im.shape)
    court = cv2.warpPerspective(im, homography_matrix_scaled,(im.shape[1],im.shape[0]), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return court

def apply_homography(best_court_corners,court_borders,im):

    court = add_court_overlay(best_court_corners,court_borders,im)
    plt.imshow(court)
    plt.tight_layout() 
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.xlim(0,im.shape[1])
    plt.ylim(0,im.shape[0])
    plt.gca().invert_yaxis()
    plt.show()
    plt.close()