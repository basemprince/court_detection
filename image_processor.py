#!/usr/bin/env python3
import skimage.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from functools import cmp_to_key
import matplotlib.pyplot as plt

def import_image(img_loc):
    return skimage.io.imread(img_loc) 

def gray_scale(image,dialation=3,kernel_size=3):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.dilate(gray,kernel,iterations=dialation)
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
        ax[i].set_title('VP Inliers #{}'.format(str(i)),fontsize=26)

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

    print(obj_ind)
    
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
        ax[j].set_title('sorted lines {}'.format(str(j)),fontsize=26)

    for a in ax:
        a.set_axis_off()

    plt.tight_layout() 
    plt.show()
    plt.close()
    return borders
