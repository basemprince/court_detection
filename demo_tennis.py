#!/usr/bin/env python3
"""Process an image with the trained neural network
Usage:
    demo_tennis.py [options] <yaml-config> <checkpoint> <images>...
    demo_tennis.py (-h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint
   <images>                      Path to images

Options:
   -h --help                     Show this screen.
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
"""

import os
import os.path as osp
import pprint
import random
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import torch
import yaml
from docopt import docopt
import scipy.io as sio

import lcnn
from lcnn.config import C, M
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from lcnn.models.HT import hough_transform

from lcnn.postprocess import postprocess
from lcnn.utils import recursive_to

PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

refine_corners = False

def c(x):
    return sm.to_rgba(x)


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/wireframe.yaml"
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)
    checkpoint = torch.load(args["<checkpoint>"], map_location=device)

    # Load model
    if os.path.isfile(C.io.vote_index):
        vote_index = sio.loadmat(C.io.vote_index)['vote_index']
    else:
        vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
        sio.savemat(C.io.vote_index, {'vote_index': vote_index})
    vote_index = torch.from_numpy(vote_index).float().contiguous().to(device)
    print('load vote_index', vote_index.shape)

    model = lcnn.models.hg(
        depth=M.depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
        vote_index=vote_index,

    )
    model = MultitaskLearner(model)
    model = LineVectorizer(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    for imname in args["<images>"]:
        print(f"Processing {imname}")
        im = skimage.io.imread(imname)
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, 2)
        im = im[:, :, :3]
        im_resized = skimage.transform.resize(im, (512, 512)) * 255
        image = (im_resized - M.image.mean) / M.image.stddev
        image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()
        with torch.no_grad():
            input_dict = {
                "image": image.to(device),
                "meta": [
                    {
                        "junc": torch.zeros(1, 2).to(device),
                        "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                        "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                        "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                    }
                ],
                "target": {
                    "jmap": torch.zeros([1, 1, 128, 128]).to(device),
                    "joff": torch.zeros([1, 1, 2, 128, 128]).to(device),
                },
                "mode": "testing",
            }
            H = model(input_dict)["preds"]

        lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
        scores = H["score"][0].cpu().numpy()
        for i in range(1, len(lines)):
            if (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        # postprocess lines to remove overlapped lines
        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
        nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

        for i, t in enumerate([0.97]):#, 0.95, 0.96, 0.97, 0.98, 0.99]):
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            
            new_nlines = np.zeros((2,2,2))
            new_nscores = np.zeros((2))
            bottom_1 = -1
            bottom_2 = -1
            for (a, b), s in zip(nlines, nscores):
                if s < t:
                    continue
                
                # is horizontal
                # has a point in the middle of the image
                if(abs(a[0] - b[0]) < abs(a[1] - b[1]) and \
                        ((a[1] - im.shape[1] / 2) * (b[1] - im.shape[1] / 2)) < 0):
                    min_bottom = min(bottom_1, bottom_2)
                    if a[0] > min_bottom and b[0] > min_bottom:
                        if(bottom_1 == min_bottom):
                            bottom_1 = (a[0]+b[0])/2
                            new_nlines[0][0] = a
                            new_nlines[0][1] = b
                            new_nscores[0] = s
                        else:
                            bottom_2 = (a[0]+b[0])/2
                            new_nlines[1][0] = a
                            new_nlines[1][1] = b
                            new_nscores[1] = s
            
            court_points = new_nlines.copy()
            if(court_points[0][0][0] > court_points[1][0][0]):
                court_points = np.flip(court_points, 0)
            for i, p in enumerate(court_points):
                court_points[i] = np.flip(court_points[i], 1)
            for i, p in enumerate(court_points):
                court_points[i] = p[p[:, 0].argsort()]
            # print(court_points)
            court_points = court_points.reshape(-1, new_nlines.shape[-1])
            
            if(refine_corners):
                offset_multipliers = [-1, 1, 0, 0]
                court_points_new = []
                for i, point in enumerate(court_points):
                    x_orig, y_orig = point
                    x = int(x_orig)
                    y = int(y_orig)
                    t=20
                    roi = (cv2.cvtColor(im[y-t:y+t, x-t:x+t],cv2.COLOR_BGR2GRAY))
                    kernel = np.array([[0, -1, 0], 
                    [-1, 5,-1], 
                    [0, -1, 0]])
                    roi = cv2.filter2D(roi, -1, kernel)
                    max_corners = 2
                    qualityLevel = 0.01
                    minDistance = 5
                    blockSize = 3
                    gradientSize = 3
                    useHarrisDetector = False
                    k = 0.04
                    corners = cv2.goodFeaturesToTrack(roi, max_corners, qualityLevel, minDistance, None, \
                        blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)
                    dist = np.linalg.norm(corners[0,:,:]-corners[1,:,:])
                    # print(dist)
                    offset = int(offset_multipliers[i] * dist)
                    corner = np.mean(corners, axis=0)
                    cv2.circle(roi, (int(corner[0,0]) + offset, int(corner[0,1])), 2, (0,0,255), cv2.FILLED)
                    cv2.imshow("crop_"+str(i), roi)
                    cv2.waitKey(0)
                    print((x_orig,y_orig))
                    print(corner[0])
                    court_points_new.append([x_orig-t+corner[0][0]+offset,y_orig-t+corner[0][1]])
            else:
                court_points_new = court_points

            print("----------------------------")
            print(court_points)
            print(court_points_new)
            print("----------------------------")
            for c in court_points_new:
                cv2.circle(im, (int(c[0]), int(c[1])), 3, (255,0,0), cv2.FILLED)
            cv2.imshow("im", im)
            cv2.waitKey(0)  
                      
            template_points = np.asarray([[147, 1839], [970, 1839], [10, 2388], [1107, 2388]])
            court_reference = cv2.imread("pictures/court_reference.png", 0)
            for c in template_points:
                cv2.circle(court_reference, (int(c[0]), int(c[1])), 3, (255,0,0), cv2.FILLED)
            cv2.imshow("Court", court_reference)
            cv2.waitKey(0)
            # T, _, _ = best_fit_transform(template_points, court_points)
            T, status = cv2.findHomography(template_points, np.asarray(court_points_new))
            
            court = add_court_overlay(im, T, overlay_color=(255, 0, 0))
            for c in court_points_new:
                cv2.circle(court, (int(c[0]), int(c[1])), 3, (255,0,0), cv2.FILLED)
            court = cv2.cvtColor(court, cv2.COLOR_RGB2BGR)
            cv2.imshow("Court", court)
            cv2.waitKey(0)
            cv2.imwrite("out.png", court)
                
            # for (a, b), s in zip(new_nlines, new_nscores):
            #     if s < t:
            #         continue
            #     plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=2, zorder=s)
            #     plt.scatter(a[1], a[0], **PLTOPTS)
            #     plt.scatter(b[1], b[0], **PLTOPTS)
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.imshow(im)
            # # plt.savefig(imname.replace(".png", f"-{t:.02f}.svg"), bbox_inches="tight")
            # plt.show()
            # plt.close()

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def add_court_overlay(frame, homography, overlay_color=(255, 255, 255)):
    court_reference = cv2.imread("pictures/court_reference.png", 0)
    court = cv2.warpPerspective(court_reference, homography, frame.shape[1::-1])
    frame[court == 255, :] = overlay_color
    return frame

if __name__ == "__main__":
    main()
