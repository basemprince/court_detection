#!/usr/bin/env python3
from lcnn.config import C, M
import pprint
import random
import numpy as np
import torch
import os
import scipy.io as sio
from lcnn.models.HT import hough_transform
import lcnn
from lcnn.models.line_vectorizer import LineVectorizer
from lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
import skimage.transform
from lcnn.postprocess import postprocess
import matplotlib.pyplot as plt

class hough_lines_prio:
    def __init__(self):
        config_file = "config/wireframe.yaml"
        C.update(C.from_yaml(filename=config_file))
        M.update(C.model)
        self.C = C
        self.M = M
        # pprint.pprint(C, indent=4)

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device_name = "cpu"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        # print("Let's use", torch.cuda.device_count(), "GPU(s)!")

        self.device = torch.device(device_name)
        self.checkpoint = torch.load("pretrained_models/ht_lcnn/checkpoint.pth.tar", map_location=self.device)
        self.load_model()

    def load_model(self):
        # Load model
        if os.path.isfile(self.C.io.vote_index):
            vote_index = sio.loadmat(self.C.io.vote_index)['vote_index']
        else:
            vote_index = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
            sio.savemat(self.C.io.vote_index, {'vote_index': vote_index})
        self.vote_index = torch.from_numpy(vote_index).float().contiguous().to(self.device)
        print('load vote_index', vote_index.shape)

        model = lcnn.models.hg(
            depth=self.M.depth,
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=self.M.num_stacks,
            num_blocks=self.M.num_blocks,
            num_classes=sum(sum(self.M.head_size, [])),
            vote_index=self.vote_index,

        )
        model = MultitaskLearner(model)
        self.model = LineVectorizer(model)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def image_resize(self,im):
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, 2)
        im = im[:, :, :3]
        im_resized = skimage.transform.resize(im, (512, 512)) * 255
        image = (im_resized - M.image.mean) / M.image.stddev
        # cv2.imshow("img",image)
        # cv2.waitKey(0) 
        image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()
        return image
    
    def extract_lines(self,image,im):
        with torch.no_grad():
            input_dict = {
                "image": image.to(self.device),
                "meta": [
                    {
                        "junc": torch.zeros(1, 2).to(self.device),
                        "jtyp": torch.zeros(1, dtype=torch.uint8).to(self.device),
                        "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(self.device),
                        "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(self.device),
                    }
                ],
                "target": {
                    "jmap": torch.zeros([1, 1, 128, 128]).to(self.device),
                    "joff": torch.zeros([1, 1, 2, 128, 128]).to(self.device),
                },
                "mode": "testing",
            }
            H = self.model(input_dict)["preds"]
        lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
        scores = H["score"][0].cpu().numpy()
        print('lines: ',lines, ', scores: ', scores)
        for i in range(1, len(lines)):
            if (lines[i] == lines[0]).all():
                lines = lines[:i]
                scores = scores[:i]
                break

        # postprocess lines to remove overlapped lines
        diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
        self.nlines, self.nscores = postprocess(lines, scores, diag * 0.01, 0, False)
        return self.nlines, self.nscores

    def score_filtering(self,im,t,perc_tp_remove):
        remove = im.shape[0] * perc_tp_remove
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        nlines_scored = []
        for (a, b), s in zip(self.nlines, self.nscores):
            if s < t or a[0] < remove or b[0] < remove:
                continue
            nlines_scored.append((a,b))
            plt.plot([a[1], b[1]], [a[0], b[0]], linewidth=2)
        nlines_scored = np.array(nlines_scored)
        plt.imshow(im)
        plt.show()
        plt.close()
        return nlines_scored

if __name__ == "__main__":
    hlp = hough_lines_prio()