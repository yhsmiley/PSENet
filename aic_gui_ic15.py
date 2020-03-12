import os
import cv2
import sys
import time
import collections
import torch
import argparse
import numpy as np

from torch.autograd import Variable
from torch.utils import data

import PSENet.models as models

from PSENet.pse import pse

from argparse import Namespace
from PIL import Image
import torchvision.transforms as transforms

from PSENet.aic_ocr.text_engine import TextEngine

class AIC_Demo(object):
    def __init__(self):
        self.img = None

        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = str(torch.cuda.current_device())
            self.gpu = True
        else:
            self.gpu = False


    def get_img(self, img):
        img = img[:, :, [2, 1, 0]]
        return img

    def scale(self, img, long_size):
        h, w = img.shape[0:2]
        scale = long_size * 1.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        return img

    def debug(self, imgs):
        col = []
        for i in range(len(imgs)):
            row = []
            for j in range(len(imgs[i])):
                row.append(imgs[i][j])
            res = np.concatenate(row, axis=1)
            col.append(res)
        res = np.concatenate(col, axis=0)
        return res

    def test(self, args):
        # Setup Model
        if args.arch == "resnet50":
            model = models.resnet50(pretrained=True, num_classes=7, scale=args.scale)
        elif args.arch == "resnet101":
            model = models.resnet101(pretrained=True, num_classes=7, scale=args.scale)
        elif args.arch == "resnet152":
            model = models.resnet152(pretrained=True, num_classes=7, scale=args.scale)
        
        for param in model.parameters():
            param.requires_grad = False

        model = model.cuda()
        
        if args.resume is not None:                                         
            if os.path.isfile(args.resume):
                print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                
                # model.load_state_dict(checkpoint['state_dict'])
                d = collections.OrderedDict()
                for key, value in checkpoint['state_dict'].items():
                    tmp = key[7:]
                    d[tmp] = value
                model.load_state_dict(d)

                print("Loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                sys.stdout.flush()
            else:
                print("No checkpoint found at '{}'".format(args.resume))
                sys.stdout.flush()

        model.eval()

        img = self.get_img(self.img)
        org_img = img[:, :, [2, 1, 0]]
        org_img = np.expand_dims(org_img, axis=0)
        org_img = torch.from_numpy(org_img)
        
        scaled_img = self.scale(img, args.long_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
        img = scaled_img.unsqueeze(0)

        img = Variable(img.cuda())
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()

        torch.cuda.synchronize()

        with torch.no_grad():
            outputs = model(img)

        score = torch.sigmoid(outputs[:, 0, :, :])
        outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

        text = outputs[:, 0, :, :]
        kernels = outputs[:, 0:args.kernel_num, :, :] * text

        score = score.data.cpu().numpy()[0].astype(np.float32)
        text = text.data.cpu().numpy()[0].astype(np.uint8)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        
        # c++ version pse
        pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))
        # python version pse
        # pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))
        
        scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
        label = pred
        label_num = np.max(label) + 1
        bboxes = []
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < args.min_area / (args.scale * args.scale):
                continue

            score_i = np.mean(score[label == i])
            if score_i < args.min_score:
                continue

            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect) * scale
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))

        torch.cuda.synchronize()

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)
        
        text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
        results = self.debug([[text_box]])
        return results

    def run_ocr(self, img):
        self.img = img

        # text detection
        args = Namespace(arch='resnet50', binary_th=1.0, kernel_num=7, long_size=1200, min_area=800.0, min_kernel_area=5.0, min_score=0.93, resume='PSENet/pretrained/ic15_res50_pretrain_ic17.pth.tar', scale=1)
        output = self.test(args)

        # text recognizer
        text_engine = TextEngine(cuda=True)
        recognized_text = text_engine.recognize_text_aic(cropped_images)

        print('recognized: {}'.format(list(recognized_text)))
        exit()

        return output