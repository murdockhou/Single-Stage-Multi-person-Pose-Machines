#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: spm.py
@time: 2019/9/9 下午2:54
@desc:
'''
import numpy as np
import math
import os
import cv2
from utils.utils import draw_gaussian, clip, draw_ttfnet_gaussian

class SingleStageLabel():
    def __init__(self, img_info, img_path, annos, sigma = 7.):
        '''
        img_info: read by coco.loadImgs(img_id)
        img_path: path where image was
        annos: read by coco.loadAnns(ann_ids)
        '''
        self.img = cv2.imread(os.path.join(img_path, img_info['file_name']))
        self.annos = annos
        self.orih = img_info['height']
        self.oriw = img_info['width']
        self.sigma = sigma
        self.tau = 7.
        # hierarchical SPR, 0->1->2, 3->4->5, 6->7->8, 9->10->11, only use the first 12 points of ai-format
        self.level = [[0, 1, 2],
                      [3, 4, 5],
                      [6, 7, 8],
                      [9, 10, 11],
                      ]

    def __call__(self, inh, inw, scale, njoints):
        '''
        inh & inw: input height & width resolution of network
        scale: network downsample scale
        njoints: number of joints need to learn
        '''
        self.scale = scale
        self.njoints = njoints
        self.outh = inh//scale
        self.outw = inw//scale
        self.Z = math.sqrt(self.outw**2 + self.outh**2)
        # self.Z = 1

        scale_x = self.oriw / self.outw
        scale_y = self.orih / self.outh

        # create input image for network
        img = self.img.astype(np.float32) / 255.0
        img = cv2.resize(img, (inw, inh), interpolation=cv2.INTER_CUBIC)

        # create label
        self.center_map = np.zeros(shape=(self.outh, self.outw, 1), dtype=np.float32)
        self.kps_offset = np.zeros(shape=(self.outh, self.outw, njoints*2), dtype=np.float32)
        self.kps_count  = np.zeros(shape=(self.outh, self.outw, njoints*2), dtype=np.uint)
        self.kps_weight = np.zeros(shape=(self.outh, self.outw, njoints*2), dtype=np.float32)
        for ann in self.annos:
            # [x1, y1, w, h] -> [x1, y1, x2, y2]
            bbox = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]]
            bbox = [bbox[0]/scale_x, bbox[1]/scale_y, bbox[2]/scale_x, bbox[3]/scale_y]
            kps  = ann['keypoints']
            assert len(kps) == njoints * 3
            for i in range(njoints):
                kps[3*i+0] /= scale_x
                kps[3*i+1] /= scale_y
            self.create_spm_label(bbox, kps)

        self.kps_count[self.kps_count == 0] += 1
        self.kps_offset = np.divide(self.kps_offset, self.kps_count)
        self.center_mask = np.where(self.center_map>0, 1, 0)
        
        return img, self.center_map, self.center_mask, self.kps_offset, self.kps_weight

    def create_spm_label(self, bbox, kps):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w < 1 or h < 1:
            return

        if w > h:
            center_sigmay = self.sigma
            center_sigmax = min(self.sigma*1.5, center_sigmay * w / h)
        else:
            center_sigmax = self.sigma
            center_sigmay = min(self.sigma*1.5, center_sigmax * h / w)

        centers = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        self.center_map[..., 0] = draw_ttfnet_gaussian(self.center_map[...,0], centers, center_sigmax, center_sigmay)
        # self.center_map[..., 0] = self.create_center_label(self.center_map[...,0], centers, self.tau)

        self.body_joint_displacement_v2(centers, kps, self.tau)       

    def body_joint_displacement_v2(self, center, kps, tau):
        '''
        if param tau is bigger, then two closed person on one image will cause confused for offset label.
        '''
        for single_path in self.level:
            start_joint = [center[0], center[1]]
            for i, index in enumerate(single_path):
                end_joint = kps[3*index:3*index+3]
                if end_joint[0] == 0 or end_joint[1] == 0:
                    continue
                # make new end_joint based offset
                offset_x, offset_y = end_joint[0] - start_joint[0], end_joint[1] - start_joint[1]
                next_x = center[0] + offset_x
                next_y = center[1] + offset_y

                self.create_dense_displacement_map(index, center, [next_x, next_y], tau)
                start_joint[0], start_joint[1] = end_joint[0], end_joint[1]

    def create_dense_displacement_map(self, index, start_joint, end_joint, tau):
        
        x0 = int(max(0, start_joint[0] - tau))
        y0 = int(max(0, start_joint[1] - tau))
        x1 = int(min(self.outw, start_joint[0] + tau))
        y1 = int(min(self.outh, start_joint[1] + tau))

        for x in range(x0, x1):
            for y in range(y0, y1):
                x_offset = (end_joint[0] - x) / self.Z
                y_offset = (end_joint[1] - y) / self.Z
                # print (x_offset, y_offset)

                self.kps_offset[y, x, 2*index] += x_offset
                self.kps_offset[y, x, 2*index+1] += y_offset
                # self.kps_weight[y, x, 2*index:2*index+2] = 1.
                self.kps_weight[y, x] = 1
                if end_joint[0] != x or end_joint[1] != y:
                    self.kps_count[y, x, 2*index:2*index+2] += 1


    def create_center_label(self, heatmap, centers, tau):
        center_x = centers[0]
        center_y = centers[1]
        x0 = 0
        y0 = 0
        x1 = self.outw
        y1 = self.outh
        for x in range(x0, x1):
            for y in range(y0, y1):
                dis = math.sqrt((x-center_x)**2 + (y-center_y)**2)
                heatmap[y, x] = max(math.exp(-dis/tau/tau), heatmap[y,x])
        return heatmap

    # def create_dense_displacement_map(self, index, start_joint, end_joint, sigmax=2, sigmay=2):

    #     # print('start joint {} -> end joint {}'.format(start_joint, end_joint))
    #     center_x, center_y = int(start_joint[0]), int(start_joint[1])
    #     th = 4.6052
    #     delta = np.sqrt(th * 2)

    #     x0 = int(max(0, center_x - delta * sigmax + 0.5))
    #     y0 = int(max(0, center_y - delta * sigmay + 0.5))

    #     x1 = int(min(self.outw, center_x + delta * sigmax + 0.5))
    #     y1 = int(min(self.outh, center_y + delta * sigmay + 0.5))

    #     # x0 = int(clip(start_joint[0], 0, self.width))
    #     # y0 = int(clip(start_joint[1], 0, self.height))
    #     # x1 = int(clip(x0+taux, 0, self.width))
    #     # y1 = int(clip(y0+tauy, 0, self.height))
    #     # print (x0,x1, y0,y1)
    #     for x in range(x0, x1):
    #         for y in range(y0, y1):
    #             x_offset = (end_joint[0] - x) / self.Z
    #             y_offset = (end_joint[1] - y) / self.Z
    #             # print (x_offset, y_offset)

    #             self.kps_offset[y, x, 2*index] += x_offset
    #             self.kps_offset[y, x, 2*index+1] += y_offset
    #             self.kps_weight[y, x, 2*index:2*index+2] = 1.
    #             if end_joint[0] != x or end_joint[1] != y:
    #                 self.kps_count[y, x, 2*index:2*index+2] += 1

