#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: spm_model_test.py
@time: 2019/9/11 下午8:16
@desc:
'''
import tensorflow as tf
import os
import cv2
import argparse

import numpy as np
from nets.spm_model import SpmModel
from decoder.decode_spm import SpmDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--video', default=None, type=str)
parser.add_argument('--imgs', default=None, type=str)
parser.add_argument('--score', default=0.6, type=float)
parser.add_argument('--dist', default=20., type=float)
parser.add_argument('--netH', default=512, type=int)
parser.add_argument('--netW', default=512, type=int)
parser.add_argument('--ckpt', default='/home/hsw/server/ckpt-74/ckpt-79', type=str)
args = parser.parse_args()

colors = [[0,0,255],[255,0,0],[0,255,0],[0,255,255],[255,0,255],[255,255,0]]
netH = args.netH
netW = args.netW
score = args.score
dist = args.dist

@tf.function
def infer(model, inputs):

    center_map, kps_reg_map = model(inputs)

    return center_map, kps_reg_map

def run(model, img):
    img_show = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    # 只在最右边或者最下边填充0, 这样不会影响box或者点的坐标值, 所以无需再对box或点的坐标做改变
    #if w > h:
    #    img = cv2.copyMakeBorder(img, 0, w - h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    #else:
    #    img = cv2.copyMakeBorder(img, 0, 0, 0, h - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    img_ori = img.copy()
    img = cv2.resize(img, (netH, netW), interpolation=cv2.INTER_CUBIC)
    img_input = np.expand_dims(img.astype(np.float32) / 255., axis=0)
    factor_x = img_ori.shape[1] / (netW / 4)
    factor_y = img_ori.shape[0] / (netH / 4)

    center_map, kps_reg_map = infer(model, img_input)
    # label = outputs[0]
    spm_decoder = SpmDecoder(factor_x, factor_y, netH // 4, netW // 4)
    joints, centers = spm_decoder([center_map[0], kps_reg_map[0]], score_thres=score, dis_thres=dist)
    print(centers)

    for j, single_person_joints in enumerate(joints):
        cv2.circle(img_show, (int(centers[j][0]), int(centers[j][1])), 8, colors[j % 6], thickness=-1)
        for i in range(14):
            x = int(single_person_joints[2 * i])
            y = int(single_person_joints[2 * i + 1])
            cv2.circle(img_show, (x, y), 4, colors[j % 6], thickness=-1)
            # cv2.putText(img_show, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 1)

    cv2.imshow('result', img_show)
    k = cv2.waitKey(0)


if __name__ == '__main__':

    use_gpu = False
    use_nms = True

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    inputs = tf.keras.Input(shape=(netH, netW, 3), name='modelInput')
    outputs = SpmModel(inputs, 14, is_training=False)
    model = tf.keras.Model(inputs, outputs)

    ckpt = tf.train.Checkpoint(net=model)
    ckpt.restore(args.ckpt)

    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
        ret, img = cap.read()
        while (ret):
            run(model, img)
            ret, img = cap.read()
    elif os.path.isdir(args.imgs):
        for img_name in os.listdir(args.imgs):
            print('----------------------------------------------')
            img = cv2.imread(os.path.join(args.imgs, img_name))
            run(model, img)
    elif os.path.isfile(args.imgs):
        img = cv2.imread(args.imgs)
        run(model, img)

    else:
        print ('You Must Provide one video or imgs/img_path')



