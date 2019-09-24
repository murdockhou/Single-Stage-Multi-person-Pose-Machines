#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: dataset.py
@time: 2019/7/24 下午2:25
@desc:
'''

import tensorflow as tf
import numpy as np
import random
import json
import cv2
import os
import math
from config.center_config import center_config as center_params
from utils.utils import clip, gaussian_radius, prepare_bbox, read_json, prepare_kps
from utils.data_aug import data_aug
from encoder.spm import SingleStageLabel

id_bboxs_dict = None
id_kps_dict = None
img_path = None
params = center_params

def get_dataset(mode = 'train'):
    assert mode in ['train', 'val']
    global id_bboxs_dict, img_path, params, id_kps_dict

    if mode == 'train':
        json_file = params['train_json_file']
        img_path  = params['train_img_path']
    else:
        json_file = params['val_json_file']
        img_path  = params['val_img_path']

    img_ids, id_bboxs_dict, id_kps_dict = read_json(json_file)
    if mode == 'train':
        random.shuffle(img_ids)
        dataset = tf.data.Dataset.from_tensor_slices(img_ids)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(img_ids)

    dataset = dataset.shuffle(buffer_size=1000).repeat(1)

    if mode == 'train':
        dataset = dataset.map(tf_parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(tf_parse_func_for_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def tf_parse_func_for_val(img_id):
    [img_id, height, width, img] = tf.py_function(paser_func_for_val, [img_id], [tf.string, tf.float32, tf.float32, tf.float32])
    return img_id, height, width, img

def paser_func_for_val(img_id):

    global id_bboxs_dict, params, img_path, id_kps_dict

    if not type(img_id) == type('123'):
        img_id = img_id.numpy()
        if type(img_id) == type(b'123'):
            img_id = str(img_id, encoding='utf-8')

    img = cv2.imread(os.path.join(img_path, img_id + '.jpg'))

    # padding img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    # 只在最右边或者最下边填充0, 这样不会影响box或者点的坐标值, 所以无需再对box或点的坐标做改变
    if w > h:
        img = cv2.copyMakeBorder(img, 0, w - h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        img = cv2.copyMakeBorder(img, 0, 0, 0, h - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # create img input
    orih, oriw, oric = img.shape
    neth, netw = params['height'], params['width']
    img = cv2.resize(img, (netw, neth), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.  # conver to 0~1 tools if focal loss is right

    return img_id, orih, oriw, img

def tf_parse_func(img_id):
    [img, label] = tf.py_function(paser_func, [img_id], [tf.float32, tf.float32])
    return img, label

def paser_func(img_id):
    global id_bboxs_dict, params, img_path, id_kps_dict

    if not type(img_id) == type('123'):
        img_id = img_id.numpy()
        if type(img_id) == type(b'123'):
            img_id = str(img_id, encoding='utf-8')

    bboxs = id_bboxs_dict[img_id]
    kps = id_kps_dict[img_id]
    img = cv2.imread(os.path.join(img_path, img_id+'.jpg'))
    #  data aug
    # img, bboxs, kps = data_aug(img, bboxs, kps)

    # padding img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    # 只在最右边或者最下边填充0, 这样不会影响box或者点的坐标值, 所以无需再对box或点的坐标做改变
    if w > h:
        img = cv2.copyMakeBorder(img, 0, w-h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        img = cv2.copyMakeBorder(img, 0, 0, 0, h-w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # create center label
    orih, oriw, oric = img.shape
    neth, netw = params['height'], params['width']
    outh, outw = neth//params['scale'], netw//params['scale']

    centers, sigmas, whs = prepare_bbox(bboxs, orih, oriw, outh, outw)
    # center_encoder = centerEncoder(outh, outw, centers, sigmas, whs)
    # center_label = center_encoder()

    # create keypoints label
    keypoints, kps_sigmas = prepare_kps(kps, orih, oriw, outh, outw)
    # kps_encoder = kpsEncoder(outh, outw, centers, keypoints, sigmas=kps_sigmas)
    # kps_label = kps_encoder()

    spm_label = SingleStageLabel(outh, outw, centers, sigmas, keypoints)
    label = spm_label()

    # create img input
    img = cv2.resize(img, (netw, neth), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255. # conver to 0~1 tools if focal loss is right

    # read by tensorflow, 不是很方便做iamge和box同时做数据增强
    # image = tf.io.read_file(os.path.join(img_path, img_id+'.jpg'))
    # image = tf.image.decode_jpeg(image)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.image.resize_with_pad(image, netw, neth)

    # label = np.concatenate([center_label, kps_label], axis=-1)
    return img, label
