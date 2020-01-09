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
from config.spm_config import spm_config as params
from utils.utils import clip, gaussian_radius, prepare_bbox, read_json, prepare_kps
from utils.data_aug import data_aug
from encoder.spm import SingleStageLabel
from pycocotools.coco import COCO

def get_dataset(num_gpus = 1, mode = 'train'):
    assert mode in ['train', 'test']

    if mode == 'train':
        json_file = params['train_json_file']
        img_path  = params['train_img_path']
    else:
        json_file = params['test_json_file']
        img_path  = params['test_img_path']

    coco = COCO(json_file)
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)

    def parser_func(img_id):
        if type(img_id) != int:
            img_id = int(img_id.numpy())
            assert type(img_id) == int

        img_info = coco.loadImgs(img_id)[0]
        ann_ids  = coco.getAnnIds(img_id, cat_ids)
        annos    = coco.loadAnns(ann_ids)

        spm = SingleStageLabel(img_info, img_path, annos)
        img, center_map, center_mask, kps_offset, kps_weight = spm(params['height'], params['width'], params['scale'], params['num_joints'])

        return img, center_map, center_mask, kps_offset, kps_weight

    def tf_parser_func(img_id):
        [img, center_map, center_mask, kps_offset, kps_weight] = tf.py_function(
            func=parser_func, inp=[img_id], Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        )
        
        img.set_shape([params['height'], params['width'], 3])
        center_map.set_shape([params['height']//params['scale'], params['width']//params['scale'], 1])
        center_mask.set_shape([params['height']//params['scale'], params['width']//params['scale'], 1])
        kps_offset.set_shape([params['height']//params['scale'], params['width']//params['scale'], params['num_joints']*2])
        kps_weight.set_shape([params['height']//params['scale'], params['width']//params['scale'], params['num_joints']*2])

        return img, center_map, center_mask, kps_offset, kps_weight


    if mode == 'train':
        random.shuffle(img_ids)
    dataset = tf.data.Dataset.from_tensor_slices(img_ids)
    dataset = dataset.map(tf_parser_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(params['batch_size'] * num_gpus, drop_remainder=True).repeat(1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset