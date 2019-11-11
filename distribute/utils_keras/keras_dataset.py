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

def get_dataset(num_gpus = 1, mode = 'train'):
    assert mode in ['train', 'val']

    if mode == 'train':
        json_file = params['train_json_file']
        img_path  = params['train_img_path']
    else:
        json_file = params['val_json_file']
        img_path  = params['val_img_path']

    img_ids, id_bboxs_dict, id_kps_dict = read_json(json_file)

    def paser_func(img_id):
        if not type(img_id) == type('123'):
            img_id = img_id.numpy()
            if type(img_id) == type(b'123'):
                img_id = str(img_id, encoding='utf-8')

        bboxs = id_bboxs_dict[img_id]
        kps = id_kps_dict[img_id]
        img = cv2.imread(os.path.join(img_path, img_id + '.jpg'))

        # data aug
        img, bboxs, kps = data_aug(img, bboxs, kps)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # create center label
        orih, oriw, oric = img.shape
        neth, netw = params['height'], params['width']
        outh, outw = neth // params['scale'], netw // params['scale']

        centers, sigmas, whs = prepare_bbox(bboxs, orih, oriw, outh, outw)
        keypoints, kps_sigmas = prepare_kps(kps, orih, oriw, outh, outw)

        spm_label = SingleStageLabel(outh, outw, centers, sigmas, keypoints)
        center_map, kps_map, kps_map_weight = spm_label()

        # create img input
        img = cv2.resize(img, (netw, neth), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.

        return img, center_map, kps_map, kps_map_weight

    def tf_parse_func(img_id):
        [img, center_map, kps_map, kps_map_weight] = tf.py_function(paser_func, [img_id],
                                                                    [tf.float32, tf.float32, tf.float32, tf.float32])

        img.set_shape([params['height'], params['width'], 3])
        center_map.set_shape([params['height'] // params['scale'], params['width'] // params['scale'], 1])
        kps_map.set_shape([params['height'] // params['scale'], params['width'] // params['scale'], 14 * 2])
        kps_map_weight.set_shape([params['height'] // params['scale'], params['width'] // params['scale'], 14 * 2])

        kps_map_with_weight = tf.concat([kps_map, kps_map_weight], axis=-1)

        return img, center_map, kps_map_with_weight

    dataset = tf.data.Dataset.from_tensor_slices(img_ids).repeat(-1)
    dataset = dataset.map(tf_parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(params['batch_size'] * num_gpus, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset




