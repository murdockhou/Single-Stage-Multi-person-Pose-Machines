#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: dataset_test.py
@time: 2019/7/23 下午3:34
@desc:
'''

import cv2
import numpy as np
import os

use_dataset = False

if use_dataset:
    import tensorflow as tf
    from decoder.decode_spm import SpmDecoder
    from dataset.dataset import get_dataset
    from config.spm_config import spm_config as params
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    mode = 'train'
    dataset = get_dataset(mode=mode)

    colors = [[0,0,255],[255,0,0],[0,255,0]]
    for epco in range(1):
        for step, (img, center_map, center_mask, kps_map, kps_map_weight) in enumerate(dataset):
            # print (step)
            # print (img[0].shape)
            # img1 = img[0]
            # label1 = label[0]
            # break
            print ('epoch {} / step {}'.format(epco, step))
            img = (img.numpy()[0] * 255).astype(np.uint8)

            spm_decoder = SpmDecoder(4, 4, params['height']//4, params['width']//4)
            results = spm_decoder([center_map[0].numpy(), kps_map[0].numpy()])

            for j, result in enumerate(results):
                center = result['center']
                single_person_joints = result['joints']
                cv2.circle(img, (int(center[0]), int(center[1])), 5, colors[j%3], thickness=-1)
                for i in range(params['num_joints']):
                    x = int(single_person_joints[2*i])
                    y = int(single_person_joints[2*i+1])
                    cv2.circle(img, (x,y), 4, colors[j%3],thickness=-1)
            cv2.imshow('label', img)
            k = cv2.waitKey(0)
            if k == 113:
                break

# tools on label without dataset
else:
    from utils.utils import *
    from config.spm_config import spm_config as params
    from encoder.spm import SingleStageLabel
    from decoder.decode_spm import SpmDecoder
    from utils.data_aug import data_aug
    from pycocotools.coco import COCO

    json_file = '/media/hsw/E/datasets/multipose_with_only_12_body_joints/aitrain.json'
    img_path = '/media/hsw/E/datasets/ai-challenger/ai_train/train'
    coco = COCO(json_file)
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)

    colors = [[0,0,255],[255,0,0],[0,255,0]]
    
    for img_id in img_ids:
        print ('--------------------------------------------------------------')
        img_info = coco.loadImgs(img_id)[0]
        ann_ids  = coco.getAnnIds(img_id, cat_ids)
        annos    = coco.loadAnns(ann_ids)
        
        ################# show ori label ##########################
        img_ori = cv2.imread(os.path.join(img_path, img_info['file_name']))
        for j, ann in enumerate(annos):
            # [x1, y1, w, h] -> [x1, y1, x2, y2]
            bbox = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]]
            kps  = ann['keypoints']
            # print (kps)
            assert len(kps) == 12 * 3
            for i in range(12):
                x = int(kps[i*3+0])
                y = int(kps[i*3+1])
                v = kps[i*3+2]
                cv2.circle(img_ori, (x,y),4,colors[j%3],thickness=-1)
                cv2.putText(img_ori, str(i), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 250), 1)
        cv2.imshow('ori', img_ori)
        ###########################################################


        spm = SingleStageLabel(img_info, img_path, annos)
        img, center_map, center_mask, kps_offset, kps_weight = spm(params['height'], params['width'], params['scale'], params['num_joints'])

        cv2.imshow('center', center_map)
        #  data aug
        # img, bboxs, kps = data_aug(img, bboxs, kps)

        factor_x = 4
        facotr_y = 4
        spm_decoer = SpmDecoder(factor_x, facotr_y, 128, 128)
        results = spm_decoer([center_map, kps_offset])

        for j, result in enumerate(results):
            center = result['center']
            single_person_joints = result['joints']
            cv2.circle(img, (int(center[0]), int(center[1])), 5, colors[j%3], thickness=-1)
            for i in range(params['num_joints']):
                x = int(single_person_joints[2*i])
                y = int(single_person_joints[2*i+1])
                cv2.circle(img, (x,y), 4, colors[j%3],thickness=-1)
                # cv2.putText(img, str(i), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 250), 1)
        cv2.imshow('label', img)
        k = cv2.waitKey(0)
        if k == 113:
            break


