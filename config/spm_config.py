#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: spm_config.py
@time: 2019/7/24 下午7:17
@desc:
'''

spm_config = {}

spm_config['height'] = 512
spm_config['width'] = 512
spm_config['scale'] = 4
spm_config['batch_size'] = 4
spm_config['joints'] = 14
spm_config['kps_sigma'] = 2.5

# spm_config['train_json_file'] = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/train10.json'
spm_config['train_json_file'] = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json'
spm_config['train_img_path'] = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902'

spm_config['val_json_file'] = '/media/hsw/E/datasets/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
spm_config['val_img_path'] = '/media/hsw/E/datasets/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'

spm_config['finetune'] = None
