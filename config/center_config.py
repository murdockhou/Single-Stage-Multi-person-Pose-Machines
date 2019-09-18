#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: center_config.py
@time: 2019/7/24 下午7:17
@desc:
'''

center_config = {}

center_config['height'] = 512
center_config['width'] = 512
center_config['scale'] = 4
center_config['batch_size'] = 2
center_config['joints'] = 14
center_config['objs'] = 1
# center_config['train_json_file'] = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/train10.json'
center_config['train_json_file'] = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json'
center_config['train_img_path'] = '/media/hsw/E/datasets/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902'

# center_config['finetune'] = '/media/hsw/E/ckpt/spm_net/2019-09-12-14-19'
center_config['finetune'] = None
center_config['ckpt'] = '/media/hsw/E/ckpt/spm_net'