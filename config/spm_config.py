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
spm_config['num_joints'] = 12

spm_config['use_coco'] = False
if spm_config['use_coco']:
    spm_config['train_json_file'] = '/media/hsw/E/datasets/multipose_with_only_12_body_joints/coco_person_keypoints_train2017.json'
    spm_config['train_img_path'] = '/media/hsw/E/datasets/coco/images/train2017'

    spm_config['test_json_file'] = '/media/hsw/E/datasets/multipose_with_only_12_body_joints/coco_person_keypoints_val2017.json'
    spm_config['test_img_path'] = '/media/hsw/E/datasets/coco/images/val2017'
else:
    spm_config['train_json_file'] = '/media/hsw/E/datasets/multipose_with_only_12_body_joints/aitrain.json'
    spm_config['train_img_path'] = '/media/hsw/E/datasets/ai-challenger/ai_train/train'

    spm_config['test_json_file'] = '/media/hsw/E/datasets/multipose_with_only_12_body_joints/aival.json'
    spm_config['test_img_path'] = '/media/hsw/E/datasets/ai-challenger/ai_test/ai_val/val'

spm_config['finetune'] = None
