import tensorflow as tf
import os
import cv2
import numpy as np
import json
import time


from nets.spm_model import SpmModel
from dataset.dataset import get_dataset
from config.center_config import center_config as params
from decoder.decode_spm import SpmDecoder

colors = [[0,0,255],[255,0,0],[0,255,0],[0,255,255],[255,0,255],[255,255,0]]
netH = params['height']
netW = params['width']
score = 0.6
dist = 20
# ckpt_path = '/home/hsw/server/ckpt-74/ckpt-84/ckpt-87'
ckpt_path = '/home/hsw/server/multi_pose/spm/keras/ckpt_'
# params['val_json_file'] = 'jsons/val_500.json'
ckpts = [1, 2, 3, 4]

@tf.function
def infer(model, inputs):

    center_map, kps_reg_map = model(inputs)

    return center_map, kps_reg_map


if __name__ == '__main__':

    use_gpu = True
    use_nms = True

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    inputs = tf.keras.Input(shape=(netH, netW, 3), name='modelInput')
    outputs = SpmModel(inputs, 14, is_training=False)
    model = tf.keras.Model(inputs, outputs)

    # ckpt = tf.train.Checkpoint(net=model)
    # ckpt.restore(ckpt_path).assert_existing_objects_matched()

    for ckpt in ckpts:
        model.load_weights(ckpt_path+str(ckpt))

        val_dataset = get_dataset(mode='val')
        predictions = []

        for step, (imgids, heights, widths, imgs) in enumerate(val_dataset):
            s = time.time()
            center_map, kps_reg_map = infer(model, imgs)
            imgids = imgids.numpy()
            for b in range(params['batch_size']):
                factor_x = widths[b].numpy() / (netW / 4)
                factor_y = heights[b].numpy() / (netH / 4)
                spm_decoder = SpmDecoder(factor_x, factor_y, netH // 4, netW // 4)
                joints, centers = spm_decoder([center_map[b], kps_reg_map[b]], score_thres=score, dis_thres=dist)
                img_id = str(imgids[b], encoding='utf-8')

                predict = {}
                predict['image_id'] = img_id
                kps = {}
                bbox = {}
                human = 1
                for j, single_person_joints in enumerate(joints):
                    joints = []
                    for i in range(14):
                        x = int(single_person_joints[2 * i])
                        y = int(single_person_joints[2 * i + 1])
                        v = 1
                        joints += [x, y, v]
                    kps['human' + str(human)] = joints
                    # bbox['human' + str(human)] = None
                    human += 1
                predict['keypoint_annotations'] = kps
                predictions.append(predict)
            e = time.time()
            print("processing.... {} / {}, time cost == {}".format(step, 30000 // params['batch_size'], e-s))

        print(len(predictions))
        if 'keras' in ckpt_path:
            json_file = 'jsons/keras/ckpt_16_finetune/' + str(ckpt) + '_predicts.json'
        else:
            json_file = 'jsons/' + str(ckpt) + '_predicts.json'

        with open(json_file, 'w') as fw:
            json.dump(predictions, fw)
            print(ckpt, ' is done.')

