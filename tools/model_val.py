import tensorflow as tf
import os
import cv2
import numpy as np
import json
from nets.spm_model import SpmModel
from dataset.dataset import get_dataset
from loss.losses import spm_loss
from config.center_config import center_config as params
from decoder.decode_spm import SpmDecoder

colors = [[0,0,255],[255,0,0],[0,255,0],[0,255,255],[255,0,255],[255,255,0]]
netH = params['height']
netW = params['width']
score = 0.6
dist = 20
ckpt_path = '/media/hsw/E/ckpt/spm_net/ckpt-20'

params['val_json_file'] = 'jsons/val_500.json'

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
    ckpt = tf.train.Checkpoint(net=model)
    ckpt.restore(ckpt_path)

    val_dataset = get_dataset(mode='val')
    predictions = []

    for step, (imgids, heights, widths, imgs) in enumerate(val_dataset):
        center_map, kps_reg_map = model(imgs)
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
        print ("processing.... {} / {}".format(step, 500//params['batch_size']))
    print (len(predictions))
    with open('jsons/' + ckpt_path.split('/')[-1]+'_predicts.json', 'w') as fw:
        json.dump(predictions, fw)


            #### test
            # factor_x = 4
            # factor_y = 4
            # spm_decoder = SpmDecoder(factor_x, factor_y, netH // 4, netW // 4)
            # joints, centers = spm_decoder([center_map[b], kps_reg_map[b]], score_thres=score, dis_thres=dist)
            # img_ori = (imgs[b].numpy() * 255).astype(np.uint8)
            # for j, single_person_joints in enumerate(joints):
            #     cv2.circle(img_ori, (int(centers[j][0]), int(centers[j][1])), 8, colors[j % 6], thickness=-1)
            #     for i in range(14):
            #         x = int(single_person_joints[2 * i])
            #         y = int(single_person_joints[2 * i + 1])
            #         cv2.circle(img_ori, (x, y), 4, colors[j % 6], thickness=-1)
            #         cv2.putText(img_ori, str(i), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 250), 1)
            # cv2.imshow('batch_%d' % b, img_ori)
            # k = cv2.waitKey(0)
            # if k == 113:
            #     break
            ####


