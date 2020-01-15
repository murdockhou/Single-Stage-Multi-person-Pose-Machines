#!/usr/bin/python3
# encoding: utf-8
'''
@author: matthew hsw
@contact: murdockhou@gmail.com
@software: pycharm
@file: losses.py
@time: 2019/7/25 下午2:41
@desc:
'''

import tensorflow as tf

def focal_loss(gt_center_map, pred_center_map, gt_center_map_mask=None):
    if gt_center_map_mask is not None:
        gt_center_map = tf.math.multiply(gt_center_map, gt_center_map_mask)
        pred_center_map = tf.math.multiply(pred_center_map, gt_center_map_mask)

    pos_inds = tf.cast(tf.equal(gt_center_map, 1.0), dtype=tf.float32)
    neg_inds = 1.0 - pos_inds
    # neg_inds=tf.cast(tf.greater(gt,0.0),dtype=tf.float32)-pos_inds
    neg_weights = tf.pow(1.0 - gt_center_map, 4.0)

    loss = 0.0
    pred = tf.clip_by_value(pred_center_map, 1e-4, 1.0 - 1e-4)
    pos_loss = tf.math.log(pred) * tf.pow(1.0 - pred, 2.0) * pos_inds
    neg_loss = tf.math.log(1.0 - pred) * tf.pow(pred, 2.0) * neg_weights * neg_inds

    num_pos = tf.reduce_sum(pos_inds)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    # print ('pos_loss {}, neg_loss {}, pos_num {}'.format(pos_loss.numpy(), neg_loss.numpy(), num_pos.numpy()))

    if num_pos == 0.0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def l2_loss(gt_center_map, pred_center_map, gt_center_map_mask=None):
    # tf.nn.l2_loss(a-b) * 2 == tf.reduce_sum(tf.losses.mean_squared_error(a, b))
    if gt_center_map_mask is not None:
        gt_center_map = tf.math.multiply(gt_center_map, gt_center_map_mask)
        pred_center_map = tf.math.multiply(pred_center_map, gt_center_map_mask)

    loss = tf.losses.mean_squared_error(gt_center_map, pred_center_map)
    return tf.reduce_sum(loss) / 100.

def reg_l1loss(pred,gt,mask):
    '''
    :param pred: (batch,h,w,2)
    :param gt: (batch,h,w,2)
    :param mask:(batch,h,w,1)
    :return:
    '''
    num_pos=(tf.reduce_sum(mask)+tf.convert_to_tensor(1e-4))
    loss=tf.abs(pred-gt)*mask
    loss=tf.reduce_sum(loss)/num_pos
    return loss

def center_loss(gt_center_map, gt_center_reg_map, gt_wh_map, gt_center_map_mask, gt_center_mask,
                pred_center_map, pred_center_reg_map, pred_wh_map):
    '''

    :param gt_center_map:
    :param gt_center_reg_map:
    :param gt_wh_map:
    :param gt_center_map_mask: 是指的打了高斯核的位置是哪些
    :param gt_center_mask:  是指哪些位置是真正的center点，比gt_center_map_mask表示的范围要小
    :param pred_center_map:
    :param pred_center_reg_map:
    :param pred_wh_map:
    :return:
    '''
    lambda_size = 0.1
    lambda_off = 1

    loss1 = focal_loss(gt_center_map, tf.nn.sigmoid(pred_center_map), None)
    # loss1 = l2_loss(gt_center_map, pred_center_map, None)
    loss2 = lambda_off * reg_l1loss(gt_center_reg_map, pred_center_reg_map, gt_center_mask)
    loss3 = lambda_size * reg_l1loss(gt_wh_map, pred_wh_map, gt_center_mask)

    loss = loss1 + loss2 + loss3
    return loss, [loss1, loss2, loss3]

def keypoints_loss(gt_center_kps_offset, gt_center_kps_mask, pred_center_kps_offset,
                   gt_kps_heatmap, gt_kps_heatmap_mask, pred_kps_heatmap,
                   gt_kps_offset, gt_kps_mask, pred_kps_offset):
    '''

    :param gt_center_kps_offset:
    :param gt_center_kps_mask: 某个中心点预测的14个关节点中，哪些点是真正的关节点
    :param pred_center_kps_offset:
    :param gt_kps_heatmap:
    :param gt_kps_heatmap_mask: 类似center_loss中的gt_center_map_mask
    :param pred_kps_heatmap:
    :param gt_kps_offset:
    :param gt_kps_mask: 类似center_loss中的gt_center_mask
    :param pred_kps_offset:
    :return:
    '''

    center_kps_offset_loss = reg_l1loss(pred_center_kps_offset, gt_center_kps_offset, gt_center_kps_mask)
    # heatmap_loss = focal_loss(gt_kps_heatmap, tf.nn.sigmoid(pred_kps_heatmap))
    heatmap_loss = tf.reduce_sum(tf.nn.l2_loss(gt_kps_heatmap-pred_kps_heatmap))
    kps_offset = reg_l1loss(pred_kps_offset, gt_kps_offset, gt_kps_mask)

    loss = center_kps_offset_loss+heatmap_loss+kps_offset

    return loss,  [center_kps_offset_loss, heatmap_loss, kps_offset]

def SmoothL1Loss(label, pred):

        t = tf.abs(label - pred)

        return tf.reduce_sum(
            tf.where(
                t <= 1, 0.5 * t * t, 0.5 * (t - 1)
            )
        )


def spm_loss(center_map, center_mask,
             center_offset, center_offset_mask,
             pred_center_map, pred_center_offset):

    center_weight = 1
    offset_weight = 1

    # tf.reduce_mean(tf.keras.losses.MSE()) 和 pytorch中的torch.nn.MSELoss()取默认参数时值一样
    # root_joint_loss = tf.reduce_mean(tf.keras.losses.MSE(gt_root_joint*gt_root_joint_mask, root_pred*gt_root_joint_mask))


    # huber loss 就是 smooth l1 loss，和pytorch中的torch.nn.SmoothL1Loss()结果一致
    # huber_loss = tf.losses.Huber()
    # pred_joint_loss = huber_loss(gt_joint_offset*gt_joint_offset_weight, root_kps_offset_pred*gt_joint_offset_weight)

    # nums1 = tf.reduce_sum(gt_root_joint_mask)
    # nums2 = tf.reduce_sum(gt_joint_offset_weight)

    # print (tf.reduce_sum(center_mask), center_mask.shape)
    center_loss = tf.reduce_sum(tf.nn.l2_loss(center_map * center_mask - pred_center_map * center_mask)) 
    center_offset_loss = SmoothL1Loss(center_offset * center_offset_mask, pred_center_offset * center_offset_mask) 

    # root_joint_loss = tf.reduce_sum(tf.nn.l2_loss(gt_root_joint-root_pred)) 
    # pred_joint_loss = SmoothL1Loss(gt_joint_offset, root_kps_offset_pred)


    return [center_loss * center_weight, center_offset_loss * offset_weight]

