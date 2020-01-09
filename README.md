Unofficial implementation of arxiv paper ["Single-Stage Multi-Person Pose Machines"](https://arxiv.org/abs/1908.09220), detail imformation can see this paper or check this [csdn link](https://blog.csdn.net/Murdock_C/article/details/100545377) only for reference.

## TODO
 
- [x] ~~custom distribute training is not work well, I trained for 10 epochs and nothing can be learned at all. So if anyone is familiar with this, please help me to check it and make it work.~~
The custom distribute training is right, but I write `checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)` is different with in `spm_model.py`. Because I write`checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model)` in `spm_model.py`, the parameter in `tf.train.Checkpoint`
is different, one is `net=model` and another is `model=model`. So, if I use checkpoints saved by  `checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)`, it is impossible using `checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model)` to restore it. So, we must keep parameters in `tf.train.Checkpoint` as same as possible.
By the way, it's a good way add `checkpoint.restore(ckpt_path).assert_existing_objects_matched()` to find restore error as soon as possible.
- [x] using tf.keras to run distribute training 
- [ ] add coco eval while training

## Requirement
* tensorflow 2.0.0
* python 3.6
* cuda 10
* [imgaug](https://github.com/aleju/imgaug) == 0.3.0
* pycocotools

## About Dataset

we use the first 12 points of ai-challenger format, which can found in this [website](https://challenger.ai/competition/keypoint). Maybe disabled, MSCOCO dataset is ok too, but need to delete five points on head and change its format just like ai-challenger. Note that we still use pycocotools to load data, so if you use ai-challenger, you need to translate its annos file format into coco annos format. [here](utils/convert_ai_format_to_coco.ipynb) is a convert code just for reference. 


## Network Structure

In this repo, just use [hrnet](https://github.com/VXallset/deep-high-resolution-net.TensorFlow) as for its body network, you can replace this body with any other network as you like. Please check for here: **`nets/spm_model.py`** 

## Single Gpu Training

`python3 main.py`

All config can be found in `config/center_config.py`

## Multi-GPU Training

`python3 distribute/custom_train.py`

~~**Note that if you have four gpus and its ids is [0, 1, 2, 3], and you want to use gpu id [2, 3] is not work very well for now. You can only use gpu id [0, 1] or [0, 1, 2] will work fine. I didn't know why and wish someone can tell me.**~~ 

The reason why we set `os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'` but can not use ` gpu_ids = [2, 3] ` is that tensorflow has already make gpu 2/3 on machine re-declear to 0/1. So, if we want to use `gpu_ids = [2, 3]`, just write:

```
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
gpu_ids = [0, 1]
devices = ['/device:GPU:{}'.format(i) for i in gpu_ids]
strategy = tf.distribute.MirroredStrategy(devices=devices)
```
in using distribute training.
  

## Test on images

`python3 tools/spm_model_test.py`

## Eval

create predicts json file

`python3 tools/model_val.py`

eval

`python3 tools/ai_format_kps_eval.py --ref true_label.json --submit predict.json`

detailed information can be found [here](https://github.com/AIChallenger/AI_Challenger_2017/tree/master/Evaluation/keypoint_eval) 

## About loss

In `spm_loss` function, you need carefully to set value of two different kinds of losses in order to make them balanced in numerical.

## ai_formate joints:

 1. right_shoulder 
 2. right_elbow
 3. right_wrist
 4. left_shoulder 
 5. left_elbow
 6. left_wrist
 7. right_hip  
 8. right_knee 
 9. right_ankle 
 10. left_hip 
 11. left_knee 
 12. left_ankle
 
## Thanks
[hrnet tensorlfow implementation](https://github.com/VXallset/deep-high-resolution-net.TensorFlow)

[PPN network for SPM author's another work](https://github.com/NieXC/pytorch-ppn)
 
 
