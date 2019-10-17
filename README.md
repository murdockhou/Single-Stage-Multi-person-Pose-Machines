Unofficial implementation of arxiv paper ["Single-Stage Multi-Person Pose Machines"](https://arxiv.org/abs/1908.09220), detail imformation can see this paper or check this [csdn link](https://blog.csdn.net/Murdock_C/article/details/100545377) only for reference.

## Requirement
* tensorflow 2.0.0
* python 3.6
* cuda 10
* [imgaug](https://github.com/aleju/imgaug) == 0.3.0

## Train Dataset

we use ai-challenger format dataset, which can found in this [website](https://challenger.ai/competition/keypoint). Maybe disabled, MSCOCO dataset is ok too, just change some file to suitable for its format.

## Network Structure

In this repo, just use [hrnet](https://github.com/VXallset/deep-high-resolution-net.TensorFlow) as for its body network, you can replace this body with any other network as you like. Please check for here: **`nets/spm_model.py`** 

## Single Gpu Training

`python3 main.py`

All config can be found in `config/center_config.py`

## Multi-GPU Training

`python3 distribute_main.py`

**Note that if you have four gpus and its ids is [0, 1, 2, 3], and you want to use gpu id [2, 3] is not work very well for now. You can only use gpu id [0, 1] or [0, 1, 2] will work fine. I didn't know why and wish someone can tell me.**

## Test on images

`python3 tools/spm_model_test.py`

## Eval

create predicts json file

`python3 tools/model_val.py`

eval

`python3 tools/ai_format_kps_eval.py --ref true_label.json --submit predict.json`

detailed information can be found [here](https://github.com/AIChallenger/AI_Challenger_2017/tree/master/Evaluation/keypoint_eval) 

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
 13. head
 14. neck
 
## Thanks
[hrnet tensorlfow implementation](https://github.com/VXallset/deep-high-resolution-net.TensorFlow)

[PPN network for SPM author's another work](https://github.com/NieXC/pytorch-ppn)
 
 