Unofficial implementation of arxiv paper ["Single-Stage Multi-Person Pose Machines"](https://arxiv.org/abs/1908.09220), detail imformation can see this paper or check this [csdn link](https://blog.csdn.net/Murdock_C/article/details/100545377) only for reference.

## Requirement
* tensorflow 2.0
* python 3.6
* cuda 10
* [imgaug](https://github.com/aleju/imgaug)

## Train Dataset

we use ai-challenger format dataset, which can found in this [website](https://challenger.ai/competition/keypoint). Maybe disabled, MSCOCO dataset is ok too, just change some file to suitable for its format.

## Network Structure

In this repo, just use [hrnet](https://github.com/VXallset/deep-high-resolution-net.TensorFlow) as for its body network, you can replace this body with any other network as you like. Please check for here: **`nets/spm_model.py`** 

## Train

`python3 main.py`

All config can be found in `config/center_config.py`

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
 
 