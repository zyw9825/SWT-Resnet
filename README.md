# SWT-ResNet: A Novel Hybrid Model for Visual Defects Recognition Based on Stationary Wavelet Transform and CNN

Qiang Cui, Yafeng Li, Hongwei Bianc, Jie Kong, Yunwei Dong.

If you have questions about our methods and code, please contact: cqnwpu@163.com

# Project Completion Notice

The code for this project was finalized on October 14, 2024.

If you find this code is useful and helpful to your work, please cite our paper in your research work. Thanks.

### Running environment:

The proposed methods are implemented in Python 3.8 with PyTorch framework on a desktop computer equipped with an NVIDIA RTX 3080 GPU.
- opencv-python==4.10
- numpy==1.24.4
- torch==1.10.0+cu113
- torchvision==0.11.1+cu113
- albumentations==1.4.14
- PyWavelets==1.4.1
- matplotlib==3.5.0
- scikit-learn==1.0.2

### Dataset used in this paper:

1. [PCB_parts](https://www.kaggle.com/datasets/martinvajkuny/pcb-parts)
2. [Severstal Steel](https://www.kaggle.com/competitions/severstal-steel-defect-detection)
3. [AMT](https://pan.baidu.com/s/1lofG73Xg4Hz6ytBP30eBmg?pwd=79l7)
4. [HRIPCB](https://robotics.pkusz.edu.cn/resources/dataset/)

### Experimental results:

1.	classification based on SWT-ResNet with ** dataset.

### How to use:

1. find the `train.py` file in the folder:
2. train the model by running the `train.py` file.
3. evaluating results will automatically show after training.

