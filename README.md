# SWT-ResNet: A Novel Hybrid Model for Visual Defects Recognition Based on Stationary Wavelet Transform and CNN

<<<<<<< HEAD
Qiang Cui, Yafeng Li, Hongwei Bianc, Jie Kong, Yunwei Dong. ************************


# Project Completion Notice

Due to the collaborative nature of this work with the research institute, which involves confidential content, a model training and validation platform has been designed using a PyQt interface. The executable program can be found in the folder ui. 

The code publicly shares the design concepts of SWT-ResNet50 and the usage of the stationary wavelet transform. However, the actual implementation of the network architecture and training process is left to the user. Please note that the functions `wave1` through `wave4` within the SWT-ResNet50 network architecture need to be specifically designed for your tasks while maintaining channel merging with the backbone network.


### Running environment:
The proposed methods are implemented in Python 3.8 with PyTorch framework on a desktop computer equipped with an NVIDIA RTX 3080 GPU.
opencv-python==4.10
numpy==1.24.4
torch==1.10.0+cu113
torchvision==0.11.1+cu113
albumentations==1.4.14
PyWavelets==1.4.1
matplotlib==3.5.0
scikit-learn==1.0.2

### Dataset used in this paper:
1. [PCB_parts](https://www.kaggle.com/datasets/martinvajkuny/pcb-parts)
2. [Severstal Steel](https://www.kaggle.com/competitions/severstal-steel-defect-detection)
3. [AMT](https://pan.baidu.com/s/1lofG73Xg4Hz6ytBP30eBmg?pwd=79l7)
4. [HRIPCB](https://robotics.pkusz.edu.cn/resources/dataset/)

### Experimental results:
1.	classification based on SWT-ResNet with ** dataset.


# How to Use

## Adapting to Paper Results

This executable file(ui.exe) is designed to validate and evaluate the effectiveness of the methods and models proposed in the paper. Follow these steps:

1. **Select Model Corresponding to Dataset Parameters**:  
   In the application interface, **choose the model** you wish to use from the dropdown menu. These models correspond to different datasets and parameter configurations discussed in the paper. **Ensure that the selected model matches your dataset** for optimal results.

2. **Select the Folder Containing the Validation Dataset**:  
   Choose the folder that contains your validation dataset. **Each class of data should be placed in its own subfolder**, adhering to the expected structure. For example, your folder structure might look like this:

3. **Click "Export Validation Results"**:  
   Once you have selected the model and validation dataset, **click the "Export Validation Results" button**. This will initiate the validation process and output the results to a specified location.
=======
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

>>>>>>> f71b4be874d315ccadba1a67ee96d3aa762e1527
