# AoANet-Paddle

基于[paddle](https://github.com/PaddlePaddle/Paddle)框架的[Attention on Attention for Image Captioning](https://arxiv.org/abs/1908.06954)实现

## 一、简介

本项目基于[paddle](https://github.com/PaddlePaddle/Paddle)复现[Attention on Attention for Image Captioning](https://arxiv.org/abs/1908.06954)中所提出的`Attention on Attention`模型。该模型在传统的`self-attention`注意力机制的基础上，添加了`gate`机制以过滤和`query`不相关的`attention`信息。同时，作者还引入`multi-head attention`用于建模不同目标之间的关系。

**论文:**

* [1] L. Huang, W. Wang, J. Chen, X. Wei, "Attention on Attention for Image Captioning", ICCV, 2019.

**参考项目:**

* [https://github.com/husthuaan/AoANet](https://github.com/husthuaan/AoANet) [官方实现]

## 二、复现精度

> 所有指标均为模型在[COCO2014](https://cocodataset.org/)的测试集评估而得

| 指标 | BlEU-1 | BlEU-2 | BlEU-3 | BlEU-4 | METEOR | ROUGE-L | CIDEr-D | SPICE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 原论文 | 0.805 | 0.652 | 0.510 | 0.391 | 0.290 | 0.589 | 1.289 | 0.227|
| 复现精度 | - | - | - | - | - | - | - | - |


## 三、数据集

本项目所使用的数据集为[COCO2014](https://cocodataset.org/)。该数据集共包含123287张图像，每张图像对应5个标题。训练集、验证集和测试集分别为113287、5000、5000张图像及其对应的标题。本项目使用预提取的`bottom-up`特征，可以从[这里](https://github.com/peteanderson80/bottom-up-attention)下载得到（我们提供了脚本下载该数据集的标题以及图像特征，见[download_dataset.sh](https://github.com/fuqianya/bottom-up-attention-paddle/download_dataset.sh)）。

## 四、环境依赖

* 硬件：CPU、GPU ( > 11G )

* 软件：
    * Java 1.8.0
    * PaddlePaddle == 2.1.0

## 五、快速开始

### step1: clone 

```bash
# clone this repo
git clone https://github.com/fuqianya/AoANet-Paddle.git
cd AoANet-Paddle
```

### step2: 安装环境及依赖

```bash
pip install -r requirements.txt
```

### step3: 下载数据

```bash
# 下载数据集及特征
bash ./download_dataset.sh
# 下载
```

### step4: 数据集预处理

```python
python prepro.py
```

### step5: 