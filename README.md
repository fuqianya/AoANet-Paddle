# AoANet-Paddle

基于[paddle](https://github.com/PaddlePaddle/Paddle)框架的[Attention on Attention for Image Captioning](https://arxiv.org/abs/1908.06954)实现

## 一、简介

本项目基于[paddle](https://github.com/PaddlePaddle/Paddle)复现[Attention on Attention for Image Captioning](https://arxiv.org/abs/1908.06954)中所提出的`Attention on Attention`模型。该模型在传统的`self-attention`注意力机制的基础上，添加了`gate`机制以过滤和`query`不相关的`attention`信息。同时，作者还引入`multi-head attention`用于建模不同目标之间的关系。

**注: AI Studio项目地址: [https://aistudio.baidu.com/aistudio/projectdetail/2470054](https://aistudio.baidu.com/aistudio/projectdetail/2470054).**

**您可以使用[AI Studio](https://aistudio.baidu.com/)平台在线运行该项目!**

**论文:**

* [1] L. Huang, W. Wang, J. Chen, X. Wei, "Attention on Attention for Image Captioning", ICCV, 2019.

**参考项目:**

* [https://github.com/husthuaan/AoANet](https://github.com/husthuaan/AoANet) [官方实现]

## 二、复现精度

> 所有指标均为模型在[COCO2014](https://cocodataset.org/)的测试集评估而得

| 指标 | BlEU-1 | BlEU-2 | BlEU-3 | BlEU-4 | METEOR | ROUGE-L | CIDEr-D | SPICE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 原论文 | 0.805 | 0.652 | 0.510 | 0.391 | 0.290 | 0.589 | 1.289 | 0.227 |
| 复现精度 | 0.802 | 0.648 | 0.504 | 0.385 | 0.286 | 0.585 | 1.271 | 0.222 |

## 三、数据集

本项目所使用的数据集为[COCO2014](https://cocodataset.org/)。该数据集共包含123287张图像，每张图像对应5个标题。训练集、验证集和测试集分别为113287、5000、5000张图像及其对应的标题。本项目使用预提取的`bottom-up`特征，可以从[这里](https://github.com/peteanderson80/bottom-up-attention)下载得到（我们提供了脚本下载该数据集的标题以及图像特征，见[download_dataset.sh](https://github.com/fuqianya/AoANet-Paddle/blob/main/download_dataset.sh)）。

## 四、环境依赖

* 硬件：CPU、GPU ( > 11G )

* 软件：
    * Python 3.8
    * Java 1.8.0
    * PaddlePaddle == 2.1.0

## 五、快速开始

### step1: clone 

```bash
# clone this repo
git clone https://github.com/fuqianya/AoANet-Paddle.git --recursive
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
# 下载与计算评价指标相关的文件
bash ./coco-caption/get_google_word2vec_model.sh
bash ./coco-caption/get_stanford_models.sh
```

### step4: 数据集预处理

```python
python prepro.py
```

### step5: 训练

训练过程过程分为两步(详情见论文3.3节):

* Training with Cross Entropy (XE) Loss

  ```bash
  bash ./train_xe.sh
  ```

* CIDEr-D Score Optimization

  ```bash
  bash ./train_rl.sh
  ```
### step6: 测试

* 测试`train_xe`阶段的模型

  ```bash
  python eval.py --model log/log_aoa/model.pth --infos_path log/log_aoa/infos_aoa.pkl --num_images -1 --language_eval 1 --beam_size 2 --batch_size 100 --split test
  ```
* 测试`train_rl`阶段的模型
  ```bash
  python eval.py --model log/log_aoa_rl/model.pth --infos_path log/log_aoa_rl/infos_aoa.pkl --num_images -1 --language_eval 1 --beam_size 2 --batch_size 100 --split test
  ```

你将分别得到和以下分数相似的结果:
```python
{'Bleu_1': 0.7729384559899702, 'Bleu_2': 0.6163398035383025, 'Bleu_3': 0.4790123137715982, 'Bleu_4': 0.36944349063530374, 'METEOR': 0.2848188431924821, 'ROUGE_L': 0.5729849683867054, 'CIDEr': 1.1842173801790759, 'SPICE': 0.21650786258302354}
```

```python
{'Bleu_1': 0.8054903453672397, 'Bleu_2': 0.6523038976984842, 'Bleu_3': 0.5096621263772566, 'Bleu_4': 0.39140307771618477, 'METEOR': 0.29011216375635934, 'ROUGE_L': 0.5890369750273199, 'CIDEr': 1.2892294296245852, 'SPICE': 0.22680092759866174}
```

### 使用预训练模型进行预测

模型下载: [谷歌云盘](https://drive.google.com/drive/folders/1SjMtmtu9z5tdmZUplUGOBnIA5jyv_PSu?usp=sharing)

将下载的模型权重以及训练信息放到`log`目录下, 运行`step6`的指令进行测试。

## 六、代码结构与详细说明

```bash
├── cider              　# 计算评价指标工具
├── coco-caption       　# 计算评价指标工具
├── config
│　 └── config.py        # 模型的参数设置
├── data            　   # 预处理的数据
├── log             　   # 存储训练模型及历史信息
├── model
│   └── AoAModel.py    　# 定义模型结构
│   └── dataloader.py  　# 加载训练数据
│   └── loss.py        　# 定义损失函数
├── utils 
│   └── eval_utils.py  　# 测试工具
│   └── utils.py    　   # 其他工具
├── download_dataset.sh　# 数据集下载脚本
├── prepro.py          　# 数据预处理
├── train.py           　# 训练主函数
├── eval.py            　# 测试主函数
├── train_xe.sh       　 # 训练脚本
├── train_rl.sh       　 # 训练脚本
└── requirement.txt   　 # 依赖包
```

模型、训练的所有参数信息都在`config.py`中进行了详细注释，详情见`config/config.py`。

## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| :---: | :---: |
| 发布者 | fuqianya |
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1.0 |
| 应用场景 | 多模态 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [预训练模型](https://drive.google.com/drive/folders/1SjMtmtu9z5tdmZUplUGOBnIA5jyv_PSu?usp=sharing) \| [训练日志](https://drive.google.com/file/d/1_sfdhtL7hGQSbBL4kRw4_bBP8y6QdzZu/view?usp=sharing)  |
