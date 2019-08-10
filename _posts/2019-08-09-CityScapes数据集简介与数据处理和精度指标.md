---

layout:     post
title:      CityScapes数据集相关简介
subtitle:   数据下载、预处理、精度指标
date:       2019-08-10
author:     Nick
header-img: img/博客背景.jpg
catalog: true
tags:
    - 数据集
---

## 1. [数据集简介](https://www.cityscapes-dataset.com/)

Cityscapes数据集，即城市景观数据集，其中包含从50个不同城市的街景中记录的各种立体视频序列，除了更大的20000个弱注释帧之外，还有高质量的5000帧像素级注释。Cityscapes数据集共有fine和coarse两套评测标准，前者提供5000张精细标注的图像，后者提供5000张精细标注外加20000张粗糙标注的图像。train、val、test总共5000张精细释，2975张训练图，500张验证图和1525张测试图，每张图片大小都是`1024x2048`，官网下载是不包含测试集的标签的，需要在线评估。
Cityscapes数据集旨在用于：

* 评估视觉算法在语义城市场景理解主要任务中的表现：像素级，实例级和全景语义标签;
* 支持旨在利用大量（弱）注释数据的研究，例如用于训练深度神经网络；

```python
#       name                     id    trainId   category          catId        color
Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0      (  0,  0,  0) ),
Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0      , (  0,  0,  0) ),
Label(  'rectification border' ,  2 ,      255 , 'void'            , 0      , (  0,  0,  0) ),
Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0      , (  0,  0,  0) ),
Label(  'static'               ,  4 ,      255 , 'void'            , 0      , (  0,  0,  0) ),
Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0      , (111, 74,  0) ),
Label(  'ground'               ,  6 ,      255 , 'void'            , 0      , ( 81,  0, 81) ),
Label(  'road'                 ,  7 ,        0 , 'flat'            , 1      , (128, 64,128) ),
Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1      , (244, 35,232) ),
Label(  'parking'              ,  9 ,      255 , 'flat'            , 1      , (250,170,160) ),
Label(  'rail track'           , 10 ,      255 , 'flat'            , 1      , (230,150,140) ),
Label(  'building'             , 11 ,        2 , 'construction'    , 2      , ( 70, 70, 70) ),
Label(  'wall'                 , 12 ,        3 , 'construction'    , 2      , (102,102,156) ),
Label(  'fence'                , 13 ,        4 , 'construction'    , 2      , (190,153,153) ),
Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2      , (180,165,180) ),
Label(  'bridge'               , 15 ,      255 , 'construction'    , 2      , (150,100,100) ),
Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2      , (150,120, 90) ),
Label(  'pole'                 , 17 ,        5 , 'object'          , 3      , (153,153,153) ),
Label(  'polegroup'            , 18 ,      255 , 'object'          , 3      , (153,153,153) ),
Label(  'traffic light'        , 19 ,        6 , 'object'          , 3      , (250,170, 30) ),
Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3      , (220,220,  0) ),
Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4      , (107,142, 35) ),
Label(  'terrain'              , 22 ,        9 , 'nature'          , 4      , (152,251,152) ),
Label(  'sky'                  , 23 ,       10 , 'sky'             , 5      , ( 70,130,180) ),
Label(  'person'               , 24 ,       11 , 'human'           , 6      , (220, 20, 60) ),
Label(  'rider'                , 25 ,       12 , 'human'           , 6      , (255,  0,  0) ),
Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7      , (  0,  0,142) ),
Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7      , (  0,  0, 70) ),
Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7      , (  0, 60,100) ),
Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7      , (  0,  0, 90) ),
Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7      , (  0,  0,110) ),
Label(  'train'                , 31 ,       16 , 'vehicle'         , 7      , (  0, 80,100) ),
Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7      , (  0,  0,230) ),
Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7      , (119, 11, 32) ),
Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7      , (  0,  0,142) ),
```

| category（大类） | classes （小类）                                             |
| :--------------- | :----------------------------------------------------------- |
| flat             | road、sidewalk、parking、rail track                          |
| human            | person、rider                                                |
| vehicle          | car、truck、bus、on rails、motorcycle、bicycle、caravan、trailer |
| construction     | building、wall、fence、guard rail、bridge、tunnel            |
| object           | pole、pole group、traffic sign、traffic light                |
| nature           | vegetation、 terrain                                         |
| sky              | sky                                                          |
| void             | ground、dynamic、static                                      |

## 2. [数据集下载](https://www.cityscapes-dataset.com/downloads/)

![img](C:\Users\CV\Documents\GitHub\niecongchong.github.io\img\2019-08-10-1.png)

![img](C:\Users\CV\Documents\GitHub\niecongchong.github.io\img\2019-08-10-2.png)

**分别解压之后包含两个文件夹：**

* `gtFine`：每张原图对应4个标签文件

  * `bochum_xxxxxx_xxxxxx_gtFine_color.png`:可视化出来的彩色图像，对应第一部分表格最后一列；

  * `bochum_xxxxxx_xxxxxx_gtFine_instanceIds.png`:实例分割标签；

  * `bochum_xxxxxx_xxxxxx_gtFine_labelIds.png`:对应第一部分id列；
  * `bochum_xxxxxx_xxxxxx_gtFine_polygons.png`:手工标注的第一手数据；

* `leftImg8bit`
  * train: 18个城市
  - val:  3个城市
  - test:  6个城市

## 3. 数据预处理

考虑到原有的`1024x2048`的图像太大，显存实在是紧张，所以论文里面一般都会对其进行裁剪，我遵循[HRNet](https://arxiv.org/pdf/1904.04514.pdf)中的设置，将其处理成`512x1024`大小的训练图片，为了获得多尺度输入，采用resize和slide window crop两种方式：

```python
import numpy as np
import os
from glob import glob
import cv2

def genarate_dataset(data_dir, convert_dict, target_size, save_dir=None, flags=['train', 'val', 'test']):
    for flag in flags:
        save_num = 0
        # 获取待裁剪影像和label的路径
        images_paths = glob(data_dir + "leftImg8bit/" + flag + "/*/*_leftImg8bit.png")
        images_paths = sorted(images_paths)
        gts_paths = glob(data_dir + "gtFine/" + flag + "/*/*gtFine_labelIds.png")
        gts_paths = sorted(gts_paths)
        print(len(gts_paths))

        # 遍历每一张图片
        for image_path, gt_path in zip(images_paths, gts_paths):
            # 确保图片和标签对应
            image_name = os.path.split(image_path)[-1].split('_')[0:3]  
            # e.g. ['zurich', '000121', '000019']
            gt_name = os.path.split(gt_path)[-1].split('_')[0:3]
            assert image_name == gt_name

            # 读取图片和标签，并转换标签为0-19（20类，0是未分类）
            image = cv2.imread(image_path)
            gt = cv2.imread(gt_path, 0)
            binary_gt = np.zeros_like(gt)
            # 循环遍历字典的key，并累加value值
            for key in convert_dict.keys():
                index = np.where(gt == key)
                binary_gt[index] = convert_dict[key]

            # 尺寸
            target_height, target_width = target_size

            # ----------------- resize ----------------- #
            # resize, 参数输入是 宽×高, 不是常用的高×宽（多少行多少列）
            resize_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resize_gt = cv2.resize(binary_gt, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

            # save_path
            image_save_path = save_dir + flag + "/images/" + str(save_num) + "_resize.png"
            gt_save_path = save_dir + flag + "/gts/" + str(save_num) + "_resize.png"

            # save
            cv2.imwrite(image_save_path, resize_image)
            cv2.imwrite(gt_save_path, resize_gt)

            # 每保存一次图片和标签，计数加一
            save_num += 1

            # ----------------- slide crop ----------------- #
            h_arr = [0, 256, 512]
            w_arr = [0, 512, 1024]

            # 遍历长宽起始坐标列表,将原始图片随机裁剪为512大小
            for h in h_arr:
                for w in w_arr:
                    # crop
                    crop_image = image[h: h + target_height, w: w + target_width, :]
                    crop_gt = binary_gt[h: h + target_height, w: w + target_width]

                    # save_path
                    image_save_path = save_dir + flag + "/images/" + str(save_num) + "_crop.png"
                    gt_save_path = save_dir + flag + "/gts/" + str(save_num) + "_crop.png"

                    # save
                    cv2.imwrite(image_save_path, crop_image)
                    cv2.imwrite(gt_save_path, crop_gt)

                    # 每保存一次图片和标签，计数加一
                    save_num += 1

# 依据https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
# 不同之处在于将trainId里面的255定为第0类，原本的0-18类向顺序后加`1`
pixLabels = { 0:  0,    1:  0,    2:  0,    3:  0,    4:  0,    5:  0,
              6:  0,    7:  1,    8:  2,    9:  0,   10:  0,   11:  3,
             12:  4,   13:  5,   14:  0,   15:  0,   16:  0,   17:  6,
             18:  0,   19:  7,   20:  8,   21:  9,   22: 10,   23: 11,
             24: 12,   25: 13,   26: 14,   27: 15,   28: 16,   29:  0,
             30:  0,   31: 17,   32: 18,   33: 19,   -1:  0}

genarate_dataset(data_dir='../cityscapes/',
                 convert_dict=pixLabels,
                 target_size=(512, 1024),
                 save_dir='../CityScapesDataset/')
```

## 4. 精度指标

Cityscapes评测集给出的四项相关指标`IoU class / iIoU class / IoU category / iIou category`.

作为IoUclass之外的另外三样指标iIoUclass,IoUcategory,iIoUcategory，其设置的初衷在于对模型整体进行评估的基础上，从不同的角度对模型算法的效果进行探讨，以便进一步优化和改进模型的效果。

* `IoUClass`更多考虑的是全局场景的分割结果准确度，其所考量的是整个场景的划分准确度，因此Cityscapes评测集将其默认的优先参照指标。
* 而`iIoUClass`指标则是在`IoUClass`的基础上，对图像场景进行实例级的分割和结果预测。该指标以场景中的每个实例作为分割对象，类别相同的不同个体实例在该项评估中分别进行评测以得到最终的结果，该指标更加看重算法在各个实例之间的分割准确度。
* 此外，IoU指标又根据分割的粒度区分为`Class`和`Category`，其中19个细分类别（Class）被划分为7个粗分类别（Category）进行粗粒度的分割，并给出相应的评估指标IoUCategory和iIoUCategory。