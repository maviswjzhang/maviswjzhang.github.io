---
layout:     post
title:      谷歌Colaboratory环境配置
subtitle:   提供免费Tesla K80 GPU
date:       2019-05-21
author:     Nick
header-img: img/博客背景.jpg
catalog: true
tags:
    - 环境搭建
---

# 1. Colaboratory 简介

Colaboratory 是一个 Google 研究项目，旨在帮助传播机器学习培训和研究成果。它就是一个 **Jupyter 笔记本**环境，并且完全在**云端**运行。总结来说，Colaboratory提供了jupyter笔记本编译环境和Tesla K80 GPU资源，我们只需要上传自己的数据集（谷歌云盘）和训练网络代码，就可以开始薅羊毛啦！

# 2. 配置步骤（尽量在翻墙环境下）

## 2.1 [谷歌账号注册](<https://accounts.google.com/signup/v2/webcreateaccount?flowName=GlifWebSignIn&flowEntry=SignUp>)

这块很简单了，能翻墙，傻瓜式下一步下一步。

## 2.2 访问[谷歌云盘](<https://drive.google.com/>)

访问并登陆谷歌云盘。谷歌云盘的作用主要有两个，一是存储数据集，二是存储新建的Colaboratory jupyter 文件。

**2.2.1 上传数据集**

* 左上角新建 --> 上传文件或文件夹 
* *此处建议上传数据集的压缩包（.zip格式），好处一是上传数据快，二是占用云盘空间小（总共免费15G）*

**2.2.2 新建Colaboratory**

* 左上角新建 --> 更多 --> Colaboratory(如果更多中没有Colaboratory选项，则选择关联更多应用，搜索Colaboratory并关联)

![tu](/img/2019-21-2.png)

## 2.3 Colaboratory 的jupyter notebook使用教程

![tu](/img/2019-21-3.png)

**2.3.1 页面整体布局**

* 上面菜单栏与本地jupyter相似
* 左侧显示云端的文件结构
* 右上角是**已连接**的云端服务器，每次进入要重新连接，就会自动给你分配资源

**2.3.2 设置GPU运行**

* 修改 --> 笔记本设置 -->在如下页面中修改硬件加速器维GPU并保存。

![tu ](/img/2019-21-4.png)

**2.3.3 运行jupyter 文件**

* **挂载谷歌云盘**：可以理解为链接谷歌云盘，代码如下

  ```python
  from google.colab import drive
  drive.mount("/ncc",force_remount=True) 
  ```

  其中` '/ncc' `是虚拟文件夹名称，理解起来就是，将你谷歌云盘下的文件链接到当前虚拟环境，具体链接到你所在虚拟服务器文件系统下的`/ncc//My Drive/`，也就是说你可以在该路径下找到你的谷歌云盘数据。

  运行上面程序后，会出现如下结果，

  ![tu](/img/2019-21-5.png)

  点击该链接，将拿到的密钥输入进入并回车

* **安装依赖库**

  Colaboratory 自带了 Tensorflow、Matplotlib、Numpy、Pandas 等深度学习基础库。如果还需要其他依赖，如 Keras，可以新建代码块，输入

  ```python
  !pip install keras
  ```

* **数据集解压**

  ```python
  !unzip "/ncc/My Drive/seg_label.zip"
  ```

  解压输入文件路径：`'/ncc/My Drive/seg_label.zip'；`

  解压输出文件路径：`'/content/'`，此时对应的把自己程序的数据路径改到该路径下就行；

* **程序代码执行**

  此处放上用keras运行mnist数据的程序示例，该数据集直接从keras中下载，不需要本地上传。

  我实验结果大概9s每个epoch，12个epoch可以达到99.18的精度。

  ```python
  '''Trains a simple convnet on the MNIST dataset.
  Gets to 99.25% test accuracy after 12 epochs
  (there is still a lot of margin for parameter tuning).
  16 seconds per epoch on a GRID K520 GPU.
  '''
  
  from __future__ import print_function
  import keras
  from keras.datasets import mnist
  from keras.models import Sequential
  from keras.layers import Dense, Dropout, Flatten
  from keras.layers import Conv2D, MaxPooling2D
  from keras import backend as K
  
  batch_size = 128
  num_classes = 10
  epochs = 12
  
  # input image dimensions
  img_rows, img_cols = 28, 28
  
  # the data, shuffled and split between train and test sets
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  
  if K.image_data_format() == 'channels_first':
      x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
      input_shape = (1, img_rows, img_cols)
  else:
      x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      input_shape = (img_rows, img_cols, 1)
  
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')
  
  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  
  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
  
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])
  ```

  
