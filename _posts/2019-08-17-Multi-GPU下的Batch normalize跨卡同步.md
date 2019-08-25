---

layout:     post
title:      Multi-GPU下的Batch normalize跨卡同步
subtitle:   针对语义分割等高显存占用任务多GPU并行时的坑
date:       2019-08-17
author:     Nick
header-img: img/博客背景.jpg
catalog: true
tags:
    - 语义分割
    - Training Tricks
    - 网络结构
---

## 1. 为什么要跨卡同步 Batch Normalization

现有的标准 Batch Normalization 因为使用数据并行（Data Parallel），是单卡的实现模式，只对单个卡上对样本进行归一化，相当于减小了批量大小（batch-size）, 若不进行同步BN，moving mean、moving variance参数会产生较大影响，造成BN层失效。对于比较消耗显存的训练任务时，往往单卡上的相对批量过小，影响模型的收敛效果。 在图像语义分割的实验中，使用大模型的效果反而变差，实际上就是BN在作怪。 跨卡同步 Batch Normalization 可以使用全局的样本进行归一化，这样相当于‘增大‘了批量大小，这样训练效果不再受到使用 GPU 数量的影响。 最近在图像分割、物体检测的论文中，使用跨卡BN也会显著地提高实验效果，所以跨卡 BN 已然成为竞赛刷分、发论文的必备神器。

## 2. Batch Normalization

### 2.1 [工作原理](https://arxiv.org/abs/1502.03167)

**内部协转移**（Internal Covariate Shift）：由于训练时网络参数的改变导致的网络层输出结果分布的不同。这正是导致网络训练困难的原因。

因此提出了两种简化方式来加速收敛速度：
1）对特征的每个维度进行标准化，忽略白化中的去除相关性；
2）在每个mini-batch中计算均值和方差来替代整体训练集的计算。

**BN前向过程如下：**

![img](/img/2019-08-17-6.png)

**BN反向过程如下：**

![img](/img/2019-08-17-7.png)

![img](/img/2019-08-17-1.jpg)

在训练时计算mini-batch的均值和标准差并进行反向传播训练，而测试时并没有batch的概念，训练完毕后需要提供固定的![img](/img/2019-08-17-8.png)供测试时使用。论文中对所有的mini-batch的![2019-08-17-9](/img/2019-08-17-9.png)取了均值:

![2019-08-17-10](/img/2019-08-17-10.png)

**测试阶段**，同样要进行归一化和缩放平移操作，唯一不同之处是不计算均值和方差，而使用训练阶段记录下来的![img](/img/2019-08-17-8.png)。

![img](/img/2019-08-17-11.png)

**Batch Norm优点:**

* 减轻过拟合
* 改善梯度传播（权重不会过高或过低）
* 容许较高的学习率，能够提高训练速度。
* 减轻对初始化权重的强依赖，使得数据分布在激活函数的非饱和区域，一定程度上解决梯度消失问题。
* 作为一种正则化的方式，在某种程度上减少对dropout的使用。

### 2.2 BN训练与测试过程

BN层有4个参数，gamma、beta、moving mean、moving variance。其中gamma、beta为学习参数，moving mean、moving variance为数据集统计均值与方差，不可学习。在训练过程中：

![img](/img/2019-08-17-4.png)

y为BN层输出，此时归一化的均值与方差为当前mini-batch的均值与方差。同时也记录moving mean、moving variance的值，每训练一个batch，moving mean、moving variance就更新一次。注意此参数更新过程不是学习过程，而是纯粹的计算train-set在当前BN数据分布过程，因此不能将算作是学习过程。decay为一个接近于1的值，比如0.9997。

在测试过程中：

![img](/img/2019-08-17-5.png)

### 2.3 数据并行

深度学习平台在多卡（GPU）运算的时候都是采用的数据并行（DataParallel），如下图:

![2019-08-17-2](/img/2019-08-17-2.jpg)

每次迭代，输入被等分成多份，然后分别在不同的卡上前向（forward）和后向（backward）运算，并且求出梯度，在迭代完成后合并 梯度、更新参数，再进行下一次迭代。因为在前向和后向运算的时候，每个卡上的模型是单独运算的，所以相应的Batch Normalization 也是在卡内完成，所以实际BN所归一化的样本数量仅仅局限于卡内，相当于批量大小（batch-size）减小了。

### 2.4 跨卡同步（Cross-GPU Synchronized）或 同步BN（SyncBN）

跨卡同步BN的关键是在前向运算的时候拿到全局的均值和方差，在后向运算时候得到相应的全局梯度。 最简单的实现方法是先同步求均值，再发回各卡然后同步求方差，但是这样就同步了两次。实际上只需要同步一次就可以，因为总体`batch_size`对应的均值和方差可以通过每张GPU中计算得到的![img](/img/2019-08-17-13.png) 得到. 在反向传播时也一样需要同步一次梯度信息, 祥见论文[Context Encoding for Semantic Segmentation](https://arxiv.org/pdf/1803.08904.pdf)：

![img](/img/2019-08-17-12.png)

![2019-08-17-3](/img/2019-08-17-3.jpg)

这样在前向运算的时候，我们只需要在各卡上算出与，再跨卡求出全局的和即可得到正确的均值和方差， 同理我们在后向运算的时候只需同步一次，求出相应的梯度与。 我们在最近的论文[Context Encoding for Semantic Segmentation](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1803.08904.pdf) 里面也分享了这种同步一次的方法。

有了跨卡BN我们就不用担心模型过大用多卡影响收敛效果了，因为不管用多少张卡只要全局的批量大小一样，都会得到相同的效果。

### 2.5  SyncBN现有资源

* [jianlong-yuan/syncbn-tensorflow](https://link.zhihu.com/?target=https%3A//github.com/jianlong-yuan/syncbn-tensorflow)重写了TensorFlow的官方方法，可以做个实验验证一下。
* [旷视科技：CVPR 2018 旷视科技物体检测冠军论文——大型Mini-Batch检测器MegDet](https://zhuanlan.zhihu.com/p/37847559)
* [tensorpack/tensorpack](https://link.zhihu.com/?target=https%3A//github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN)里面有该论文的代码实现。

## 3. some problems

### 3.1 为什么不用全局mean和var，能不能先把整个数据集的mean和var计算出来，在训练的时候设置好mean和var并且不更新（use_global_states = True）？

不可以。因为训练过程中weights 会变，这个时候mean 和var就会变 如果用global states的话就相当于没有batchnorm 这样的话就只能用小learning rate。最后的效果不会有batchnorm的好。

### 3.2 **为什么不进行多卡同步?**

现有框架BatchNorm的实现都是只考虑了single gpu。也就是说BN使用的均值和标准差是单个gpu算的，相当于缩小了mini－batch size。至于为什么这样实现，1）因为没有sync的需求，因为对于大多数vision问题，单gpu上的mini-batch已经够大了，完全不会影响结果。2）影响训练速度，BN layer通常是在网络结构里面广泛使用的，这样每次都同步一下GPUs，十分影响训练速度。

## 4. 参考链接

1. [跨卡同步 Batch Normalization](https://zhuanlan.zhihu.com/p/40496177)

2. [Keras multi-gpu batch normalization](https://datascience.stackexchange.com/questions/47795/keras-multi-gpu-batch-normalization)

3. [caffe：同步Batch Normalization(syncbn)作用](https://blog.csdn.net/l297969586/article/details/87719753)

4. [深度学习网络层之 Batch Normalization](https://www.cnblogs.com/makefile/p/batch-norm.html)

5. [Keras的BN你真的冻结对了吗](https://zhuanlan.zhihu.com/p/56225304)
