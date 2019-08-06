---
layout:     post
title:      keras之学习率调整（内含论文中常见的poly衰减策略）
subtitle:   keras进阶（一）
date:       2019-08-06
author:     Nick
header-img: img/博客背景.jpg
catalog: true
tags:
    - Keras
---

Keras提供两种学习率调整方法，都是通过回调函数来实现。

* `LearningRateScheduler`
* `ReduceLROnPlateau`

## 1. `LearningRateScheduler`

```python
keras.callbacks.LearningRateScheduler(schedule)
```

学习速率**定时器**，也就是说，定的是啥就是啥，严格按照定时器进行更改。

- schedule: 一个函数，接受epoch作为输入（整数，从 0 开始迭代）， 然后返回一个学习速率作为输出（浮点数）。

```python
import keras.backend as K
from keras.callbacks import LearningRateScheduler
 
def scheduler(epoch):
    # 每隔10个epoch，学习率减小为原来的1/10
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

# def poly_decay(epoch):
#     # initialize the maximum number of epochs, base learning rate,
#     # and power of the polynomial
#     maxEpochs = epochs
#     baseLR = learning_rate
#     power = 1.0
# 
#     # compute the new learning rate based on polynomial decay
#     alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
# 
#     return alpha
# 
# 
# poly_reduce_lr = LearningRateScheduler(poly_decay)
 
reduce_lr = LearningRateScheduler(scheduler)
model.fit(train_x, train_y, batch_size=128, epochs=50, callbacks=[reduce_lr])
```

## 2. `ReduceLROnPlateau`

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
```

当评估指标停止提升时，降低学习速率。

- **monitor**: 被监测的指标。
- **factor**: 学习速率被降低的因数。新的学习速率 = 学习速率 * 因数
- **patience**: 没有提升的训练轮数，在这之后训练速率会被降低。
- **verbose**: 整数。0：安静，1：更新信息。
- **mode**: {auto, min, max} 其中之一。如果是 `min` 模式，学习速率会被降低如果被监测的数据已经停止下降； 在 `max` 模式，学习塑料会被降低如果被监测的数据已经停止上升； 在 `auto` 模式，方向会被从被监测的数据中自动推断出来。
- **min_delta**: 阈值，用来确定是否进入检测值的“平原区”
- **cooldown**: 在学习速率被降低之后，重新恢复正常操作之前等待的训练轮数量。
- **min_lr**: 学习速率的下边界。

```python
from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='min')
model.fit(train_x, train_y, batch_size=128, epochs=50, validation_split=0.1, callbacks=[reduce_lr])
```

