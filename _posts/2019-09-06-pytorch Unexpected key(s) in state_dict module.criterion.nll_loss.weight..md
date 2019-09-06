---
layout:     post
title:      pytorch错误记录之Unexpected key(s) in state_dict module.criterion.nll_loss.weight
subtitle:   持续更新
date:       2019-09-06
author:     Nick
header-img: img/博客背景.jpg
catalog: true
tags:
    - pytorch

---

## 1. `Unexpected key(s) in state_dict: "module.backbone.bn1.num_batches_tracked"`

报错背景：

* 训练：多卡并行（`nn.DataParallel`），已完成；
* 测试：单卡测试，模型加载时报上述错误；
* 分析原因：在训练时使用多卡并行，保存的是经过`nn.DataParallel`编译过的模型，这样编译过之后state_dict中的keys就都会多一个前缀`module`，但是测试时使用单卡，模型参数一加载就会报错，那么此处有两种解决方案：

* 解决方案：

  * 第一种：测试时将单卡的模型也使用`nn.DataParallel`进行编译，如下：

  ```python
  model = torch.nn.DataParallel(model).cuda()
  ```

  * 第二种：将保存出去的模型中的前缀module都去掉，如下：

  ```python
  checkpoint = torch.load("densenet169_rnn_fold_1_model_best_f1.pth")
  model.load_state_dict({k.replace('module.',''): v for k,v in checkpoint['state_dict'].items()})
  ```

## 2. `Unexpected key(s) in state_dict module.criterion.nll_loss.weight.`

先说说报错背景（语义分割）：

* 训练：多卡并行（`nn.DataParallel`），由于电脑关机需要从中断处继续训练;
* 损失函数：按单张图加权的交叉熵损失函数；
* 继续训练：将模型使用`nn.DataParallel`编译之后，将最后保存的pth类型的checkpoint文件加载进来，代码如下，结果产生上述错误：

```python
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

```

* 问题原因：报错原因是现有的模型的state_dict中不存在`module.criterion.nll_loss.weight`这个参数，但是呢，加载进来的模型state_dict确有这个参数，我把新模型和旧模型的state_dict一打印，发现确实如此。这主要是因为这个交叉熵损失函数是加权的，旧模型的state_dict中存储了加权系数，但是新模型并没有这个参数，那么只需要把这个参数去掉就可以了啊，具体如下：

```python
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            
            model_dict = model.state_dict()
            old_dict = {k: v for k, v in checkpoint['state_dict'].items() if (k in model_dict)}
            model_dict.update(old_dict)
            model.load_state_dict(model_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
```

