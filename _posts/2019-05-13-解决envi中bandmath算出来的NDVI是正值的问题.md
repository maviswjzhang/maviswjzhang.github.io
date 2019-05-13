---
layout:     post
title:      为什么envi中band math算出来的NDVI MNDWI NDBI等全部是正值？
subtitle:   数据类型!!!!!
date:       2019-05-13
author:     Nick
header-img: img/博客背景.jpg
catalog: true
tags:
    - ENVI
---

# 1. 主要原因

**数据类型转换**，类型改成float。

例：当两个unit相比如：

unit a=4;

unit b=5;

![ndvi =\frac{a-b}{a+b} =-\frac{1}{9}](https://private.codecogs.com/gif.latex?ndvi%20%3D%5Cfrac%7Ba-b%7D%7Ba&plus;b%7D%20%3D-%5Cfrac%7B1%7D%7B9%7D)

由于unit/unit=unit，而unit类型又不能表示负数。所以怎么办呢？那就是把unit强制转化为int..

因此：unit类型的 **-1/9** 就被强制转化为了int类型的**4294967295 / 9**

# 2.解决办法

**强制类型转换**

在波段计算器中这样写,以landsat5计算NDVI为例，其他指数类似, 三种方式：

- (a) NDVI=(float(TM4)-float(TM3)) /（float(TM4)+float(TM3)）
- (b) NDVI=（TM4*1.0-TM3*1.0）/(TM4*1.0+TM3*1.0)
- (c) 就用ndvi这个工具，改一下波段就可以了，如要计算MNDWI=(2-5)/(2+5)

[![img](http://s10.sinaimg.cn/mw690/003uYJ3pzy72ZHwNG9b39)](http://photo.blog.sina.com.cn/showpic.html#blogid=&url=http://album.sina.com.cn/pic/003uYJ3pzy72ZHwNG9b39)

 # 3.参考

<http://blog.sina.com.cn/s/blog_bf1a24270102wjmh.html>
