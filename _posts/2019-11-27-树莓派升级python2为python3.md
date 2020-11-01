---
layout:     post
title:      树莓派——升级为python3.x
subtitle:   树莓派环境搭建第三步
date:       2019-11-27
author:     Mavis
header-img: img/博客背景.jpg
catalog: true
tags:
    - 树莓派
---

## 前言

树莓派或其他linux系统虽然自带了python，但都是版本较低的python2.x。考虑到今后长期的使用，我们将其升级为python3.x。

## 1. 安装python3.6

### 1.1 更新树莓派系统

```
$ sudo  apt-get  update
$ sudo  apt-get  upgrade -y
```

### 1.2 安装python依赖环境

```
$ sudo apt-get install build-essential libsqlite3-dev sqlite3 bzip2 libbz2-dev
```

### 1.3 下载python3.6版本源码并解压和编译

```python
# 到官网下载python3.6.1，但是下载特别慢，我连接了能够科学上网的win10开放的热点，后续会更新如何让树莓派科学上网

$ wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tgz
    
# 解压

$ tar zxvf Python-3.6.1.tgz

# 切换到解压目录

$ cd Python-3.6.1

# 开始编译

$ sudo ./configure 
$ sudo make 
$ sudo make install

# 升级pip

sudo python3.6 -m pip install --upgrade pip
```

## 2. 切换默认python

为了更方便地使用python3，我们将其设置为系统默认的版本。这里有两种方法，第一种是将原有的python2.x卸载，第二种是建立软链接。

### 2.1 卸载

`sudo apt-get autoremove python2.7`


卸载完后，我们还是需要新建一个链接来使得python可以出来python3.x

`sudo ln -s /usr/bin/python3.6 /usr/bin/python`

### 2.2 建立软链接

这一步主要是将原有的python2.x版本和python之间的链接删除（不像上面直接卸载python2.x）并添加python3.x的链接。

首先我们查看python和python3的详细版本：

安装python3.6后我们可以看一下python的版本

```python
$ python --version
Python 2.7.1
$ python3 --version
Python 3.6.1
```

接下来查看python和python3的编译器位置：

```python
$ which python
/usr/bin/python
$ which python3
/usr/bin/python3.6
```

然后建立软链接：

```python
$ sudo mv /usr/bin/python /usr/bin/python2.7.1
$ sudo ln -s /usr/bin/python3.6 /usr/bin/python
```

***注：以上路径及版本号视个人情况而改变！***

## 3. 测试

```python
$ python --version
Python 3.6.1
```

或者直接输入python看返回的python环境是否为python3.x
