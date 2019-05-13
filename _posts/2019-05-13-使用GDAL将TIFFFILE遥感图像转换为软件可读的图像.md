---
layout:     post
title:      使用GDAL将TIFFFILE遥感图像转换为软件可读的图像
subtitle:   tifffile & gdal
date:       2019-05-12
author:     Nick
header-img: img/博客背景.jpg
catalog: true
tags:
    - 数据预处理
---

#  1.问题描述：

一些情况下，使用python **tifffile** 库imwrite的遥感图像无法用ENVI等软件打开，由于缺少头文件信息。

# 2.解决方案

使用GDAL库将TIFFFILE遥感图像转换为软件可读的图像。

```python
from osgeo import gdal
import tifffile

# read TIF image using TIFFFILE and write again using GDAL
def update_msi(input_file_name, output_file_name):
    # 输入：图像路径和保存路径
    # 输出：无，直接保存到硬盘

    img = tifffile.imread(input_file_name)
    rows, cols, bands = img.shape
    driver = gdal.GetDriverByName("GTiff")
    output_data = driver.Create(output_file_name, rows, cols, bands, gdal.GDT_UInt16)
    for band in range(0, bands):
        output_band = output_data.GetRasterBand(band + 1)
        output_band.WriteArray(img[:, :, band])
    output_data.FlushCache()
    output_data = None

# 调用该函数
update_msi('E:/2019DFC_DATA/Train-Track1-MSI/JAX_031_005_MSI.tif', 'a.tif')
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

