---

layout:     post
title:      Non Local Means非局部均值滤波
subtitle:   经典图像滤波算法：A non-local algorithm for image denoising
date:       2019-08-26
author:     Nick
header-img: img/博客背景.jpg
catalog: true
tags:
    - 传统图像处理
---

## 1. 简介

Non-Local Means顾名思义，这是一种非局部平均算法。何为局部平均滤波算法呢？那是在一个目标像素周围区域平滑取均值的方法，所以非局部均值滤波就意味着它使用图像中的所有像素，这些像素根据某种相似度进行加权平均。滤波后图像清晰度高，而且不丢失细节。

## 2. 原理

该算法使用自然图像中普遍存在的冗余信息来去噪声。与双线性滤波、中值滤波等利用图像局部信息来滤波不同，它利用了整幅图像进行去噪。即以图像块为单位在图像中寻找相似区域，再对这些区域进行加权平均平均，较好地滤除图像中的高斯噪声。

### 2.1 公式原理

NL-Means的滤波过程可以用下面公式来表示：

![img](C:\Users\CV\Documents\GitHub\niecongchong.github.io\img\2019-08-26-2.png)

v代表噪声图像，NL[v]代表恢复图像，对于恢复图像中的任意一个像素i，其实通过图像中所有像素的加权平均得到的，也就是w，w代表了当前像素和其余像素的相似程度。衡量相似度的方法有很多，最常用的是根据两个像素亮度差值的平方来估计。由于有噪声，单独的一个像素并不可靠，所以使用它们的邻域，只有邻域相似度高才能说这两个像素的相似度高。衡量两个图像块的相似度最常用的方法是计算他们之间的欧氏距离：

![img](C:\Users\CV\Documents\GitHub\niecongchong.github.io\img\2019-08-26-3.png)

![img](C:\Users\CV\Documents\GitHub\niecongchong.github.io\img\2019-08-26-4.png)

其中 a 是高斯核的标准差。在求欧式距离的时候，不同位置的像素的权重是不一样的，距离块的中心越近，权重越大，距离中心越远，权重越小，权重服从高斯分布。实际计算中考虑到计算量的问题，常常采用均匀分布的权重。

### 2.2 图说原理

* 为了恢复p点的值，考虑到计算量的问题，我们会先预定一个**搜索窗口**，比如以当前点为中心21x21的窗口（红色框），我们要计算当前点p与窗口内21x21=441个点的权重（相似程度），进而加权这441个点得到恢复后的值。
* 如2.1所说，单个点的相似度不可靠，因此使用以两个点为中心的两个图像块（假设7x7大小）来衡量两者的相似程度，如下，p为待恢复的点，q1和q2的邻域图像块与p邻域图像块相似，所以权重w(p,q1) 和w(p,q2) 较大，而邻域相差比较大的点q3的权重值w(p,q3) 很小。
* 接下来说怎么计算w(p, q): w(p, q)是关于两个图像块的权重，也可以理解成相似度，这里使用欧式距离来衡量相似度，对于两个7x7的图像块，首先计算点对点的欧式距离，也就是得到7x7=49个距离，然而我们需要的是一个距离值，那就把这49个距离加权，这里使用高斯加权，距离中心越远的点权重越小，这样就得到了关于p， q两个图像块的欧式距离，最后使用exp（-distance）这个常用操作就可以将欧式距离转换成相似度，也就是w(p, q)。

***这里图片中的红色方框是我后期自己加的，他应该是以p点为中心的，示意图，就没那么严格***

![img](C:\Users\CV\Documents\GitHub\niecongchong.github.io\img\2019-08-26-5.png)

## 3. python opencv实现

```python
import cv2,datetime,sys,glob
import numpy as np
import  matplotlib.pyplot as plt
import matplotlib.cm as cm

image = cv2.imread('lena512.bmp')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
noise_image = double2uint8(image + np.random.randn(*image.shape) * 20)
out_image = cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10)

plt.figure()
plt.subplot(131)
plt.axis('off')
plt.imshow(image)
plt.subplot(132)
plt.axis('off')
plt.imshow(noise_image)
plt.subplot(133)
plt.axis('off')
plt.imshow(out_image)
plt.show()  
```

![img](C:\Users\CV\Documents\GitHub\niecongchong.github.io\img\2019-08-26-1.png)

## 4. 纯python 实现

```python
import cv2
import scipy as sc
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt


def double2uint8(I, ratio=1.0):
    return np.clip(np.round(I*ratio),0,255).astype(np.uint8)

def gaussian(l, sig):
    # Generate array
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    # Generate 2D matrices by duplicating ax along two axes
    xx, yy = np.meshgrid(ax, ax)
    # kernel will be the gaussian over the 2D arrays
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    # Normalise the kernel
    final = kernel / kernel.sum()
    return final


def clamp(p):
    """return RGB color between 0 and 255"""
    if p < 0:
        return 0
    elif p > 255:
        return 255
    else:
        return p


def means_filter(image, h=10, templateWindowSize=7, searchWindow=21):
    height, width = image.shape[0], image.shape[1]
    template_radius = int(templateWindowSize / 2)
    search_radius = int(searchWindow / 2)

    # Padding the image
    padLength = template_radius + search_radius
    img = cv2.copyMakeBorder(image, padLength, padLength, padLength, padLength, cv2.BORDER_CONSTANT, value=255)

    # output image
    out_image = np.zeros((height, width), dtype='float')

    # generate gaussian kernel matrix of 7*7
    kernel = gaussian(templateWindowSize, 1)

    # Run the non-local means for each pixel
    for row in range(height):
        for col in range(width):
            pad_row = row + padLength
            pad_col = col + padLength
            center_patch = img[pad_row - template_radius: pad_row + template_radius + 1, pad_col - template_radius: pad_col + template_radius + 1]

            sum_pixel_value = 0
            sum_weight = 0

            # Apply Gaussian weighted square distance between patches of 7*7 in a window of 21*21
            for r in range(pad_row - search_radius, pad_row + search_radius):
                for c in range(pad_col - search_radius, pad_col + search_radius):
                    other_patch = img[r - template_radius: r + template_radius + 1, c - template_radius: c + template_radius + 1]
                    diff = center_patch - other_patch
                    distance_2 = np.multiply(diff, diff)
                    pixel_weight = np.sum(np.multiply(kernel, distance_2))

                    pixel_weight = np.exp(pixel_weight / (h**2))
                    sum_weight = sum_weight + pixel_weight
                    sum_pixel_value = sum_pixel_value + pixel_weight * img[r, c]

            out_image[row, col] = clamp(int(sum_pixel_value / sum_weight))
    return out_image

# Call means_filter for the input image
if __name__ == "__main__":
    image = cv2.imread('lena512.bmp', 0)
    noise_image = double2uint8(image + np.random.randn(*image.shape) * 20)
    mean_image = means_filter(image, h=10, templateWindowSize=7, searchWindow=5)
    plt.figure()
    plt.subplot(131)
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.subplot(132)
    plt.axis('off')
    plt.imshow(noise_image, cmap='gray')
    plt.subplot(133)
    plt.axis('off')
    plt.imshow(mean_image, cmap='gray')
    plt.show()
```

![img](C:\Users\CV\Documents\GitHub\niecongchong.github.io\img\2019-08-26-6.png)