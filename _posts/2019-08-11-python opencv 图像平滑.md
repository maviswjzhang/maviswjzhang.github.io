---
layout:     post
title:      OpenCV-Python学习—图像平滑
subtitle:   转载：https://www.cnblogs.com/silence-cho/p/11027218.html
date:       2019-08-11
author:     Nick
header-img: img/博客背景.jpg
catalog: true
tags:
    - Opencv
    - Python
    - 传统图像处理
---

由于种种原因，图像中难免会存在噪声，需要对其去除。噪声可以理解为灰度值的随机变化，即拍照过程中引入的一些不想要的像素点。噪声可分为椒盐噪声，高斯噪声，加性噪声和乘性噪声等，参见：https://zhuanlan.zhihu.com/p/52889476

噪声主要通过平滑进行抑制和去除，包括基于二维离散卷积的高斯平滑，均值平滑，基于统计学的中值平滑，以及能够保持图像边缘的双边滤波，导向滤波算法等。下面介绍其具体使用。

## 1. 二维离散卷积

　　理解卷积：https://www.zhihu.com/question/22298352/answer/637156871

　　　　　　　https://www.zhihu.com/question/22298352/answer/228543288

学习图像平滑前，有必要了解下卷积的知识，看完上述连接，对于图像处理中卷积应该了解几个关键词：卷积核，锚点，步长，内积，卷积模式。

* **卷积核(kernel)**：用来对图像矩阵进行平滑的矩阵，也称为过滤器（filter）
* **锚点**：卷积核和图像矩阵重叠，进行内积运算后，锚点位置的像素点会被计算值取代。一般选取奇数卷积核，其中心点作为锚点
* **步长**：卷积核沿着图像矩阵每次移动的长度
* **内积**：卷积核和图像矩阵对应像素点相乘，然后相加得到一个总和，如下图所示。（不要和矩阵乘法混淆）

![img](/img/2019-08-11-1.png)

* **卷积模式**：卷积有三种模式，FULL, SAME，VALID，实际使用注意区分使用的那种模式。（参考：https://zhuanlan.zhihu.com/p/62760780）

  * Full：全卷积，full模式的意思是，**从filter和image刚相交开始做卷积，**白色部分为填0，橙色部分为image, 蓝色部分为filter，filter的运动范围如图所示。

  ![img](/img/2019-08-11-2.png)

  * Same卷积：**当filter的锚点(K)与image的边角重合时，开始做卷积运算**，可见filter的运动范围比full模式小了一圈，same mode为full mode 的子集，即full mode的卷积结果包括same mode。
  * ![2019-08-11-3](/img/2019-08-11-3.png)
  * valid卷积：**当filter全部在image里面的时候，进行卷积运算**，可见filter的移动范围较same更小了，同样valid mode为same mode的子集。valid mode的卷积计算，填充边界中的像素值不会参与计算，即无效的填充边界不影响卷积，所以称为valid mode。
  * ![2019-08-11-4](/img/2019-08-11-4.png)

python的scipy包中提供了convolve2d()函数来实现卷积运算，其参数如下：

```python
from scipy import signal

signal.convolve2d(src,kernel,mode,boundary,fillvalue)
    
src: 输入的图像矩阵，只支持单通的（即二维矩阵）
kernel：卷积核
mode：卷积类型：full, same, valid
boundary:边界填充方式：fill，wrap, symm
fillvalue: 当boundary为fill时，边界填充的值，默认为0
```

opencv中提供了flip()函数翻转卷积核，filter2D进行same 卷积, 其参数如下：

```python
dst = cv2.flip(src,flipCode)
    src: 输入矩阵
    flipCode:0表示沿着x轴翻转，1表示沿着y轴翻转，-1表示分别沿着x轴，y轴翻转
    dst:输出矩阵（和src的shape一样）
    
cv2.filter2D(src,dst,ddepth,kernel,anchor=(-1,-1),delta=0,borderType=cv2.BORDER_DEFAULT)
    src: 输入图像对象矩阵
    dst:输出图像矩阵
    ddepth:输出矩阵的数值类型
    kernel:卷积核
    anchor：卷积核锚点，默认(-1,-1)表示卷积核的中心位置
    delat:卷积完后相加的常数
    borderType:填充边界类型
```

## 2 图像平滑

### 2.1 高斯平滑

高斯平滑即采用高斯卷积核对图像矩阵进行卷积操作。高斯卷积核是一个近似服从高斯分布的矩阵，随着距离中心点的距离增加，其值变小。这样进行平滑处理时，图像矩阵中锚点处像素值权重大，边缘处像素值权重小，下为一个3*3的高斯卷积核：

![img](/img/2019-08-11-5.png)

opencv中提供了GaussianBlur()函数来进行高斯平滑，其对应参数如下：

```python
dst = cv2.GaussianBlur(src,ksize,sigmaX,sigmay,borderType)
        src: 输入图像矩阵,可为单通道或多通道，多通道时分别对每个通道进行卷积
        dst:输出图像矩阵,大小和数据类型都与src相同
        ksize:高斯卷积核的大小，宽，高都为奇数，且可以不相同
        sigmaX: 一维水平方向高斯卷积核的标准差
        sigmaY: 一维垂直方向高斯卷积核的标准差，默认值为0，表示与sigmaX相同
        borderType:填充边界类型
```

代码使用示例和效果如下：(相比于原图，平滑后图片变模糊)

```python
#coding:utf-8

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


img = cv.imread(r"C:\Users\Administrator\Desktop\timg.jpg")
img_gauss = cv.GaussianBlur(img,(3,3),1)
cv.imshow("img",img)
cv.imshow("img_gauss",img_gauss)
cv.waitKey(0)
cv.destroyAllWindows()
```

![2019-08-11-6](/img/2019-08-11-6.png)

对于上面的高斯卷积核，可以由如下两个矩阵相乘进行构建，说明高斯核是**可分离卷积核**，因此高斯卷积操作可以分成先进行垂直方向的一维卷积，再进行一维水平方向卷积。

![2019-08-11-7](/img/2019-08-11-7.png)

opencv中getGaussianKernel()能用来产生一维的高斯核，分别获得水平和垂直的高斯核，分两步也能完成高斯卷积，获得和GaussianBlur一样的结果。其参数如下：

```python
cv2.getGaussianKernel(ksize,sigma,ktype)
        ksize:奇数，一维核长度
        sigma:标准差
        ktype:数据格式，应该为CV_32F 或者 CV_64F

返回矩阵如下：垂直的矩阵
[[ 0.27406862]
 [ 0.45186276]
 [ 0.27406862]
```

```python
#coding:utf-8

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

#convolve2d只是对单通道进行卷积，若要实现cv.GaussianBlur()多通道高斯卷积，需要拆分三个通道进行，再合并

def gaussianBlur(img,h,w,sigma,boundary="fill",fillvalue=0):
    kernel_x = cv.getGaussianKernel(w,sigma,cv.CV_64F)   #默认得到的为垂直矩阵
    
    kernel_x = np.transpose(kernel_x)  #转置操作，得到水平矩阵
    
    #水平方向卷积
    gaussian_x = signal.convolve2d(img,kernel_x,mode="same",boundary=boundary,fillvalue=fillvalue)
    
    #垂直方向卷积
    kernel_y = cv.getGaussianKernel(h,sigma,cv.CV_64F)
    gaussian_xy = signal.convolve2d(gaussian_x,kernel_y,mode="same",boundary=boundary,fillvalue=fillvalue)
    
    #cv.CV_64F数据转换为uint8
    gaussian_xy = np.round(gaussian_xy)
    gaussian_xy = gaussian_xy.astype(np.uint8)
    
    return gaussian_xy

if __name__=="__main__":
    img = cv.imread(r"C:\Users\Administrator\Desktop\timg.jpg",0)
    img_gauss = gaussianBlur(img,3,3,1)
    cv.imshow("img",img)
    cv.imshow("img_gauss",img_gauss)
    cv.waitKey(0)
    cv.destroyAllWindows()

先水平卷积，再垂直卷积
```

![2019-08-11-8](/img/2019-08-11-8.png)

### 2.2 均值平滑

高斯卷积核，对卷积框中像素值赋予不同权重，而均值平滑赋予相同权重，一个3*5的均值卷积核如下，均值卷积核也是可分离的。

![2019-08-11-9](/img/2019-08-11-9.png)

opencv的boxFilter()函数和blur()函数都能用来进行均值平滑，其参数如下：

```
cv2.boxFilter(src,ddepth,ksize,dst,anchor,normalize,borderType)
        src: 输入图像对象矩阵,
        ddepth:数据格式,位深度
        ksize:高斯卷积核的大小，格式为(宽，高)
        dst:输出图像矩阵,大小和数据类型都与src相同
        anchor：卷积核锚点，默认(-1,-1)表示卷积核的中心位置
        normalize:是否归一化 （若卷积核3*5，归一化卷积核需要除以15）
        borderType:填充边界类型
        
    cv2.blur(src,ksize,dst,anchor,borderType)
        src: 输入图像对象矩阵,可以为单通道或多通道
        ksize:高斯卷积核的大小，格式为(宽，高)
        dst:输出图像矩阵,大小和数据类型都与src相同
        anchor：卷积核锚点，默认(-1,-1)表示卷积核的中心位置
        borderType:填充边界类型
```

示例代码和使用效果如下：

```python
#coding:utf-8

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread(r"C:\Users\Administrator\Desktop\timg.jpg")
img_blur = cv.blur(img,(3,5))

# img_blur = cv.boxFilter(img,-1,(3,5))

cv.imshow("img",img)
cv.imshow("img_blur",img_blur)
cv.waitKey(0)
cv.destroyAllWindows()
```

![2019-08-11-10](/img/2019-08-11-10.png)

### 2.3 中值平滑

中值平滑也有核，但并不进行卷积计算，而是对核中所有像素值排序得到中间值，用该中间值来代替锚点值。opencv中利用medianBlur()来进行中值平滑，中值平滑特别适合用来去除椒盐噪声，其参数如下：

```python
 cv2.medianBlur(src,ksize,dst)
        src: 输入图像对象矩阵,可以为单通道或多通道
        ksize:核的大小，格式为 3      #注意不是（3,3）
        dst:输出图像矩阵,大小和数据类型都与src相同
```

```python
#coding:utf-8

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random

img = cv.imread(r"C:\Users\Administrator\Desktop\timg.jpg")
rows,cols = img.shape[:2]

#加入椒盐噪声
for i in range(100):
    r = random.randint(0,rows-1)
    c = random.randint(0,cols-1)
    img[r,c]=255


img_medianblur = cv.medianBlur(img,5)

cv.imshow("img",img)
cv.imshow("img_medianblur",img_medianblur)
cv.waitKey(0)
cv.destroyAllWindows()
```

![2019-08-11-11](/img/2019-08-11-11.png)

### 2.4 双边滤波

相比于上面几种平滑算法，双边滤波在平滑的同时还能保持图像中物体的轮廓信息。双边滤波在高斯平滑的基础上引入了灰度值相似性权重因子，所以在构建其卷积核核时，要同时考虑空间距离权重和灰度值相似性权重。在进行卷积时，每个位置的邻域内，根据和锚点的距离d构建距离权重模板，根据和锚点灰度值差异r构建灰度值权重模板，结合两个模板生成该位置的卷积核。opencv中的bilateralFilter()函数实现了双边滤波，其参数对应如下：

```python
dst = cv2.bilateralFilter(src,d,sigmaColor,sigmaSpace,borderType)
        src: 输入图像对象矩阵,可以为单通道或多通道
        d:用来计算卷积核的领域直径，如果d<=0，从sigmaSpace计算d
        sigmaColor：颜色空间滤波器标准偏差值，决定多少差值之内的像素会被计算（构建灰度值模板）
        sigmaSpace:坐标空间中滤波器标准偏差值。如果d>0，设置不起作用，否则根据它来计算d值（构建距离权重模板）
```

```python
#coding:utf-8

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
import math

img = cv.imread(r"C:\Users\Administrator\Desktop\timg.jpg")
img_bilateral = cv.bilateralFilter(img,0,0.2,40)

cv.imshow("img",img)
cv.imshow("img_bilateral",img_bilateral)
cv.waitKey(0)
cv.destroyAllWindows()

bilateralFilter
```

![2019-08-11-12](/img/2019-08-11-12.png)

同样，利用numpy也可以自己实现双边滤波算法，同样需要对每个通道进行双边滤波，最后进行合并，下面代码只对单通道进行了双边滤波，代码和效果如下图：

```python
#coding:utf-8

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
import math


def getDistanceWeight(sigmaSpace,H,W):
    r,c = np.mgrid[0:H:1,0:W:1]

    r = r-(H-1)/2
    c =c-(W-1)/2
    distanceWeight = np.exp(-0.5*(np.power(r,2)+np.power(c,2))/math.pow(sigmaSpace,2))
    return distanceWeight

def bilateralFilter(img,H,W,sigmaColor,sigmaSpace):
    distanceWeight = getDistanceWeight(sigmaSpace,H,W)
    cH = (H-1)/2
    cW = (W-1)/2
    rows,cols = img.shape[:2]
    bilateralImg = np.zeros((rows,cols),np.float32)
    for r in range(rows):
        for c in range(cols):
            pixel = img[r,c]
            rTop = 0 if r-cH<0 else r-cH
            rBottom = rows-1 if r+cH>rows-1 else r+cH
            cLeft = 0 if c-cW<0 else c-cW
            cRight = cols-1 if c+cW>cols-1 else c+cW
        
            #权重模板作用区域
            region=img[rTop:rBottom+1,cLeft:cRight+1]
            
            #灰度值差异权重
            colorWeight = np.exp(0.5*np.power(region-pixel,2.0)/math.pow(sigmaColor,2))
            print(colorWeight.shape)
            #距离权重
            distanceWeightTemp = distanceWeight[cH-(r-rTop):rBottom-r+cH+1,cW-(c-cLeft):cRight-c+cW+1]
            print(distanceWeightTemp.shape)
            #权重相乘并归一化
            weightTemp = colorWeight*distanceWeightTemp
            weightTemp = weightTemp/np.sum(weightTemp)
            bilateralImg[r][c]=np.sum(region*weightTemp)
    return bilateralImg

if __name__=="__main__":
    img = cv.imread(r"C:\Users\Administrator\Desktop\timg.jpg",0)
    img_temp = img/255.0
    img_bilateral = bilateralFilter(img_temp,3,3,0.2,19)*255
    img_bilateral[img_bilateral>255] = 255
    img_bilateral = img_bilateral.astype(np.uint8)
    cv.imshow("img",img)
    cv.imshow("img_bilateral",img_bilateral)
    cv.waitKey(0)
    cv.destroyAllWindows()
```

![2019-08-11-13](/img/2019-08-11-13.png)

### 2.5 联合双边滤波

双边滤波是根据原图中不同位置灰度相似性来构建相似性权重模板，而联合滤波是先对原图进行高斯平滑，然后根据平滑后的图像灰度值差异建立相似性模板，再与距离权重模板相乘得到最终的卷积核，最后再对原图进行处理。所以相比于双边滤波，联合双边滤波只是建立灰度值相似性模板的方法不一样。

联合双边滤波作为边缘保留滤波算法时，进行joint的图片即为自身原图片，如果将joint换为其他引导图片，联合双边滤波算法还可以用来实现其他功能。opencv 2中不支持联合双边滤波，opencv 3中除了主模块，还引入了contrib，其中的ximgproc模块包括了联合双边滤波的算法。因此如果需要使用opencv的联合双边滤波，需要安装opencv-contrib-python包。

```python
pip install opencv_python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
```

　联合双边滤波: cv2.xmingproc.jointBilateralFilter(), 其相关参数如下：

```python
dst = cv2.xmingproc.jointBilateralFilter(joint,src,d,sigmaColor,sigmaSpace,borderType)
        joint: 进行联合滤波的导向图像，可以为单通道或多通道，保持边缘的滤波算法时常采用src
        src: 输入图像对象矩阵,可以为单通道或多通道
        d:用来计算卷积核的领域直径，如果d<0，从sigmaSpace计算d
        sigmaColor：颜色空间滤波器标准偏差值，决定多少差值之内的像素会被计算（构建灰度值模板）
        sigmaSpace:坐标空间中滤波器标准偏差值。如果d>0，设置不起作用，否则根据它来计算d值（构建距离权重模板）
```

下面是联合双边滤波的使用代码和效果：（采用src的高斯平滑图片作为joint）

```python
#coding:utf-8
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random
import math

src = cv.imread(r"C:\Users\Administrator\Desktop\timg.jpg")
joint  = cv.GaussianBlur(src,(7,7),1,0)
dst = cv.ximgproc.jointBilateralFilter(joint,src,33,2,0)
# dst = cv.ximgproc.jointBilateralFilter(src,src,33,2,0) #采用src作为joint

cv.imshow("img",src)
cv.imshow("joint",joint)
cv.imshow("dst",dst)
cv.waitKey(0)
cv.destroyAllWindows()

cv2.ximgproc.jointBilateralFilter()
```

![2019-08-11-14](/img/2019-08-11-14.png)

### 2.6  导向滤波

导向滤波也是需要一张图片作为引导图片，来表明边缘，物体等信息，作为保持边缘滤波算法，可以采用自身作为导向图片。opencv 2中也暂不支持导向滤波, 同样在opencv-contrib-python包的ximgproc模块提供了导向滤波函。

导向滤波具体原理可以参考：https://blog.csdn.net/baimafujinji/article/details/74750283

opencv中导向滤波cv2.ximgproc.guidedFilter()的参数如下：

```python
导向滤波
 cv2.ximgproc.guidedFilter(guide,src,radius,eps,dDepth)
        guide: 导向图片，单通道或三通道
        src: 输入图像对象矩阵,可以为单通道或多通道
        radius:用来计算卷积核的领域直径
        eps:规范化参数， eps的平方类似于双边滤波中的sigmaColor（颜色空间滤波器标准偏差值）
            (regularization term of Guided Filter. eps2 is similar to the sigma in the color space into bilateralFilter.)
        dDepth: 输出图片的数据深度
```

其代码使用和效果如下：

```python
#coding:utf-8
import cv2 as cv

src = cv.imread(r"C:\Users\Administrator\Desktop\timg.jpg")
dst = cv.ximgproc.guidedFilter(src,src,33,2,-1)
cv.imshow("img",src)
cv.imshow("dst",dst)
cv.waitKey(0)
cv.destroyAllWindows()

cv2.ximgproc.guidedFilter
```

![2019-08-11-15](/img/2019-08-11-15.png)
