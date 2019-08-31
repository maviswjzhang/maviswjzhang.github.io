# import cv2
# import matplotlib.pyplot as plt
#
# image = cv2.imread('image.png')
# gt = cv2.imread('label.png', 0)
#
# image_flip = cv2.flip(image, -1)
# gt_flip = cv2.flip(gt, -1)
#
# plt.figure()
# plt.subplot(121)
# plt.axis('off')
# plt.imshow(image_flip)
# plt.subplot(122)
# plt.axis('off')
# plt.imshow(gt_flip)
# plt.show()

# # 导入库
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# image = cv2.imread('image.png')
# gt = cv2.imread('label.png', 0)
#
# h, w = image.shape[0], image.shape[1]
# crop_scale = np.random.randint(h//2, h) / h
# crop_h = int(crop_scale * h)
# crop_w = int(crop_scale * w)
# h_begin = np.random.randint(0, h - crop_h)
# w_begin = np.random.randint(0, w - crop_w)
#
# image_crop = image[h_begin: h_begin+crop_h, w_begin:w_begin+crop_w, :]
# gt_crop = gt[h_begin: h_begin+crop_h, w_begin:w_begin+crop_w]
#
# # resize函数使用时dsize先是w再是h，正好相反
# image_resize = cv2.resize(image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
# gt_resize = cv2.resize(gt_crop, (w, h), interpolation=cv2.INTER_NEAREST)
#
# plt.figure()
# plt.subplot(121)
# plt.axis('off')
# plt.imshow(image_resize)
# plt.subplot(122)
# plt.axis('off')
# plt.imshow(gt_resize)
# plt.show()

# # 导入库
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
#
# image = cv2.imread('image.png')
# gt = cv2.imread('label.png', 0)
#
#
# h, w = image.shape[0], image.shape[1]
# # 将图像中心设为旋转中心
# center = (w / 2, h / 2)
#
# M = cv2.getRotationMatrix2D(center, 10, scale=1.0)
# image_rotate = cv2.warpAffine(image, M, (w, h))
# gt_rotate = cv2.warpAffine(gt, M, (w, h))
#
# plt.figure()
# plt.subplot(121)
# plt.axis('off')
# plt.imshow(image_rotate)
# plt.subplot(122)
# plt.axis('off')
# plt.imshow(gt_rotate)
# plt.show()


# import matplotlib.pyplot as plt
# import cv2
# from albumentations import (
#     Compose,
#     RandomBrightnessContrast,
#     CLAHE
# )
#
#
#
# image = cv2.imread('ii.png')
# gt = cv2.imread('label.png', 0)
#
#
# h, w = image.shape[0], image.shape[1]
# aug = Compose([CLAHE(clip_limit=4,
#                      tile_grid_size=(8, 8),
#                      p=1)])
#
# augmented = aug(image=image, mask=gt)
# image_CLAHE = augmented['image']
# gt_CLAHE = augmented['mask']
#
# plt.figure()
# plt.subplot(121)
# plt.axis('off')
# plt.imshow(image_CLAHE)
# plt.subplot(122)
# plt.axis('off')
# plt.imshow(image)
# plt.show()


# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
#
#
# def truncated_linear_stretch(image, truncated_value=2, maxout=255, min_out=0):
#     def gray_process(gray, maxout=maxout, minout=min_out):
#         truncated_down = np.percentile(gray, truncated_value)
#         truncated_up = np.percentile(gray, 100 - truncated_value)
#         gray_new = ((maxout - minout) / (truncated_up - truncated_down)) * gray
#         gray_new[gray_new < minout] = minout
#         gray_new[gray_new > maxout] = maxout
#         return np.uint8(gray_new)
#
#     (b, g, r) = cv2.split(image)
#     b = gray_process(b)
#     g = gray_process(g)
#     r = gray_process(r)
#
#     # 合并每一个通道
#     result = cv2.merge((b, g, r))
#     return result
#
# image = cv2.imread('image1.png')
# image_linear2 = truncated_linear_stretch(image)
#
#
# plt.figure()
# plt.subplot(121)
# plt.axis('off')
# plt.imshow(image)
# plt.subplot(122)
# plt.axis('off')
# plt.imshow(image_linear2)
# plt.show()

# mport matplotlib.pyplot as plti
# import cv2
# import numpy as np
#
#
#
# image = cv2.imread('image.png')
# image = np.asarray(image, np.float32)
# img1 = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)
# print(img1)
#
#
#
#
# plt.figure()
# plt.subplot(121)
# plt.imshow(image)
# plt.subplot(122)
# plt.imshow(img1)
# plt.show()


# # import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = cv2.imread('lena512.bmp', 0)[200:400, 200:400]
#
# plt.figure()
# for i, sigma_color in enumerate([10, 100, 200]):
#     for j, sigma_space in enumerate([10, 100, 200]):
#         bf_img = cv2.bilateralFilter(image, 9, sigma_color, sigma_space)
#         plt.subplot(3, 3, i*3+j+1)
#         plt.axis('off')
#         plt.title('sigma_color: %d,sigma_space: %d' % (sigma_color, sigma_space))
#         plt.imshow(bf_img, cmap='gray')
#
#
# plt.show()
#
# import numpy as np
# import cv2
# import sys
# import math
# import matplotlib.pyplot as plt
#
#
# def distance(x, y, i, j):
#     return np.sqrt((x-i)**2 + (y-j)**2)
#
#
# def gaussian(x, sigma):
#     return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))
#
# def bilateral_filter_own(image, diameter, sigma_color, sigma_space):
#     width = image.shape[0]
#     height = image.shape[1]
#     radius = int(diameter / 2)
#     out_image = np.zeros_like(image)
#
#     for row in range(height):
#         for col in range(width):
#             current_pixel_filtered = 0
#             weight_sum = 0  # for normalize
#             for semi_row in range(-radius, radius + 1):
#                 for semi_col in range(-radius, radius + 1):
#                     # calculate the convolution by traversing each close pixel within radius
#                     if row + semi_row >= 0 and row + semi_row < height:
#                         row_offset = row + semi_row
#                     else:
#                         row_offset = 0
#                     if semi_col + col >= 0 and semi_col + col < width:
#                         col_offset = col + semi_col
#                     else:
#                         col_offset = 0
#                     color_weight = gaussian(image[row_offset][col_offset] - image[row][col], sigma_color)
#                     space_weight = gaussian(distance(row_offset, col_offset, row, col), sigma_space)
#                     weight = space_weight * color_weight
#                     current_pixel_filtered += image[row_offset][col_offset] * weight
#                     weight_sum += weight
#
#             current_pixel_filtered = current_pixel_filtered / weight_sum
#             out_image[row, col] = int(round(current_pixel_filtered))
#
#     return out_image
#
#
# if __name__ == "__main__":
#     image = cv2.imread('lena512.bmp', 0)[200:400, 200:400]
#     plt.figure()
#     for i, sigma_color in enumerate([10, 100, 200]):
#         for j, sigma_space in enumerate([10, 100, 200]):
#             bf_img = bilateral_filter_own(image, 9, sigma_color, sigma_space)
#             plt.subplot(3, 3, i*3+j+1)
#             plt.axis('off')
#             plt.title('sigma_color: %d,sigma_space: %d' % (sigma_color, sigma_space))
#             plt.imshow(bf_img, cmap='gray')
#     plt.show()


# import cv2
# import math
# from time import clock
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# class BilateralFilter(object):
#     """ the bilateral filter class here.
#         It can build distance weight table and similarity weight table,
#         load image, and filting it with these two table, then return the filted image.
#         Attributes:
#             factor: the factor of power of e.
#             ds: distance sigma, which denominator of delta in c.
#             rs: range sigma, which denominator of delta in s.
#             c_weight_table: the gaussian weight table of Euclidean distance,
#         which namly c.
#             s_weight_table: the gaussian weight table of The similarity function,
#         which namly s.
#             radius: half length of Gaussian kernel.
#         """
#
#     def __init__(self, diameter, sigma_color, sigma_space):
#         """init the bilateral filter class with the input args"""
#         self.sigma_space = sigma_space
#         self.sigma_color = sigma_color
#         self.space_weight_table = []
#         self.color_weight_table = []
#         self.radius = int(diameter/2)
#
#     def gaussian(self, x, sigma):
#         return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))
#
#     def build_distance_weight_table(self):
#         """bulid the c_weight_table with radius and ds"""
#         for semi_row in range(-self.radius, self.radius + 1):
#             self.space_weight_table.append([])
#             for semi_col in range(-self.radius, self.radius + 1):
#                 # calculate Euclidean distance between center point and close pixels
#                 dis = math.sqrt(semi_row * semi_row + semi_col * semi_col)
#                 space_weight = self.gaussian(dis, self.sigma_space)
#                 self.space_weight_table[semi_row + self.radius].append(space_weight)
#
#     def build_similarity_weight_table(self):
#         """build the s_weight_table with rs"""
#         for i in range(256):  # since the color scope is 0 ~ 255
#             color_weight = self.gaussian(i, self.sigma_color)
#             self.color_weight_table.append(color_weight)
#
#
#     def clamp(self, p):
#         """return RGB color between 0 and 255"""
#         if p < 0:
#             return 0
#         elif p > 255:
#             return 255
#         else:
#             return p
#
#     def bilateral_filter(self, image):
#         """ the bilateral filter method.
#         Args:
#                 image: source image
#         Returns:
#                 dest: destination image after filting.
#         """
#
#         width = image.shape[0]
#         height = image.shape[1]
#         radius = self.radius
#         self.build_distance_weight_table()
#         self.build_similarity_weight_table()
#         out_image = np.zeros_like(image)
#         red_sum = green_sum = blue_sum = 0  # 各通道亮度值加权和
#         cs_sum_red_weight = cs_sum_green_weight = cs_sum_blue_weight = 0  # 各通道权值之和，用来归一化
#         pixel_num = height * width
#
#         for row in range(height):
#             for col in range(width):
#                 # calculate for each pixel
#                 tr = image[row, col, 0]
#                 tg = image[row, col, 1]
#                 tb = image[row, col, 2]
#                 for semi_row in range(-radius, radius + 1):
#                     for semi_col in range(-radius, radius + 1):
#                         # calculate the convolution by traversing each close pixel within radius
#                         if row + semi_row >= 0 and row + semi_row < height:
#                             row_offset = row + semi_row
#                         else:
#                             row_offset = 0
#                         if semi_col + col >= 0 and semi_col + col < width:
#                             col_offset = col + semi_col
#                         else:
#                             col_offset = 0
#                         tr2 = image[row_offset, col_offset, 0]
#                         tg2 = image[row_offset, col_offset, 1]
#                         tb2 = image[row_offset, col_offset, 2]
#
#                         cs_red_weight = (
#                                 self.space_weight_table[semi_row + radius][semi_col + radius]
#                                 * self.color_weight_table[(abs(tr2 - tr))]
#                         )
#                         cs_green_weight = (
#                                 self.space_weight_table[semi_row + radius][semi_col + radius]
#                                 * self.color_weight_table[(abs(tg2 - tg))]
#                         )
#                         cs_blue_weight = (
#                                 self.space_weight_table[semi_row + radius][semi_col + radius]
#                                 * self.color_weight_table[(abs(tb2 - tb))]
#                         )
#
#                         cs_sum_red_weight += cs_red_weight
#                         cs_sum_blue_weight += cs_blue_weight
#                         cs_sum_green_weight += cs_green_weight
#
#                         red_sum += cs_red_weight * float(tr2)
#                         green_sum += cs_green_weight * float(tg2)
#                         blue_sum += cs_blue_weight * float(tb2)
#
#                 # normalization
#                 tr = int(math.floor(red_sum / cs_sum_red_weight))
#                 tg = int(math.floor(green_sum / cs_sum_green_weight))
#                 tb = int(math.floor(blue_sum / cs_sum_blue_weight))
#
#                 out_image[row, col, 0] = self.clamp(tr)
#                 out_image[row, col, 1] = self.clamp(tg)
#                 out_image[row, col, 2] = self.clamp(tb)
#
#
#                 index = row * width + col + 1
#                 percent = float(index) * 100 / pixel_num
#                 time1 = clock()
#                 used_time = time1 - time0
#                 format = "proceseeing %d of %d pixels, finished %.2f%%, used %.2f second."
#                 print(format % (index, pixel_num, percent, used_time))
#
#                 # clean value for next time
#                 red_sum = green_sum = blue_sum = 0
#                 cs_sum_red_weight = cs_sum_blue_weight = cs_sum_green_weight = 0
#
#         return out_image
#
#
# if __name__ == "__main__":
#     image = cv2.imread('lena512.bmp')[200:400, 200:400, :]
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.figure()
#     for i, sigma_color in enumerate([10, 100, 200]):
#         for j, sigma_space in enumerate([10, 100, 200]):
#             global time0
#             time0 = clock()
#             bf = BilateralFilter(9, sigma_color, sigma_space)
#             bf_img = bf.bilateral_filter(image)
#             plt.subplot(3, 3, i*3+j+1)
#             plt.axis('off')
#             plt.title('sigma_color: %d,sigma_space: %d' % (sigma_color, sigma_space))
#             plt.imshow(bf_img)
#     plt.show()

# import cv2,datetime,sys,glob
# import numpy as np
# import  matplotlib.pyplot as plt
# import matplotlib.cm as cm
#
# def double2uint8(I,ratio=1.0):
#     return np.clip(np.round(I*ratio),0,255).astype(np.uint8)
#
# def GetNlmData(I,templateWindowSize=4,searchWindow=9):
#     f=int(templateWindowSize/2)
#     t=int(searchWindow/2)
#     height,width=I.shape[:2]
#     padLength=t+f
#     I2=np.pad(I,padLength,'symmetric')  # 滑动时边界，将边缘对称折叠上去
#     I_=I2[padLength-f:padLength+f+height,padLength-f:padLength+f+width] #注意边界
#
#     res=np.zeros((height,width,templateWindowSize+2,t+t+1,t+t+1))  # 有问题？%这段主要是控制不超出索引值
#     # 其实主要是将各种参数放到一个矩阵中，便于计算
#     for i in range(-t,t+1):  # 大的滑动窗
#         for j in range(-t,t+1):
#             I2_=I2[padLength+i-f:padLength+i+f+height,padLength+j-f:padLength+f+j+width]  # 某个图像块
#             for kk in range(templateWindowSize):  # 计算得到一个高斯核,分布权重
#                 kernel=np.ones((2*kk+1,2*kk+1))
#                 kernel=kernel/kernel.sum()  # 进行归一化
#                 res[:, :, kk, i+t, j+t] = cv2.filter2D((I2_-I_)**2,-1,kernel)[f:f+height,f+width]
#             res[:,:,-2,i+t,j+t]=I2_[f:f+height,f:f+width]-I
#             res[:,:,-1,i+t,j+t]=np.exp(-np.sqrt(i**2+j**2))
#     print(res.max(),res.min())
#     return res
#
# if __name__ == "__main__":
#     image = cv2.imread('lena512.bmp')
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     noise_image = double2uint8(image + np.random.randn(*image.shape) * 20)
#     plt.figure()
#     plt.subplot(131)
#     plt.axis('off')
#     plt.imshow(image)
#     plt.subplot(132)
#     plt.axis('off')
#     plt.imshow(noise_image)
#     out_image = cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10)  # GetNlmData(noise_image.astype(np.double)/255)
#     plt.subplot(133)
#     plt.axis('off')
#     plt.imshow(out_image)
#     plt.show()

# import cv2
# import scipy as sc
# from scipy import ndimage
# import numpy as np
# from matplotlib import pyplot as plt
#
#
# def double2uint8(I, ratio=1.0):
#     return np.clip(np.round(I*ratio),0,255).astype(np.uint8)
#
# def gaussian(l, sig):
#     # Generate array
#     ax = np.arange(-l // 2 + 1., l // 2 + 1.)
#     # Generate 2D matrices by duplicating ax along two axes
#     xx, yy = np.meshgrid(ax, ax)
#     # kernel will be the gaussian over the 2D arrays
#     kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
#     # Normalise the kernel
#     final = kernel / kernel.sum()
#     return final
#
#
# def clamp(p):
#     """return RGB color between 0 and 255"""
#     if p < 0:
#         return 0
#     elif p > 255:
#         return 255
#     else:
#         return p
#
#
# def means_filter(image, h=10, templateWindowSize=7, searchWindow=21):
#     height, width = image.shape[0], image.shape[1]
#     template_radius = int(templateWindowSize / 2)
#     search_radius = int(searchWindow / 2)
#
#     # Padding the image
#     padLength = template_radius + search_radius
#     img = cv2.copyMakeBorder(image, padLength, padLength, padLength, padLength, cv2.BORDER_CONSTANT, value=255)
#
#     # output image
#     out_image = np.zeros((height, width), dtype='float')
#
#     # generate gaussian kernel matrix of 7*7
#     kernel = gaussian(templateWindowSize, 1)
#
#     # Run the non-local means for each pixel
#     for row in range(height):
#         for col in range(width):
#             pad_row = row + padLength
#             pad_col = col + padLength
#             center_patch = img[pad_row - template_radius: pad_row + template_radius + 1, pad_col - template_radius: pad_col + template_radius + 1]
#
#             sum_pixel_value = 0
#             sum_weight = 0
#
#             # Apply Gaussian weighted square distance between patches of 7*7 in a window of 21*21
#             for r in range(pad_row - search_radius, pad_row + search_radius):
#                 for c in range(pad_col - search_radius, pad_col + search_radius):
#                     other_patch = img[r - template_radius: r + template_radius + 1, c - template_radius: c + template_radius + 1]
#                     diff = center_patch - other_patch
#                     distance_2 = np.multiply(diff, diff)
#                     pixel_weight = np.sum(np.multiply(kernel, distance_2))
#
#                     pixel_weight = np.exp(pixel_weight / (h**2))
#                     sum_weight = sum_weight + pixel_weight
#                     sum_pixel_value = sum_pixel_value + pixel_weight * img[r, c]
#
#             out_image[row, col] = clamp(int(sum_pixel_value / sum_weight))
#     return out_image
#
# # Call means_filter for the input image
# if __name__ == "__main__":
#     image = cv2.imread('lena512.bmp', 0)
#     noise_image = double2uint8(image + np.random.randn(*image.shape) * 20)
#     mean_image = means_filter(image, h=10, templateWindowSize=7, searchWindow=5)
#     plt.figure()
#     plt.subplot(131)
#     plt.axis('off')
#     plt.imshow(image, cmap='gray')
#     plt.subplot(132)
#     plt.axis('off')
#     plt.imshow(noise_image, cmap='gray')
#     plt.subplot(133)
#     plt.axis('off')
#     plt.imshow(mean_image, cmap='gray')
#     plt.show()


# import numpy as np
# import math
# import scipy.signal, scipy.interpolate
# import matplotlib.pyplot as plt
# import cv2
#
#
#
# def bilateral_approximation(image, sigmaS, sigmaR, samplingS=None, samplingR=None):
#     # It is derived from Jiawen Chen's matlab implementation
#     # The original papers and matlab code are available at http://people.csail.mit.edu/sparis/bf/
#
#     # --------------- 原始分辨率 --------------- #
#     inputHeight = image.shape[0]
#     inputWidth = image.shape[1]
#     sigmaS = sigmaS
#     sigmaR = sigmaR
#     samplingS = sigmaS if (samplingS is None) else samplingS
#     samplingR = sigmaR if (samplingR is None) else samplingR
#     edgeMax = np.amax(image)
#     edgeMin = np.amin(image)
#     edgeDelta = edgeMax - edgeMin
#
#
#     # --------------- 下采样 --------------- #
#     derivedSigmaS = sigmaS / samplingS
#     derivedSigmaR = sigmaR / samplingR
#
#     paddingXY = math.floor(2 * derivedSigmaS) + 1
#     paddingZ = math.floor(2 * derivedSigmaR) + 1
#
#     downsampledWidth = int(round((inputWidth - 1) / samplingS) + 1 + 2 * paddingXY)
#     downsampledHeight = int(round((inputHeight - 1) / samplingS) + 1 + 2 * paddingXY)
#     downsampledDepth = int(round(edgeDelta / samplingR) + 1 + 2 * paddingZ)
#
#     wi = np.zeros((downsampledHeight, downsampledWidth, downsampledDepth))
#     w = np.zeros((downsampledHeight, downsampledWidth, downsampledDepth))
#
#     # 下采样索引
#     (ygrid, xgrid) = np.meshgrid(range(inputWidth), range(inputHeight))
#
#     dimx = np.around(xgrid / samplingS) + paddingXY
#     dimy = np.around(ygrid / samplingS) + paddingXY
#     dimz = np.around((image - edgeMin) / samplingR) + paddingZ
#
#     flat_image = image.flatten()
#     flatx = dimx.flatten()
#     flaty = dimy.flatten()
#     flatz = dimz.flatten()
#
#     # 盒式滤波器（平均下采样）
#     for k in range(dimz.size):
#         image_k = flat_image[k]
#         dimx_k = int(flatx[k])
#         dimy_k = int(flaty[k])
#         dimz_k = int(flatz[k])
#
#         wi[dimx_k, dimy_k, dimz_k] += image_k
#         w[dimx_k, dimy_k, dimz_k] += 1
#
#
#     # ---------------  三维卷积 --------------- #
#     # 生成卷积核
#     kernelWidth = 2 * derivedSigmaS + 1
#     kernelHeight = kernelWidth
#     kernelDepth = 2 * derivedSigmaR + 1
#
#     halfKernelWidth = math.floor(kernelWidth / 2)
#     halfKernelHeight = math.floor(kernelHeight / 2)
#     halfKernelDepth = math.floor(kernelDepth / 2)
#
#     (gridX, gridY, gridZ) = np.meshgrid(range(int(kernelWidth)), range(int(kernelHeight)), range(int(kernelDepth)))
#     # 平移，使得中心为0
#     gridX -= halfKernelWidth
#     gridY -= halfKernelHeight
#     gridZ -= halfKernelDepth
#     gridRSquared = ((gridX * gridX + gridY * gridY) / (derivedSigmaS * derivedSigmaS)) + \
#                    ((gridZ * gridZ) / (derivedSigmaR * derivedSigmaR))
#     kernel = np.exp(-0.5 * gridRSquared)
#
#     # 卷积
#     blurredGridData = scipy.signal.fftconvolve(wi, kernel, mode='same')
#     blurredGridWeights = scipy.signal.fftconvolve(w, kernel, mode='same')
#
#     # ---------------  divide --------------- #
#     blurredGridWeights = np.where(blurredGridWeights == 0, -2, blurredGridWeights)  # avoid divide by 0, won't read there anyway
#     normalizedBlurredGrid = blurredGridData / blurredGridWeights
#     normalizedBlurredGrid = np.where(blurredGridWeights < -1, 0, normalizedBlurredGrid)  # put 0s where it's undefined
#
#
#     # --------------- 上采样 --------------- #
#     (ygrid, xgrid) = np.meshgrid(range(inputWidth), range(inputHeight))
#
#     # 上采样索引
#     dimx = (xgrid / samplingS) + paddingXY
#     dimy = (ygrid / samplingS) + paddingXY
#     dimz = (image - edgeMin) / samplingR + paddingZ
#
#     out_image = scipy.interpolate.interpn((range(normalizedBlurredGrid.shape[0]),
#                                            range(normalizedBlurredGrid.shape[1]),
#                                            range(normalizedBlurredGrid.shape[2])),
#                                           normalizedBlurredGrid,
#                                           (dimx, dimy, dimz))
#     return out_image
#
#
# if __name__ == "__main__":
#     image = cv2.imread('lena512.bmp', 0)
#     mean_image = bilateral_approximation(image, sigmaS=64, sigmaR=32, samplingS=32, samplingR=16)
#     plt.figure()
#     plt.subplot(121)
#     plt.axis('off')
#     plt.imshow(image, cmap='gray')
#     plt.subplot(122)
#     plt.axis('off')
#     plt.imshow(mean_image, cmap='gray')
#     plt.show()


#
# import numpy as np
# import cv2
# import math
# import matplotlib.pyplot as plt
#
#
# def distance(x, y, i, j):
#     return np.sqrt((x-i)**2 + (y-j)**2)
#
#
# def gaussian(x, sigma):
#     return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))
#
#
# def joint_bilateral_filter(image, reference_image, diameter, sigma_color, sigma_space):
#     assert image.shape == reference_image.shape
#     width = image.shape[0]
#     height = image.shape[1]
#     radius = int(diameter / 2)
#     out_image = np.zeros_like(image)
#
#     print('===============START=================')
#     for row in range(height):
#         for col in range(width):
#             current_pixel_filtered = 0
#             weight_sum = 0  # for normalize
#             for semi_row in range(-radius, radius + 1):
#                 for semi_col in range(-radius, radius + 1):
#                     # calculate the convolution by traversing each close pixel within radius
#                     if row + semi_row >= 0 and row + semi_row < height:
#                         row_offset = row + semi_row
#                     else:
#                         row_offset = 0
#                     if semi_col + col >= 0 and semi_col + col < width:
#                         col_offset = col + semi_col
#                     else:
#                         col_offset = 0
#                     color_weight = gaussian(reference_image[row_offset][col_offset] - reference_image[row][col], sigma_color)
#                     space_weight = gaussian(distance(row_offset, col_offset, row, col), sigma_space)
#                     weight = space_weight * color_weight
#                     current_pixel_filtered += image[row_offset][col_offset] * weight
#                     weight_sum += weight
#
#             current_pixel_filtered = current_pixel_filtered / weight_sum
#             out_image[row, col] = int(round(current_pixel_filtered))
#     print('===============FINISH=================')
#     return out_image
#
#
#
# if __name__ == "__main__":
#     image = cv2.imread('lena512.bmp', 0)[200:400, 200:400]
#     blur_img = cv2.resize(image, (25, 25))
#     blur_img = cv2.resize(blur_img, (200, 200))
#     plt.figure()
#     plt.subplot(121)
#     plt.axis('off')
#     plt.title('blur image')
#     plt.imshow(blur_img, cmap='gray')
#     plt.subplot(122)
#     plt.axis('off')
#     plt.title('original image')
#     plt.imshow(image, cmap='gray')
#     plt.show()
#     plt.figure()
#     for i, sigma_color in enumerate([10, 100, 200]):
#         for j, sigma_space in enumerate([10, 100, 200]):
#             bf_img = joint_bilateral_filter(blur_img, image, 9, sigma_color, sigma_space)
#             plt.subplot(3, 3, i*3+j+1)
#             plt.axis('off')
#             plt.title('sigma_color: %d,sigma_space: %d' % (sigma_color, sigma_space))
#             plt.imshow(bf_img, cmap='gray')
#     plt.show()

# import numpy as np
# import cv2
# import math
# import matplotlib.pyplot as plt
#
# if __name__ == "__main__":
#     image = cv2.imread('lena512.bmp', 0)[200:400, 200:400]
#     blur_img = cv2.resize(image, (25, 25))
#     blur_img = cv2.resize(blur_img, (200, 200))
#     plt.imshow(blur_img)
#     plt.show()
#     plt.figure()
#     for i, sigma_color in enumerate([10, 100, 200]):
#         for j, sigma_space in enumerate([10, 100, 200]):
#             bf_img = cv2.ximgproc.jointBilateralFilter(blur_img, image, 9, sigma_color, sigma_space)
#             plt.subplot(3, 3, i*3+j+1)
#             plt.axis('off')
#             plt.title('sigma_color: %d,sigma_space: %d' % (sigma_color, sigma_space))
#             plt.imshow(bf_img, cmap='gray')
#     plt.show()


import numpy as np
import matplotlib.pyplot as plt
import cv2


class GuidedFilter:
    """
    References:
        K.He, J.Sun, and X.Tang. Guided Image Filtering. TPAMI'12.
    """
    def __init__(self, I, radius, eps):
        """
        Parameters
        ----------
        I: NDArray
            Guided image or guided feature map
        radius: int
            Radius of filter
        eps: float
            Value controlling sharpness
        """
        if len(I.shape) == 2:
            self._Filter = GrayGuidedFilter(I, radius, eps)

    def filter(self, p):
        """
        Parameters
        ----------
        p: NDArray
            Filtering input which is 2D or 3D with format
            HW or HWC
        Returns
        -------
        ret: NDArray
            Filtering output whose shape is same with input
        """
        p = (1.0 / 255.0) * np.float32(p)
        if len(p.shape) == 2:
            return self._Filter.filter(p)


class GrayGuidedFilter:
    """
    Specific guided filter for gray guided image.
    """
    def __init__(self, I, radius, eps):
        """
        Parameters
        ----------
        I: NDArray
            2D guided image
        radius: int
            Radius of filter
        eps: float
            Value controlling sharpness
        """
        self.I = (1.0 / 255.0) * np.float32(I)
        self.radius = radius * 2 + 1
        self.eps = eps

    def filter(self, p):
        """
        Parameters
        ----------
        p: NDArray
            Filtering input of 2D
        Returns
        -------
        q: NDArray
            Filtering output of 2D
        """
        # step 1
        meanI  = cv2.blur(self.I, (self.radius, self.radius))
        meanp  = cv2.blur(p, (self.radius, self.radius))
        corrI  = cv2.blur(self.I * self.I, (self.radius, self.radius))
        corrIp = cv2.blur(self.I * p, (self.radius, self.radius))
        # step 2
        varI   = corrI - meanI * meanI
        covIp  = corrIp - meanI * meanp
        # step 3
        a      = covIp / (varI + self.eps)
        b      = meanp - a * meanI
        # step 4
        meana  = cv2.blur(a, (self.radius, self.radius))
        meanb  = cv2.blur(b, (self.radius, self.radius))
        # step 5
        q = meana * self.I + meanb

        return q


def double2uint8(I, ratio=1.0):
    return np.clip(np.round(I * ratio), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    image = cv2.imread('lena512.bmp', 0)
    plt.figure()
    for i, radius in enumerate([2, 4, 8]):
        for j, e in enumerate([0.1**2, 0.2**2, 0.4**2]):
            GF = GuidedFilter(image, radius, e)
            plt.subplot(3, 3, i*3+j+1)
            plt.axis('off')
            plt.title('radius: %d, epsilon: %.2f' % (radius, e))
            plt.imshow(GF.filter(image), cmap='gray')
    plt.show()

