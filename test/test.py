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

import matplotlib.pyplot as plt
import cv2
import numpy as np



image = cv2.imread('image.png')
image = np.asarray(image, np.float32)
img1 = cv2.resize(image, (500, 500), interpolation=cv2.INTER_LINEAR)
print(img1)




plt.figure()
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(img1)
plt.show()