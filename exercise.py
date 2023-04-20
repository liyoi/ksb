# 测试学习模块 ,可忽略

import cv2
# import numpy as np
# img2 = cv2.imread("data/car3.jpg")
# src = img2
# hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)#BGR转HSV
# low_hsv = np.array([90,85,50])#这里要根据HSV表对应
# high_hsv = np.array([140,255,255])#这里填入三个max值
# mask = cv2.inRange(hsv, lowerb=low_hsv , upperb=high_hsv)#提取掩膜
#
# #黑色背景转透明部分
# mask_contrary = mask.copy()
# mask_contrary[mask_contrary==0]=1
# mask_contrary[mask_contrary==255]=0#把黑色背景转白色
# mask_bool = mask_contrary.astype(bool)
# mask_img = cv2.add(src, np.zeros(np.shape(src), dtype=np.uint8), mask=mask)
# #这个是把掩模图和原图进行叠加，获得原图上掩模图位置的区域
# mask_img=cv2.cvtColor(mask_img,cv2.COLOR_BGR2BGRA)
# mask_img[mask_bool]=[0,0,0,0]
# #这里如果背景本身就是白色，可以不需要这个操作，或者不需要转成透明背景就不需要这里的操作
# cv2.imshow("image",cv2.resize(mask_img, (400, 400)))

# [m, n, d] = img2.shape
# print(img2.shape)
# B = 118
# G = 100
# R = 23
# level = 10
# print(img2[68][42][2])
# for i in range(m):
#     for j in range(n):
#         # B-R > 15 && B-G > 15
#         if img2[i, j, 0] > 118 and img2[i, j, 1] < 100 and img2[i, j, 2] < 23:
#             img2[i, j, 2] = 255
#             img2[i, j, 1] = 255
#             img2[i, j, 0] = 255
#         else:
#             img2[i, j, 2] = 0
#             img2[i, j, 1] = 0
#             img2[i, j, 0] = 0
# cv2.imshow("img2", cv2.resize(img2, (400, 400)))
img = cv2.imread('G:\Program Files\Image\\bilibili.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("G:\Program Files\Image\\bilibili.png",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
