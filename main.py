# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# model = torch.nn.Module()
#
# torch.nn.Conv2d(15, 12, 0)
#
# JPG: str = '11.jpg'
#
# sequential = torch.nn.Sequential()


class Train:
    def __init__(self):
        pass


# def print_hi(name: str) -> None:
#     # 在下面的代码行中使用断点来调试脚本。
#     img_11 = cv2.imread(JPG)
#     img_11 = cv2.resize(img_11, dsize=(300, 300), fx=0.02, fy=0.02)
#     cv2.imshow(JPG, img_11)
#     # cv2.resizeWindow(JPG, 400, 400)
#     cv2.waitKey(0)
#     print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # n = np.random.uniform(-1, 1, 5)
    # a = range(5)
    # print(n)
    # plt.figure(figsize=(14, 8), edgecolor='green')
    # plt.bar(a, n, edgecolor='g')
    #
    # for i in range(10):
    #     print(i)
    # print_hi('PyCharm')
    t = torch.Tensor([1, 1, 2, 2,3])
    print(t)
    print(t.eq(2).sum().float().item())