import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# if torch.cuda.is_available():
#     device = torch.device("cuda")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 模型模块->核心

# 数字识别模型
class Net(nn.Module):
    def __init__(self, isAtoZ=False):
        super(Net, self).__init__()
        # self.f1=nn.Linear()
        self.net = nn.Sequential(
            # (n+2*padding-5)/stride+1 = 28
            nn.Conv2d(1, 28, 5, 1, 2),
            nn.MaxPool2d(2),  # 28*14*14
            nn.Conv2d(28, 14, 5, 1, 2),
            nn.MaxPool2d(2),  # 14*7*7
            # nn.Conv2d(14, 64, (5, 5), 1, 2),
            # nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(14 * 7 * 7, 64),
        )
        if not isAtoZ:
            self.net.add_module("7", nn.Linear(64, 10))
        else:
            self.net.add_module("7", nn.Linear(64, 36))

    def forward(self, xx: Tensor):  # x->图像
        xx = xx.float()
        xx = xx.to(device)
        xx = self.net(xx)
        return xx.to(device)


def pre_deal_img(img, size=(28, 28)) -> Tensor:
    l = []
    if type(img) is list:
        for i in img:
            ll = []
            resize = cv2.resize(i, size)
            ll.append(resize)
            l.append(ll)
    else:
        ll = []
        resize = cv2.resize(img, size)
        ll.append(resize)
        l.append(ll)
    tensor = torch.Tensor(np.array(l))
    return tensor


# 根据模型预测结果
def predict(model: Net, input: Tensor, device):
    input.to(device)
    with torch.no_grad():
        out = model(input)
        _, pre = torch.max(out.data, 1)
        return pre
