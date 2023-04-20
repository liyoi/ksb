import cv2
import torch

import model


# 预测模块

# 预测 0-9数字
def test_model(urls=None):
    test_net = torch.load('output/train_net_5.pth')
    imgs = []
    if type(urls) is list:
        for url in urls:
            view = cv2.imread(url, cv2.IMREAD_GRAYSCALE)
            imgs.append(view)
    elif urls is not None:
        view = cv2.imread(urls, cv2.IMREAD_GRAYSCALE)
        imgs.append(view)
    else:
        print("请输入需要识别图片的地址")
        return
    imgs_tensor = model.pre_deal_img(imgs, (28, 28))
    print(imgs_tensor.size())
    imgs_tensor = imgs_tensor.float()
    test_net.to(model.device)
    predict = model.predict(test_net, imgs_tensor, model.device)
    print("预测结果", predict)


# 预测英文或数字字符
def test_AZ_model(urls=None):
    test_net = torch.load('output/Az/train_net_5.pth')
    imgs = []
    if type(urls) is list:
        for url in urls:
            view = cv2.imread(url, cv2.IMREAD_GRAYSCALE)
            imgs.append(view)
    elif urls is not None:
        view = cv2.imread(urls, cv2.IMREAD_GRAYSCALE)
        imgs.append(view)
    else:
        print("请输入需要识别图片的地址")
        return
    imgs_tensor = model.pre_deal_img(imgs, (28, 28))
    print(imgs_tensor.size())
    imgs_tensor = imgs_tensor.float()
    test_net.to(model.device)
    predict = model.predict(test_net, imgs_tensor, model.device)
    print("预测结果", predict)


test_model(["mnist_data/MNIST/raw/test/8.jpg"])
test_AZ_model(["data/testset_28_28/W/W_1.jpg"])
