import cv2
import torch

import model


AZ_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
          'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z']

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
    res=[]
    if type(urls) is list:
        size=urls.__len__()
        for i in range(size):
            print(predict[i].item())
            res.append(AZ_labels[int(predict[i].item())])
    else:
        res.append(predict.item())
    print("预测结果",res)


# test_model(["mnist_data/MNIST/raw/test/8.jpg"])
test_AZ_model(["data/testset_28_28/W/W_1.jpg","data/testset_28_28/9/9_1.jpg","data/testset_28_28/D/D_1.jpg"])
