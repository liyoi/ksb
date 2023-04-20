# 这是一个示例 Python 脚本。

# 将minist数据集中的所有数据、标签提取出来
# 可忽略此模块
#
# # model = torch.nn.Module()
# #
# # torch.nn.Conv2d(15, 12, 0)
# #
# # JPG: str = '11.jpg'
# #
# # sequential = torch.nn.Sequential()
#
# # print(torch.cuda.set_device())
# class Train:
#     def __init__(self):
#         pass
#
#
# # def print_hi(name: str) -> None:
# #     # 在下面的代码行中使用断点来调试脚本。
# #     img_11 = cv2.imread(JPG)
# #     img_11 = cv2.resize(img_11, dsize=(300, 300), fx=0.02, fy=0.02)
# #     cv2.imshow(JPG, img_11)
# #     # cv2.resizeWindow(JPG, 400, 400)
# #     cv2.waitKey(0)
# #     print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。
#
#
# # 按间距中的绿色按钮以运行脚本。
# if __name__ == '__main__':
#     # n = np.random.uniform(-1, 1, 5)
#     # a = range(5)
#     # print(n)
#     # plt.figure(figsize=(14, 8), edgecolor='green')
#     # plt.bar(a, n, edgecolor='g')
#     #
#     # for i in range(10):
#     #     print(i)
#     # print_hi('PyCharm')
#     print(torch.__version__)
#     print(torch.version.cuda)
#     print(torch.cuda.is_available())
#     t = torch.Tensor([1, 1, 2, 2, 3])
#     print(t)
#     print(t.eq(2).sum().float().item())

import os

import torchvision.datasets.mnist as mnist
from skimage import io

root = "F://Python文件//ksb//mnist_data//MNIST//raw"
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
)
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
)
print("training set :", train_set[0].size())
print("test set :", test_set[0].size())


def convert_to_img(train=True):
    if train:
        f = open(root + 'train.txt', 'w')
        data_path = root + '/train/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()
    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            img_path = data_path + str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()


convert_to_img(True)  # 转换训练集
convert_to_img(False)  # 转换测试集
