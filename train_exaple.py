import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim
import torch.utils.data as data
import torchvision
from torch import Generator
from torch.utils.tensorboard import SummaryWriter

import model

# 训练模块

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device=device)

batch_size = 512

writer = SummaryWriter("./logs/train_1")


def plot_image(img, label, name):
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0], cmap="gray", interpolation="none")
        plt.title("{} {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    pass


# mnist训练集
mnist = torchvision.datasets.MNIST('/mnist_data', train=True, download=True, transform=torchvision.transforms.Compose(
    [torchvision.transforms.Resize([28, 28]),
     torchvision.transforms.ToTensor()])
                                   )
# mnist测试集
mnist_test = torchvision.datasets.MNIST('/mnist_data', train=False, download=True,
                                        transform=torchvision.transforms.Compose(
                                            [torchvision.transforms.Resize([28, 28]),
                                             torchvision.transforms.ToTensor()])
                                        )
# mnist训练数据
train_loader = data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True,
                               generator=Generator(device=device)
                               )
# mnist测试数据
test_loader = data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=True,
                              generator=Generator(device=device))

# 创建一个网络
train_net = model.Net().to(device)
# 优化器
optimizer_sgd = torch.optim.SGD(train_net.parameters(), lr=0.01, momentum=0.9)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 训练轮数
epochs = 5
#
total_train_step = 0
# 记录测试的步长
total_test_step = 0

train_losses = []
print(next(train_net.parameters()).device)
a = 0
for epoch in range(epochs):
    train_net.train()
    print(f"第{epoch}轮训练开始了")
    for data in train_loader:
        imgs, labels = data
        if a == 0:
            a += 1
            print(imgs.size())
        imgs = imgs.to(device)  # 图片数据
        labels = labels.to(device)  # 标签数据
        output = train_net(imgs)
        loss = loss_fn(output, labels)
        optimizer_sgd.zero_grad()
        loss.backward()
        optimizer_sgd.step()
        total_train_step += 1
        if total_train_step % 10 == 0:
            print(f"训练次数{total_train_step} Loss:{loss.item()}")
        # a = a.view(a.size(0), 28 * 28).to(device)
        # out = train_net(a).to(device)
        # b_hot = one_hot(b).to(device)
        # loss = nn.functional.mse_loss(out, b_hot).to(device)  #
        # optimizer_sgd.zero_grad()  # 清零梯度
        # loss.backward()
        # optimizer_sgd.step()  # 计算梯度
        # if batch_idx % 10 == 0:
        #     print( batch_idx, loss.item())
        # train_losses.append(loss.data)

    train_net.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            output = train_net(imgs)
            loss = loss_fn(output, labels)
            total_test_loss += loss
    print(f"整体测试集的Loss: {total_test_loss}")
    total_test_step += 1
    torch.save(train_net, f"output/train_net_{epoch + 1}.pth")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)

writer.close()
# total_correct = 0
# for x, y in test_loader:
#     x = x.view(x.size(0), 28 * 28)
#     out = train_net(x)
#     pred: Tensor = out.argmax(dim=1).to(device)
#     pred = pred.eq(y.to(device)).to(device)
#     pred = pred.sum().to(device)
#     correct = pred.float().to(device).item()
#     total_correct += correct

# total_sum = len(test_loader.dataset)
# acc = total_correct / total_sum
# print('test acc:', acc)

# x, y = next(iter(test_loader))
# out = train_net(x.view(x.size(0), 28 * 28))
# pred = out.argmax(dim=1)
# plot_image(x, pred, 'test')
