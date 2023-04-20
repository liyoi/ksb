import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim
import torch.utils.data as data
import torchvision
from torch import Tensor
#设备
device = torch.device("cpu")
#设置默认设备
torch.set_default_device(device=device)


def plot_image(img, label, name):
    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap="gray", interpolation="none")
        plt.title("{} {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    pass


def one_hot(label, deepth=10):
    out1 = torch.zeros(label.size(0), deepth)
    tensor = torch.LongTensor(label)
    idx = tensor.view(-1, 1)
    out1.scatter_(dim=1, index=idx, value=1)
    return out1


batch_size = 512

mnist = torchvision.datasets.MNIST('/mnist_data', train=True, download=True, transform=torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.1307,), (0.3081,))
     ])
                                   )
# generator = torch.Generator(device='cuda')
train_loader = data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True
                              )
test_loader = data.DataLoader(dataset=torchvision.datasets.MNIST('mnist_data/', train=True, download=True,
                                                                 transform=torchvision.transforms.Compose(
                                                                     [torchvision.transforms.ToTensor(),
                                                                      torchvision.transforms.Normalize((0.1307,),
                                                                                                       (0.3081,))
                                                                      ])), batch_size=batch_size, shuffle=True
                              )
train_loader_data = train_loader


# 创建一个网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.f1=nn.Linear()
        self.f1 = nn.Linear(28 * 28, 256)
        self.f2 = nn.Linear(256, 64)
        self.f3 = nn.Linear(64, 10)
        self.cuda()
        self.to(device)

    def forward(self, xx):  # x->图像
        xx = fn.selu(self.f1(xx.to(device)).to(device)).to(device)
        xx = fn.selu(self.f2(xx).to(device)).to(device)
        xx = self.f3(xx.to(device)).to(device)
        return xx


train_net = Net()

optimizer_sgd = torch.optim.SGD(train_net.parameters(), lr=0.01, momentum=0.9)

train_losses = []
print(next(train_net.parameters()).device)
for epoch in range(3):
    for batch_idx, (a, b) in enumerate(train_loader):
        a = a.view(a.size(0), 28 * 28).to(device)
        out = train_net(a).to(device)
        b_hot = one_hot(b).to(device)
        loss = nn.functional.mse_loss(out, b_hot).to(device)  #
        optimizer_sgd.zero_grad()  # 清零梯度
        loss.backward()
        optimizer_sgd.step()  # 计算梯度
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

        train_losses.append(loss.data)

total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = train_net(x)
    pred: Tensor = out.argmax(dim=1).to(device)
    pred = pred.eq(y.to(device)).to(device)
    pred = pred.sum().to(device)
    correct = pred.float().to(device).item()
    total_correct += correct

total_sum = len(test_loader.dataset)
acc = total_correct / total_sum
print('test acc:', acc)

x, y = next(iter(test_loader))
out = train_net(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')
# if __name__ == '__main__':
#
#
#     print(train_loader_data.batch_size)
#     x, y = next(iter(train_loader))
#     print(x[0][0] * 0.3081 + 0.1307)
#     plt.imshow(x[0][0] * 0.3081 + 0.1307, cmap="gray", interpolation="none")
#     plt.show()
#     cv2.imshow("img", x[0][0] * 0.3081 + 0.1307)
#     pass
