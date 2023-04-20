import torch

import model

test_net: model.Net = torch.load("output/train_net_5.pth")

test_net.eval()

test_net()
