"""
用于测试模型输入输出
"""

from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from models.MySimpleNet import MySimpleNet
import torch
import torch.nn.functional as F

img = Image.open('test.jpg')
tra = transforms.Compose([transforms.ToTensor(),
                         transforms.Resize((28,28)),
                         transforms.Grayscale(num_output_channels=1)])
img = tra(img)
print(img.shape)
img = torch.unsqueeze(img, dim=0)
print(img.shape)
img = img.cuda()

m = MySimpleNet().cuda()
m.load_state_dict(torch.load('trained_pts/MySimpleNet.pt'))


output = m(img)
output = F.softmax(output, dim=1)
pred = output.data.max(dim=1)[1]  # 得到最大值下标

# 得到输出后需要再转成CPU才能参与下一步计算
output, pred = output.cpu(), pred.cpu()

c = list(range(0, 10))
output = list(output.data.numpy().squeeze())
dic = dict(zip(c, output))
pred = pred.numpy().squeeze()
