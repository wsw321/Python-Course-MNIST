import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class MySimpleNet(nn.Module):
    def __init__(self):
        super(MySimpleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=10)
        )

    def forward(self, t):
        out = self.conv1(t)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape(-1, 64 * 7 * 7)
        out = self.fc1(out)
        return out


if __name__ == '__main__':
    """
    测试代码是否能正常输出
    """
    X = np.ones([1, 1, 28, 28],dtype=float)
    model = MySimpleNet()

    model.eval()
    model.cuda()

    X = torch.from_numpy(X.astype(np.float32)).cuda()
    output = model(X)
    output = F.softmax(output,dim=1)
    pred = output.data.max(dim=1)[1]  # 得到最大值下标

    # 得到输出后需要再转成CPU
    output, pred = output.cpu(), pred.cpu()

    c = list(range(0, 10))
    output = list(output.data.numpy().squeeze())
    dic = dict(zip(c, output))
    pred = pred.numpy().squeeze()

    print(dic)
    print(pred)
