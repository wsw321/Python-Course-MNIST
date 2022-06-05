import torch
import torchvision
import cv2

# using torch api to generate dataset
train_dataset = torchvision.datasets.MNIST(root='MNIST',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='MNIST',
                                          train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

# generate dataloader

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=100,
                                               shuffle=True)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=100,
                                              shuffle=True)

if __name__ == '__main__':
    # draw first batch picture to test
    train_img, train_label = next(iter(train_dataloader))
    print(train_label)  # 打印前100个测试集的标签
    img = torchvision.utils.make_grid(train_img, nrow=10)
    img = img.numpy().transpose(1, 2, 0)  # img(CHW)->cv2(HWC)
    cv2.imshow('Examples in MNIST', img)
    cv2.waitKey()
    # do not forget close cv2 window to end this program...

