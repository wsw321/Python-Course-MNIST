from models.MySimpleNet import MySimpleNet
from models.ResNet import resnet18, resnet34, resnet50
import torch
from torch import nn
import torch.nn.functional as F
from dataset import train_dataset, test_dataset
from dataset import train_dataloader, test_dataloader
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def train(train_dataloader, device, net, loss_func, optimizer, args):
    # Show Some Info
    print(f'Train Set:{len(train_dataset)}|Img Size:{train_dataset.__getitem__(1)[0].shape}')
    net.train()
    net.to(device)
    path = args.save_pt_dir
    if args.save_results:
        train_loss_list = []
    if not os.path.isdir(path):
        os.mkdir(path)

    results_dir = args.save_results_root_dir + '/' + args.exp
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    best_loss = 10000.0

    for epoch in range(args.epoch):
        for i, (data, label) in enumerate(train_dataloader):
            data = data.to(device)
            label = label.to(device)

            output = net(data)
            train_loss = loss_func(output, label)
            if args.save_results:
                train_loss_list.append(train_loss.item())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if train_loss < best_loss:
                print(f'Epoch:{epoch+1} | Batch:{i} | best train loss:{train_loss}')
                save_path = f'{path}/{args.model}.pt'
                torch.save(net.state_dict(), save_path)
                best_loss = train_loss
    if args.save_results:
        pic_path = results_dir + '/' + 'train_loss.jpg'
        draw_and_save_pic(train_loss_list, pic_path)


def test(test_dataloader, device, net, loss_func, args):
    model_path = f'{args.save_pt_dir}/{args.model}.pt'
    net.load_state_dict(torch.load(model_path))

    results_dir = args.save_results_root_dir + '/' + args.exp
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    net.eval()
    net.to(device)

    test_loss_list = []
    total = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(test_dataloader):
            data = data.to(device)
            label = label.to(device)
            output = net(data)  # batch_size * 10
            test_loss = loss_func(output, label)
            test_loss_list.append(test_loss.item())

            out_label = torch.argmax(output, dim=1)  # get batch_size * 1(index)
            total += (sum(label.eq(out_label)).item())
    if args.save_results:
        pic_path = results_dir + '/' + 'test_loss.jpg'
        txt_path = results_dir + '/' + 'test_acc.txt'
        with open(txt_path, 'w+', encoding='UTF-8') as f:
            f.write(f'平均loss: {np.mean(test_loss_list)}, 测试集准确率: {total / len(test_dataset)}')

        draw_and_save_pic(test_loss_list, pic_path)
    return np.mean(test_loss_list), total / len(test_dataset)


def draw_and_save_pic(loss_list, path):
    fig = plt.figure()
    x_label = list(range(1, len(loss_list)+1))  # x轴的坐标
    plt.plot(x_label, loss_list, color='b', linewidth=1.0)  # 构建折线图，可以设置线宽，颜色属性
    plt.title("loss-iteration")  # 设置标题，这里只能显示英文，中文显示乱码
    plt.ylabel("loss")  # 设置y轴名称
    plt.xlabel("iteration")  # 设置x轴名称
    # plt.show()  # 将图形显示出来
    # TODO:保存图片到目录
    plt.savefig(path)


tra = transforms.Compose([transforms.ToTensor(),
                          transforms.Resize((28, 28)),  # 3*28*28
                          transforms.Grayscale(num_output_channels=1)
                          ])  # 1*28*28


# 给可视化界面的接口函数
def evaluate(model, X):
    model.cuda()
    model.eval()
    X = tra(X)
    X = torch.unsqueeze(X, dim=0)  # 第0维添加batch为1
    X = X.cuda()

    output = model(X)
    output = F.softmax(output, dim=1)
    pred = output.data.max(dim=1)[1]  # 得到最大值下标

    # 得到输出后需要再转成CPU
    output, pred = output.cpu(), pred.cpu()

    c = list(range(0, 10))
    output = list(output.data.numpy().squeeze())
    dic = dict(zip(c, output))
    pred = pred.numpy().squeeze()
    return dic, pred



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='MNIST DEMO')
    parser.add_argument('--epoch', help="epoch to run", default=30)
    parser.add_argument("--save_pt_dir", help="dir to save model.pt", type=str, default='trained_pts')
    parser.add_argument("--lr", help="learning rate", type=float, default=0.0001)
    parser.add_argument("--use_gpu", help="use GPU", type=bool, default=True)
    parser.add_argument("--model", help="which model to use", type=str, default='MySimpleNet')
    parser.add_argument("--save_results", help="display training loss", type=bool, default=True)
    parser.add_argument("--save_results_root_dir", default="vis_results")
    parser.add_argument("--exp", default="...", help="save result to root_dir/exp")
    parser.add_argument("--phase", help="train or test", type=str, default='train')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args.exp = args.model
    if args.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')

    model_dict = {'MySimpleNet': MySimpleNet,
                  'ResNet18': resnet18,
                  'ResNet34': resnet34,
                  'ResNet50': resnet50
                  }
    # TODO:add resnet
    Net = model_dict[args.model]
    net = Net()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.phase == 'train':
        train(train_dataloader, device, net, loss_func, optimizer, args)
    if args.phase == 'test':
        test_loss, test_acc = test(test_dataloader, device, net, loss_func, args)
        print(test_loss)
        print(test_acc)
"""
train: python run.py --model MySimpleNet/ResNet18/...  
test: python run.py --model MySimpleNet/ResNet18/...  --phase test
"""