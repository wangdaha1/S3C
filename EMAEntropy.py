# validate Exponential Moving Average Entropy on cifar10 and mnist
# use resnet18, run 80 epochs to see the EMAE distribution and corresponding images


import os
import argparse
import torch
# from datasets import NoisyAndImbalancedDataloader
from datetime import datetime

import torchvision.datasets
from torchvision import models
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import torch.nn as nn
from scipy import stats
import torchvision.transforms as transforms
import torch.utils.data.dataloader

from tools import *

# parse arguments
parser = argparse.ArgumentParser(description='to validate that EMA entropy is reasonable')
parser.add_argument('--cuda_visible_devices', default='7')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--torch_seed', type=int, default=100, help='the random seed used in torch related')
parser.add_argument('--recording_file', default='validate_EMA_try5', help='name of recording file')
parser.add_argument('--dataset', default='CIFAR10', type=str,help='CIFAR10 or MNIST')
parser.add_argument('--num_classes', type=int, default=10, help='remember to change along with the dataset')

# training
parser.add_argument('--epochs', type=int, default=80, metavar='N')
parser.add_argument('--lr', default=1e-1, type=float,help='initial learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='exponential moving average coefficient')
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--valid_batch_size', type=int, default=64)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--adjust_lr', action='store_true')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices


class New_CIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        original_tuple = super(New_CIFAR10, self).__getitem__(index)
        tuple_with_path = (original_tuple + (index,))
        return tuple_with_path

class New_MNIST(torchvision.datasets.MNIST):
    def __getitem__(self, index):
        original_tuple = super(New_MNIST, self).__getitem__(index)
        tuple_with_path = (original_tuple + (index,))
        return tuple_with_path

def cal_ema(past, gamma, new):
    new_ema = gamma*past+(1-gamma)*new
    return new_ema


def main():
    # basic settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.torch_seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(args.torch_seed)  # 为当前GPU设置随机种子

    # record the results
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(BASE_DIR, "results", args.recording_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'), sys.stdout)
    start_time = datetime.now()
    start_time_str = datetime.strftime(start_time, '%m-%d_%H-%M')
    print(start_time_str)
    print(args)

    # ====================== create dataset =================== #
    cifar10_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])])
    cifar10_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])])
    mnist_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
    ]) # 这样做是为了能够直接用resnet18

    assert args.dataset in ['CIFAR10', 'MNIST']
    if args.dataset=='CIFAR10':
        trainset = New_CIFAR10(root='./public_data/cifar10', train=True, download=False, transform=cifar10_train_transform)
        testset = torchvision.datasets.CIFAR10(root='./public_data/cifar10', train=False, download=False, transform=cifar10_test_transform)
    elif args.dataset=='MNIST':
        trainset = New_MNIST(root = './public_data/mnist', train=True, download=True, transform=mnist_transform)
        testset = torchvision.datasets.MNIST(root = './public_data/mnist', train=False, download=True, transform=mnist_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    validloader = torch.utils.data.DataLoader(testset, batch_size=args.valid_batch_size, shuffle=False, num_workers=4)

    # ========================== MODEL ========================== #
    model = models.resnet18(num_classes=args.num_classes)
    model.to(device)

    # ======================= loss function ===================== #
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # ======================= optimizer ========================= #
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)

    # ========================== training ======================= #
    entropy_dict={}  # record the emaentropy consistently
    loss_dict={}   #  record the emaloss consistently
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    for epoch in range(args.epochs):
        if args.adjust_lr==True:
            adjust_learning_rate(optimizer, epoch + 1, args.epochs)  # 调整learning rate
        loss_avg_train, acc_avg_train, entropy_dict, loss_dict = train(trainloader, model, criterion, optimizer, device, entropy_dict, loss_dict, args.gamma)
        loss_avg_eval, acc_avg_eval = evaluate(validloader, model, criterion, device)
        loss_rec['train'].append(loss_avg_train); loss_rec['valid'].append(loss_avg_eval)
        acc_rec['train'].append(acc_avg_train); acc_rec['valid'].append(acc_avg_eval)

        # draw acc and loss graph per 10 epochs and save the recording matrix
        if (epoch+1)%5==0:
            draw_line(loss_rec['train'], loss_rec['valid'], log_dir, 'loss')
            draw_line(acc_rec['train'], acc_rec['valid'], log_dir, 'acc')
        # print information per epoch
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f}".format\
                  (epoch + 1, args.epochs,acc_avg_train, acc_avg_eval, loss_avg_train, loss_avg_eval))

    # save entropy_dict and loss_dict
    tf = open(os.path.join(log_dir, "EMAEntropy_dict.pkl"), "wb")
    pickle.dump(entropy_dict, tf)
    tf.close()
    tf1 = open(os.path.join(log_dir, "EMALoss_dict.pkl"), "wb")
    pickle.dump(loss_dict, tf1)
    tf1.close()

    # =================================== ending ================================== #
    end_time = datetime.now()
    end_time_str = datetime.strftime(end_time, '%m-%d_%H-%M')
    print("Training is Finished at " + end_time_str + " ! Yeah")
    f = open(os.path.join(log_dir, 'log.txt'), 'a')
    sys.stdout = f
    sys.stderr = f


def train(train_loader, model,criterion,  optimizer, device, dict_entropy, dict_loss, gamma):
    ''' train one epoch of the model,
    '''
    loss_train = AverageMeter()
    acc_train = AverageMeter()
    model.train()
    for i, data in enumerate(train_loader):
        (inputs, targets, indexes) = data
        inputs, targets = inputs.cuda(device), targets.cuda(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        losses = torch.nn.CrossEntropyLoss(reduction='none')(outputs, targets)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        acc=np.sum((predicted==targets).tolist())/len(targets)
        outputs_softmax = nn.Softmax(dim=1)(outputs)
        entropys =stats.entropy(outputs_softmax.cpu().detach().numpy(), axis=1)
        loss_train.update(loss)
        acc_train.update(acc)
        # calculate and record emaentropy and emaloss
        indexes=indexes.numpy()
        for i in range(len(indexes)):
            index = indexes[i]
            if index not in dict_entropy.keys():
                dict_entropy[index]=entropys[i]
                dict_loss[index]=losses[i].item()
            else:
                past_entropy = dict_entropy[index]
                new_entropy = entropys[i]
                ema_entropy = cal_ema(past_entropy, gamma, new_entropy)
                past_loss = dict_loss[index]
                new_loss = losses[i].item()
                ema_loss = cal_ema(past_loss, gamma, new_loss)
                dict_entropy[index] = ema_entropy
                dict_loss[index] = ema_loss

    return loss_train.avg, acc_train.avg, dict_entropy, dict_loss

def evaluate(valid_loader, model,criterion,  device):
    '''evaluate the model after one epoch of training
    '''
    loss_valid = AverageMeter()
    acc_valid = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            inputs, targets = data
            inputs, targets = inputs.cuda(device), targets.cuda(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            acc = np.sum((predicted == targets).tolist()) / len(targets)
            loss_valid.update(loss)
            acc_valid.update(acc)

    return loss_valid.avg, acc_valid.avg

class Logger(object):
    """
    save the output of the console into a log file
    """
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# def save_model(model_state, epoch, acc, path, best):
#     if best==True:
#         model_saved_path = os.path.join(path, f"best_model.pth")
#     if best==False:
#         model_saved_path = os.path.join(path, f"last_model.pth")
#     state_to_save = {'model_state_dict': model_state, 'auc_dict': acc, 'epoch': epoch}
#     torch.save(state_to_save, model_saved_path)

def adjust_learning_rate(optimizer, epoch, epoch_total):
    """Sets the learning rate to the initial LR divided by 5 at epoch_point1, epoch_point2 and epoch_point3"""
    lr = args.lr * ((0.2 ** int(epoch >= epoch_total/2)) * (0.2 ** int(epoch >= epoch_total/1.5))* (0.2 ** int(epoch >= epoch_total/1.2)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def draw_line(train, valid, out_dir, mode):
    '''绘制训练和验证集的loss曲线/acc曲线
    '''
    assert mode in ['loss', 'acc']
    plt_x = np.arange(1, len(train)+1)
    plt.plot(plt_x, train, label='Train')
    plt.plot(plt_x, valid, label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel(mode)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()

# def to_pil(data):
#     r = Image.fromarray(data[0])
#     g = Image.fromarray(data[1])
#     b = Image.fromarray(data[2])
#     pil_img = Image.merge('RGB', (r,g,b))
#     return pil_img
#
# def visualize(data, label, label_names):
#     # img = to_pil(data)
#     label_name = label_names[label]
#     plt.imshow(data)
#     plt.title(label_name)

def validate_EMAE():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(BASE_DIR, "results", args.recording_file)
    EMAEntropy_dict = pickle.load(open(os.path.join(log_dir, 'EMAEntropy_dict.pkl'), 'rb'))
    EMALoss_dict = pickle.load(open(os.path.join(log_dir, 'EMALoss_dict.pkl'), 'rb'))

    # draw the distributions of EMAE and EMAL
    entropys = list(EMAEntropy_dict.values())
    losses = list(EMALoss_dict.values())
    plt.subplot(1,2,1)
    plt.hist(entropys, bins=80)
    plt.title("EMAEntropy distribution")
    plt.subplot(1,2,2)
    plt.hist(losses, bins=80)
    plt.title("EMALoss distribution")
    plt.savefig(os.path.join(log_dir, 'distributions.jpg'))

    # show some images with different value of EMAE
    # 画出distributions的最小的100个图片 中间的100个 最大的100个图片（还是从前后1000个里面随机选100个？）并且标上label
    # entropy
    EMAEntropy_sorted = sorted(EMAEntropy_dict.items(), key=lambda a: a[1]) # from small to large
    length_data = len(EMAEntropy_sorted)
    EMAE_sorted_indexes = [EMAEntropy_sorted[i][0] for i in range(length_data)]
    EMAE_easy100_indexes = EMAE_sorted_indexes[:100]
    EMAE_hard100_indexes = EMAE_sorted_indexes[-100:]
    EMAE_mid100_indexes = EMAE_sorted_indexes[int(length_data/2)-50:int(length_data/2)+50]

    # show pics according to indexes
    if args.dataset=='CIFAR10':
        classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        trainset = New_CIFAR10(root='./public_data/cifar10', train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
        # 每张图片的index是在这里面就定好了的 一一对应 然后每次用CIFAR10函数读取的时候图片对应的index也是一直不会变的（只要数据集不变）  后面的dataloader只是把index去打乱然后去分配batch罢了
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
        # easy100
        count=0; plt.figure(figsize=(10, 10))
        for i, data in enumerate(trainloader):
            (inputs, targets, indexes) = data
            indexes = indexes.numpy()
            for j in range(len(indexes)):
                if indexes[j] in EMAE_easy100_indexes:
                    count+=1
                    plt.subplot(10, 10, count)
                    plt.imshow(inputs[j].numpy().transpose(1, 2, 0))
                    plt.title(str(classes[targets[j].item()]),fontdict = {'fontsize' : 4})
                    plt.xticks([])
                    plt.yticks([])
        plt.savefig(os.path.join(log_dir, 'EMAEntropy_easy100.jpg'))

        # middle100
        count = 0;plt.figure(figsize=(10, 10))
        for i, data in enumerate(trainloader):
            (inputs, targets, indexes) = data
            indexes = indexes.numpy()
            for j in range(len(indexes)):
                if indexes[j] in EMAE_mid100_indexes:
                    count += 1
                    plt.subplot(10, 10, count)
                    plt.imshow(inputs[j].numpy().transpose(1, 2, 0))
                    plt.title(str(classes[targets[j].item()]), fontdict={'fontsize': 4})
                    plt.xticks([])
                    plt.yticks([])
        plt.savefig(os.path.join(log_dir, 'EMAEntropy_mid100.jpg'))

        # hard100
        count = 0;plt.figure(figsize=(10, 10))
        for i, data in enumerate(trainloader):
            (inputs, targets, indexes) = data
            indexes = indexes.numpy()
            for j in range(len(indexes)):
                if indexes[j] in EMAE_hard100_indexes:
                    count += 1
                    plt.subplot(10, 10, count)
                    plt.imshow(inputs[j].numpy().transpose(1, 2, 0))
                    plt.title(str(classes[targets[j].item()]), fontdict={'fontsize': 4})
                    plt.xticks([])
                    plt.yticks([])
        plt.savefig(os.path.join(log_dir, 'EMAEntropy_hard100.jpg'))

    else:
        return 0


if __name__ == '__main__':
    validate_EMAE()
    # main()

