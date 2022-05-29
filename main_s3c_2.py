# S3C: Self-Screening and Self-Correction for noisy label data
# adapt: chaneg EMAE to AE: Average Entropy, to see whether the results will be better than S3C (not S3C_1)

import sys
import os
import os.path
import argparse
import random
import numpy as np
from datetime import datetime
import pandas as pd
from scipy import stats
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from tools import *

# parse arguments
parser = argparse.ArgumentParser(description='Self-Screening and Self-Correction')
parser.add_argument('--recording_file', default='caogao', help='name of recording file')
parser.add_argument('--cuda_visible_devices', default='7')
parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10 or cifar100 or mnist')
parser.add_argument('--split_per', default=0.9, type=float, help='percent of data for training')
# training
# parser.add_argument('--warm_up', type=int, default=3, help='warm up training epochs')
parser.add_argument('--update_times', type=int, default=3, help='times of updating selected training dataset')
parser.add_argument('--n_epoch', type=int, default=6, help='each time training epoch')
parser.add_argument('--threshold_final', type=float, default=0.95, help='threshold, percentage for screening data according to EMAE')  # 这个threshold怎么选是个问题啊
parser.add_argument('--threshold_start', type=float, default=0.7, help='try to adopt a linear time-changing threshold')
parser.add_argument('--seed', type=int, default=123, help='the random seed used in torch related')
# noise
parser.add_argument('--noise_type', type=str, help='pairflip, symmetric, instance', default='symmetric')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.2)
# usually don't change
parser.add_argument('--no_cuda', action='store_true', help='disable CUDA training')
# parser.add_argument('--gamma', type=float, default=0.9, help='coef for EMAE, usually 0.9/0.99/0.999')
parser.add_argument('--lr', default=0.01, type=float,help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='used in train, valid and test')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=1e-4)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices


# prepare for different datasets: mnist, cifar10/100
if args.dataset == 'mnist':
    # mnist的还要检查一下 先不用这个了吧
    args.num_classes = 10
    args.modelName = 'Lenet'
    args.data_path = './public_data/mnist'
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, ))])
    train_val_dataset = MNIST_withpath(root=args.data_path, train=True, download=True) # used for train and validation
    test_dataset = MNIST(root=args.data_path, train=False, transform=transform_test, download=True) # clean test
elif args.dataset == 'cifar10':
    args.num_classes = 10
    args.modelName = 'ResNet18'
    args.data_path = './public_data/cifar10'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_val_dataset = CIFAR10_withpath(root=args.data_path, train=True, download=True)
    test_dataset = CIFAR10(root=args.data_path, train=False, transform=transform_test, download=True)
elif args.dataset == 'cifar100':
    args.num_classes = 100
    args.modelName = 'ResNet34'
    args.data_path = './public_data/cifar100'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))])
    train_val_dataset = CIFAR100_withpath(root=args.data_path, train=True, download=True)
    test_dataset = CIFAR100(root=args.data_path, train=False, transform=transform_test, download=True)


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(BASE_DIR, "results", args.recording_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_save_dir = os.path.join(log_dir, "model")
    model_save_path = model_save_dir+"/" + args.dataset + "-" + args.noise_type + "-" + str(args.noise_rate) + "-seed" + str(args.seed) + "-best.hdf5"
    model_lasttime_save_path = model_save_dir+"/" + args.dataset + "-" + args.noise_type + "-" + str(args.noise_rate) + "-seed" + str(args.seed) + "-best(lasttime).hdf5"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # log_writer = SummaryWriter() # tensorboard
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'), sys.stdout)
    start_time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M')
    print("Training starts at "+ start_time_str + " !!!!")
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # ============ create noisy dataset ============= #
    train_data, val_data, train_noisy_labels, val_noisy_labels, train_clean_labels, val_clean_labels, train_paths, val_paths= \
        dataset_split(np.array(train_val_dataset.data), np.array(train_val_dataset.targets), np.array(train_val_dataset.absolute_paths), args.noise_rate, args.noise_type, args.split_per, args.seed, args.num_classes)
    train_dataset = Train_Dataset(train_data, train_noisy_labels, train_clean_labels, train_paths, transform_train) # 后面要画出train_data里面的图的时候，只要保证这个数据集是不变的就可以了 上面的seed应该可以保证的
    val_dataset = Train_Dataset(val_data, val_noisy_labels, val_clean_labels, val_paths, transform_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    # ========= model, criterion, optimizer ========== #
    model = createModel(args.modelName, args.num_classes)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    # =============== recording files ================ #
    emae_dict = {}  # absolute_path: EMAE
    pred_dict = {} # absolute: predicted label
    recording_file = pd.DataFrame(columns=(['path', 'true_label', 'noisy_label']+['cor_'+str(i+1) for i in range(args.update_times)])) # for analysis after completion
    recording_file_0 = pd.DataFrame(columns=(['path', 'true_label', 'noisy_label']+['nocor_'+str(i+1) for i in range(args.update_times)])) # 记录Ci里的数据 nocor就是改变之前的label
    # loss_rec = {"train": [], "valid": []}
    # acc_rec = {"train": [], "valid": []}
    train_state_epoch = [0 for i in range(args.update_times+1)] # record num of training epochs in all states
    emae_dict, pred_dict, recording_file, recording_file_0 = Initialize(emae_dict, pred_dict, recording_file, recording_file_0, train_loader) # initialize the EMAE dict and recording file
    # warm up early stopping
    n_add = 0
    inner_val_acc = 0

    total_epochs = 0  # 记录至今为止累计训练了多少个epoch 因为是换成AE了，计算的时候需要有这个来辅助的
    # ============== warm-up training ================ #
    # in this time, we use all training data to train the model, and use early stopping to monitor the training process
    for epoch in range(args.n_epoch): # may be different from later training epochs
        loss_avg_train, acc_avg_train, emae_dict = train(model, train_loader, optimizer, device, criterion, emae_dict, total_epochs)
        loss_avg_eval, acc_avg_eval = evaluate(model, val_loader, criterion, device)
        # loss_rec['train'].append(loss_avg_train); loss_rec['valid'].append(loss_avg_eval)
        # acc_rec['train'].append(acc_avg_train); acc_rec['valid'].append(acc_avg_eval)
        # if (epoch+1)%10==0:
        #     draw_line(loss_rec['train'], loss_rec['valid'], log_dir, 'loss')
        #     draw_line(acc_rec['train'], acc_rec['valid'], log_dir, 'acc')
        print("Warm up training: Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f}, LR:{:.4f}".format \
                  (epoch + 1, args.n_epoch ,acc_avg_train, acc_avg_eval, loss_avg_train, loss_avg_eval, optimizer.param_groups[0]['lr']))
        total_epochs+=1
        # early stopping
        if (acc_avg_eval > inner_val_acc + n_add):
            inner_val_acc = acc_avg_eval
            torch.save(model.state_dict(), model_save_path)
            train_state_epoch[0]=epoch+1
            n_add = 0  # validation accuracy still increasing, reset n_add
        else:
            n_add += 0.001 #
        if n_add > 0.02: # 1% parameter
            print("No improved in 20 epochs, stop warm up training!")
            break
    # test the model after warm-up training
    loss_temp, acc_temp = test(model, test_loader, criterion, device)
    print("Testing results after warm-up: Average acc: ", acc_temp, ", Average loss: ", loss_temp)

    # ============= alternative training ============== #
    complete_train_dataset = train_dataset; complete_train_loader = train_loader # complete dataloader
    outer_val_acc = inner_val_acc
    torch.save(model.state_dict(), model_lasttime_save_path)
    early_stop = False
    for update_time in range(args.update_times):
        print("updating training datasets")
        model = createModel(args.modelName, args.num_classes)
        model.load_state_dict(torch.load(model_save_path))

        # test the model after last time of training:
        if update_time>0:
            loss_temp, acc_temp = test(model, test_loader, criterion, device)
            print("Testing results after ", update_time, "th of training: Average acc: ", acc_temp, ", Average loss: ", loss_temp)

        threshold = args.threshold_start+(args.threshold_final-args.threshold_start)/(args.update_times-1)*update_time
        new_train_dataset, unselected_dataset = screen_correct_train_dataset(complete_train_dataset, complete_train_loader, model, threshold, emae_dict, pred_dict, recording_file, recording_file_0, update_time+1, device)
        new_train_loader = DataLoader(dataset=new_train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        unselected_loader = DataLoader(dataset=unselected_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        inner_val_acc = 0 # record the best acc in this time of training
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        scheduler = MultiStepLR(optimizer, milestones=[int((args.n_epoch)/2), int((args.n_epoch)/1.25)])
        for epoch in range(args.n_epoch):
            loss_avg_train, acc_avg_train, emae_dict = train(model, new_train_loader, optimizer, device, criterion, emae_dict, total_epochs)
            loss_avg_eval, acc_avg_eval = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            # loss_rec['train'].append(loss_avg_train); loss_rec['valid'].append(loss_avg_eval)
            # acc_rec['train'].append(acc_avg_train); acc_rec['valid'].append(acc_avg_eval)
            # if (epoch + 1) % 5 == 0:
                # draw_line(loss_rec['train'], loss_rec['valid'], log_dir, 'loss')
                # draw_line(acc_rec['train'], acc_rec['valid'], log_dir, 'acc')
            print("{:1d}th update training: Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{:.4f}".format \
                    (update_time+1, epoch + 1, args.n_epoch, acc_avg_train, acc_avg_eval, loss_avg_train, loss_avg_eval, optimizer.param_groups[0]['lr']))
            # predict step, update unselected data's EMAE
            emae_dict = predict(model, unselected_loader, device, emae_dict, total_epochs)
            total_epochs+=1
            # 这里面其实不是early stopping 是把所有的epoch都跑完 但是下一次选数据继续训练模型的时候load的是这一次在val上表现最好的
            if acc_avg_eval>inner_val_acc:
                inner_val_acc = acc_avg_eval
                torch.save(model.state_dict(), model_save_path)
                train_state_epoch[update_time+1] = epoch+1

        if outer_val_acc>inner_val_acc: # 这一次最好的还比不上上一次最好的: 也就是停止更新数据再去训练了
            print("The model achieves best acc on validation is : ....-best(lasttime) one!")
            early_stop = True
            train_state_epoch[update_time+1]=0
            break # 退出update_times循环
        outer_val_acc = inner_val_acc # set the best acc as this time of training and save corresponding model
        torch.save(model.state_dict(), model_lasttime_save_path)

    # ===================== test ======================= #
    model_final = createModel(args.modelName, args.num_classes)
    if early_stop:
        model_final.load_state_dict(torch.load(model_lasttime_save_path))
    else:
        model_final.load_state_dict(torch.load(model_save_path))
    loss_avg_test, acc_avg_test = test(model_final, test_loader, criterion, device)

    # ============= save files and ending ============== #
    tf = open(os.path.join(log_dir, "EMAEntropy_dict.pkl"), "wb")
    pickle.dump(emae_dict, tf)
    tf.close()
    recording_file.to_csv(os.path.join(log_dir, "Recording.csv"))
    recording_file_0.to_csv(os.path.join(log_dir, "Recording_0.csv"))

    end_time = datetime.now()
    end_time_str = datetime.strftime(end_time, '%m-%d_%H-%M')
    print("Training is Finished at " + end_time_str + " !!!!")
    print("Testing results (model which achieves best val acc): Average acc: ", acc_avg_test, ", Average Loss: ", loss_avg_test)
    print("Finally training epochs of each state: [warm up, update training]", train_state_epoch)
    f = open(os.path.join(log_dir, 'log.txt'), 'a')
    sys.stdout = f
    sys.stderr = f


# def cal_ema(past, gamma, new):
#     new_ema = gamma*past+(1-gamma)*new
#     return new_ema


def screen_correct_train_dataset(train_dataset, train_loader, model, threshold, emae_dict, pred_dict, recording_file, recording_file_0, update_times, device):
    ''' screen and correct data
    '''
    emae_threshold = np.percentile(list(emae_dict.values()), threshold*100) # calculate the threshold of threshold(%) EMAE in emae_dict
    # calculate pred_dict
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            (imgs, targets, clean_labels, paths) = data
            inputs, targets = imgs.cuda(device), targets.cuda(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(paths)):
                path = paths[j]
                pred_dict[path] = predicted[j].item()
    # screen and correct data
    selected_data = []; unselected_data = []
    selected_targets = []; unselected_targets = []
    selected_clean_labels = []; unselected_clean_labels = []
    selected_paths = []; unselected_paths = []
    data1, targets1, clean_labels1, paths1= train_dataset.getData()
    for j in range(len(paths1)):
        path = paths1[j]
        index = list(paths1).index(path)
        if emae_dict[path]<= emae_threshold: # screen
            selected_data.append(data1[index])
            selected_targets.append(pred_dict[path])
            selected_clean_labels.append(clean_labels1[index])
            selected_paths.append(path)
            recording_file.loc[recording_file['path']==path, 'cor_'+str(update_times)] = pred_dict[path] # 要调试就把这一行注释掉
            recording_file_0.loc[recording_file_0['path']==path, 'nocor_'+str(update_times)] = recording_file_0.loc[recording_file_0['path']==path, 'noisy_label']
        else:
            unselected_data.append(data1[index])
            unselected_targets.append(targets1[index])
            unselected_clean_labels.append(clean_labels1[index])
            unselected_paths.append(path)

    # temp_dataset = Train_Dataset(data1, targets1, clean_labels1, paths1, transform=None) # since we want the original not transformed data
    # temp_loader = DataLoader(temp_dataset, batch_size=64, num_workers=8, shuffle=False, pin_memory=True)
    # for i, data in enumerate(temp_loader):
    #     (imgs, targets, clean_labels, paths) = data
    #     for j in range(len(paths)):
    #         path = paths[j]
    #         if emae_dict[path] <= emae_threshold:  # screen
    #             selected_data.append(imgs[j].numpy())
    #             selected_clean_labels.append(clean_labels[j].item())
    #             selected_paths.append(path)
    #             selected_targets.append(pred_dict[path])  # correct
    #             # recording_file.loc[recording_file['path']==path, 'cor_'+str(update_times)] = predicted[j].item() # update recording file
    #         else:
    #             unselected_data.append(imgs[j].numpy())
    #             unselected_clean_labels.append(clean_labels[j].item())
    #             unselected_paths.append(path)
    #             unselected_targets.append(targets[j].item())

    selected_dataset = Train_Dataset(selected_data, selected_targets, selected_clean_labels, selected_paths, transform=transform_train)
    unselected_dataset = Train_Dataset(unselected_data, unselected_targets, unselected_clean_labels, unselected_paths, transform=transform_train)
    return selected_dataset, unselected_dataset


def Initialize(emae_dict, pred_dict, recording_file, recording_file_0, train_loader):
    for i, data in enumerate(train_loader):
        (imgs, targets, clean_labels, absolute_paths) = data
        for j in range(len(targets)):
            path = absolute_paths[j]
            # initialize
            emae_dict[path] = None
            pred_dict[path] = None
            recording_file = recording_file.append([{'path': path, 'true_label': clean_labels[j].item(), 'noisy_label': targets[j].item()}], ignore_index=True) # 要调试就把这一行注释掉
            recording_file_0 = recording_file_0.append([{'path': path, 'true_label': clean_labels[j].item(), 'noisy_label': targets[j].item()}], ignore_index=True)
    return emae_dict, pred_dict, recording_file, recording_file_0


def train(model, train_loader, optimizer, device, criterion, emae_dict, total_epoch):
    loss_train = AverageMeter()
    acc_train = AverageMeter()
    model.train()
    for i, data in enumerate(train_loader):
        (imgs, targets, clean_labels, paths) = data
        inputs, targets = imgs.cuda(device), targets.cuda(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        acc = np.sum((predicted == targets).tolist()) / len(targets)
        outputs_softmax = nn.Softmax(dim=1)(outputs)
        entropys = stats.entropy(outputs_softmax.cpu().detach().numpy(), axis=1)
        loss_train.update(loss)
        acc_train.update(acc)

        # record EMAE to emae_dict
        for j in range(len(targets)):
            path = paths[j]
            if emae_dict[path]==None:
                emae_dict[path] = entropys[j]
            else:
                past_emae = emae_dict[path]
                new_entropy = entropys[j]
                ema_entropy = total_epoch/(total_epoch+1)*past_emae+1/(total_epoch+1)*new_entropy
                # ema_entropy = cal_ema(past_emae, gamma, new_entropy)
                emae_dict[path] = ema_entropy

    return loss_train.avg, acc_train.avg, emae_dict


def evaluate(model, valid_loader, criterion, device):
    loss_valid = AverageMeter()
    acc_valid = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            imgs, targets, _, _ = data
            inputs, targets = imgs.cuda(device), targets.cuda(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            acc = np.sum((predicted == targets).tolist()) / len(targets)
            loss_valid.update(loss)
            acc_valid.update(acc)

    return loss_valid.avg, acc_valid.avg


def predict(model, unselected_loader, device, emae_dict, total_epoch):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(unselected_loader):
            (imgs, targets, clean_labels, paths) = data
            inputs, targets = imgs.cuda(device), targets.cuda(device)
            outputs = model(inputs)
            outputs_softmax = nn.Softmax(dim=1)(outputs)
            entropys = stats.entropy(outputs_softmax.cpu().detach().numpy(), axis=1)
            for j in range(len(targets)):
                path = paths[j]
                past_emae = emae_dict[path]
                new_entropy = entropys[j]
                ema_entropy = total_epoch / (total_epoch + 1) * past_emae + 1 / (total_epoch + 1) * new_entropy
                # ema_entropy = cal_ema(past_emae, gamma, new_entropy)
                emae_dict[path] = ema_entropy
    return emae_dict


def test(model, test_loader, criterion, device):
    loss_test = AverageMeter()
    acc_test = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            imgs, targets  = data
            inputs, targets = imgs.cuda(device), targets.cuda(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            acc = np.sum((predicted == targets).tolist()) / len(targets)
            loss_test.update(loss)
            acc_test.update(acc)
    return loss_test.avg, acc_test.avg


def analysis():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(BASE_DIR, "results", "S3C_cifar10_symmetric0.4_second_try")
    df = pd.read_csv(os.path.join(log_dir,"Recording.csv"))
    df_0 = pd.read_csv(os.path.join(log_dir, "Recording_0.csv"))
    threshold_start = 0.8
    threshold_end = 0.9
    update_times = 10
    label_precision, label_precision_0 = [], []
    length = len(df)
    for i in range(update_times):
        threshold = threshold_start+(threshold_end-threshold_start)/(update_times-1)*i
        label_precision_0.append(np.sum(df_0['true_label'] == df_0['nocor_'+str(i+1)])/(length*threshold))
        label_precision.append(np.sum(df['true_label'] == df['cor_'+str(i+1)])/(length*threshold))
    print("before selection: ", label_precision_0)
    print("after correction: ", label_precision)


# def draw_imgs(paths):
#     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     for i in range(len(paths)):



# def validate_EMAE(recording_file):
#     '''画出没有被选进来的图片
#     '''


if __name__ == '__main__':
    # analysis()
    main()