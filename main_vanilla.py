# class imbalance & noisy label dataset only using resnet18

import os
import time
import argparse
import torch
from datasets import NoisyAndImbalancedDataloader
from datetime import datetime
from torchvision import models
import sys
import numpy as np
import matplotlib.pyplot as plt


# parse arguments
parser = argparse.ArgumentParser(description='vanilla class imbalance')
parser.add_argument('--cuda_visible_devices', default='7')
parser.add_argument('--recording_file', default='caogao', help='name of recording file')
parser.add_argument('--dataset', default='CIFAR10', type=str,help='dataset (CIFAR10 or CIFAR100)')
parser.add_argument('--num_classes', type=int, default=10, help='remember to change along with the dataset')

# create imbalanced & noisy dataset
parser.add_argument('--imb_factor', type=float, default=1)
parser.add_argument('--noise_rate', type=float, default=0.1)
parser.add_argument('--noise_asym', type=bool, default=False)
parser.add_argument('--seed', type=int, default=42, help='the random seed used in creating noisy dataset')

# training
parser.add_argument('--torch_seed', type=int, default=100, help='the random seed used in torch related')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=300, metavar='N')
parser.add_argument('--epoch_point1', type=int, default=150)
parser.add_argument('--epoch_point2', type=int, default=200)
parser.add_argument('--epoch_point3', type=int, default=250)
parser.add_argument('--lr', default=1e-1, type=float,help='initial learning rate')
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--valid_batch_size', type=int, default=64)
parser.add_argument('--momentum', default=0.9, type=float)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices



def main():

    # basic settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # if use_cuda==True:
    #     torch.backends.cudnn.benchmark = True  # 让CNN加速运算
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

    # =========== create imbalanced & noisy datasets =========== #
    dataloaders = NoisyAndImbalancedDataloader(train_batch_size=args.train_batch_size,\
                                               eval_batch_size=args.valid_batch_size,\
                                               dataset_type=args.dataset, seed=args.seed, asym=args.noise_asym,\
                                               noise_rate=args.noise_rate, imb_factor=args.imb_factor).data_loaders
    train_loader = dataloaders['train_dataloader']; valid_loader = dataloaders['valid_dataloader']

    # ========================== MODEL ========================== #
    model = models.resnet18(num_classes=args.num_classes)
    model.to(device)

    # ======================= loss function ===================== #
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # ======================= optimizer ========================= #
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)

    # ======================= train model ======================= #
    best_acc_eval = 0;  best_epoch=0 # 根据acc来选取最佳model

    # ==================== recording matrix ===================== #
    num_training_data = len(train_loader.dataset)
    recording_matrix_pred = np.zeros((num_training_data, args.epochs+3), dtype=int)  # index, true_label, target
    recording_matrix_loss = np.zeros((num_training_data, args.epochs+3), dtype=int)
    for i in range(0, num_training_data):
        recording_matrix_pred[i, 0]=i # 直接i就行 因为在dataset里面indexes就是从0开始的
        recording_matrix_loss[i, 0]=i
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}

    # ========================== training ======================= #
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch + 1, args.epoch_point1, args.epoch_point2, args.epoch_point3)  # 调整learning rate

        loss_avg_train, acc_avg_train = train(train_loader, model, optimizer, criterion, device, recording_matrix_pred, recording_matrix_loss, epoch)
        loss_avg_eval, acc_avg_eval = evaluate(valid_loader, model, criterion, device)
        loss_rec['train'].append(loss_avg_train); loss_rec['valid'].append(loss_avg_eval)
        acc_rec['train'].append(acc_avg_train); acc_rec['valid'].append(acc_avg_eval)

        # save the best model according to acc_avg_eval
        if acc_avg_eval > best_acc_eval: # save the best model
            best_acc_eval = acc_avg_eval; best_epoch = epoch
            # save_model(model.state_dict(), epoch+1, acc_avg_eval, log_dir, best=True)
        # if epoch==args.epochs-1: # save the last model
            # save_model(model.state_dict(), epoch+1, acc_avg_eval, log_dir, best=False)

        # draw acc and loss graph per 10 epochs and save the recording matrix
        if (epoch+1)%1==0:
            draw_line(loss_rec['train'], loss_rec['valid'], log_dir, 'loss')
            draw_line(acc_rec['train'], acc_rec['valid'], log_dir, 'acc')
            np.save(os.path.join(log_dir, 'recording_matrix_pred'), recording_matrix_pred)
            np.save(os.path.join(log_dir, 'recording_matrix_loss'), recording_matrix_loss)
        # print information per epoch
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f}".format\
                  (epoch + 1, args.epochs,acc_avg_train, acc_avg_eval, loss_avg_train, loss_avg_eval))

    # =================================== ending ================================== #
    print('Best validation accuracy: %f in %d epoch' %(best_acc_eval, best_epoch+1))
    end_time = datetime.now()
    end_time_str = datetime.strftime(end_time, '%m-%d_%H-%M')
    print("Training is Finished at " + end_time_str + " ! Yeah")
    f = open(os.path.join(log_dir, 'log.txt'), 'a')
    sys.stdout = f
    sys.stderr = f


def train(train_loader, model, optimizer, criterion, device, recording_train_pred, recording_train_loss, epoch):
    ''' train one epoch of the model
    '''
    loss_train = AverageMeter()
    acc_train = AverageMeter()
    model.train()
    correct_cnts = 0
    for i, data in enumerate(train_loader):
        (inputs, targets, true_labels, indexes) = data
        inputs, targets = inputs.cuda(device), targets.cuda(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        acc=np.sum((predicted==targets).tolist())/len(targets)
        loss_train.update(loss)
        acc_train.update(acc)

        # 计算每个sample的loss
        losses = torch.nn.CrossEntropyLoss(reduction='none')(outputs, targets)

        # recording
        indexes=indexes.numpy()

        for j in range(0, len(indexes)):
            if epoch==0: # record true label and target label
                recording_train_pred[indexes[j], 1]=true_labels[j].item()
                recording_train_loss[indexes[j], 1]=true_labels[j].item()
                recording_train_pred[indexes[j], 2]=targets[j].item()
                recording_train_loss[indexes[j], 2]=targets[j].item()
            recording_train_pred[indexes[j], epoch+3]=predicted[j].item()
            recording_train_loss[indexes[j], epoch+3]=losses[j].item()

    return loss_train.avg, acc_train.avg

def evaluate(valid_loader, model, criterion, device):
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

def save_model(model_state, epoch, acc, path, best):
    if best==True:
        model_saved_path = os.path.join(path, f"best_model.pth")
    if best==False:
        model_saved_path = os.path.join(path, f"last_model.pth")
    state_to_save = {'model_state_dict': model_state, 'auc_dict': acc, 'epoch': epoch}
    torch.save(state_to_save, model_saved_path)

def adjust_learning_rate(optimizer, epoch, epoch_point1, epoch_point2, epoch_point3):
    """Sets the learning rate to the initial LR divided by 5 at epoch_point1, epoch_point2 and epoch_point3"""
    lr = args.lr * ((0.2 ** int(epoch >= epoch_point1)) * (0.2 ** int(epoch >= epoch_point2))* (0.2 ** int(epoch >= epoch_point3)))
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


if __name__ == '__main__':
    main()
    #
    # from datasets import NoisyAndImbalancedDataloader, NoisyDataloader
    # # dataloaders = NoisyDataloader().data_loaders
    # dataloaders = NoisyAndImbalancedDataloader(imb_factor=0.1).data_loaders  # 错在这里
    # traindataloader = dataloaders['train_dataloader']
    # for i, data in enumerate(traindataloader):
    #     (input, target, true_label, index) = data
    #     if i <=3:
    #         print(true_label)
    #         print(target)
    #         print(index)
