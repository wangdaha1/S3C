import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import re

parser = argparse.ArgumentParser(description='analysis')
parser.add_argument('--recorded_file', default='cifar10_symmetric0.3_imb0.1')
parser.add_argument('--epochs', default=300)
parser.add_argument('--num', default=1)
args = parser.parse_args()

def result_plot():
    # 把不同case的图画到一起看 ACC
    with open(r"./results/cifar10_symmetric0.3_imb0.1/log.txt") as f:
        lines = f.readlines()
    results_1 = [s for s in lines if ("Train Acc:" in s)]
    train_acc_1 = [float(re.findall(r"Train Acc:(.+?)%", str)[0]) for str in results_1]
    valid_acc_1 = [float(re.findall(r"Valid Acc:(.+?)%", str)[0]) for str in results_1]
    with open(r"./results/cifar10_symmetric0.3_balance/log.txt") as f:
        lines = f.readlines()
    results_2 = [s for s in lines if ("Train Acc:" in s)]
    train_acc_2 = [float(re.findall(r"Train Acc:(.+?)%", str)[0]) for str in results_2]
    valid_acc_2 = [float(re.findall(r"Valid Acc:(.+?)%", str)[0]) for str in results_2]
    with open(r"./results/cifar10_clean_imb0.1/log.txt") as f:
        lines = f.readlines()
    results_3 = [s for s in lines if ("Train Acc:" in s)]
    train_acc_3 = [float(re.findall(r"Train Acc:(.+?)%", str)[0]) for str in results_3]
    valid_acc_3 = [float(re.findall(r"Valid Acc:(.+?)%", str)[0]) for str in results_3]
    with open(r"./results/cifar10_clean_balance/log.txt") as f:
        lines = f.readlines()
    results_4 = [s for s in lines if ("Train Acc:" in s)]
    train_acc_4 = [float(re.findall(r"Train Acc:(.+?)%", str)[0]) for str in results_4]
    valid_acc_4 = [float(re.findall(r"Valid Acc:(.+?)%", str)[0]) for str in results_4]

    # plot
    epochs = np.arange(1, len(results_1) + 1)
    plt.plot(epochs, train_acc_1, label='noise&imb_train', color='orangered')
    plt.plot(epochs, valid_acc_1, label='noise&imb_valid', color='salmon')
    plt.plot(epochs, train_acc_2, label='noise_train', color='forestgreen')
    plt.plot(epochs, valid_acc_2, label='noise_valid', color='lightgreen')
    plt.plot(epochs, train_acc_3, label='imb_train', color='darkorange')
    plt.plot(epochs, valid_acc_3, label='imb_valid', color='orange')
    plt.plot(epochs, train_acc_4, label='vanilla_train', color='darkblue')
    plt.plot(epochs, valid_acc_4, label='vanilla_valid', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend(loc='lower right', fontsize=8)
    plt.show()

    # 把不同case的图画到一起看看 LOSS
    with open(r"./results/cifar10_symmetric0.3_imb0.1/log.txt") as f:
        lines = f.readlines()
    results_1 = [s for s in lines if ("Train loss:" in s)]
    train_loss_1 = [float(re.findall(r"Train loss:(.+?) ", str)[0]) for str in results_1]
    valid_loss_1 = [float(str[-7:]) for str in results_1]
    with open(r"./results/cifar10_symmetric0.3_balance/log.txt") as f:
        lines = f.readlines()
    results_2 = [s for s in lines if ("Train loss:" in s)]
    train_loss_2 = [float(re.findall(r"Train loss:(.+?) ", str)[0]) for str in results_2]
    valid_loss_2 = [float(str[-7:]) for str in results_2]
    with open(r"./results/cifar10_clean_imb0.1/log.txt") as f:
        lines = f.readlines()
    results_3 = [s for s in lines if ("Train loss:" in s)]
    train_loss_3 = [float(re.findall(r"Train loss:(.+?) ", str)[0]) for str in results_3]
    valid_loss_3 = [float(str[-7:]) for str in results_3]
    with open(r"./results/cifar10_clean_balance/log.txt") as f:
        lines = f.readlines()
    results_4 = [s for s in lines if ("Train loss:" in s)]
    train_loss_4 = [float(re.findall(r"Train loss:(.+?) ", str)[0]) for str in results_4]
    valid_loss_4 = [float(str[-7:]) for str in results_4]

    # plot
    epochs = np.arange(1, len(results_1) + 1)
    plt.plot(epochs, train_loss_1, label='noise&imb_train', color='orangered')
    plt.plot(epochs, valid_loss_1, label='noise&imb_valid', color='salmon')
    plt.plot(epochs, train_loss_2, label='noise_train', color='forestgreen')
    plt.plot(epochs, valid_loss_2, label='noise_valid', color='lightgreen')
    plt.plot(epochs, train_loss_3, label='imb_train', color='darkorange')
    plt.plot(epochs, valid_loss_3, label='imb_valid', color='orange')
    plt.plot(epochs, train_loss_4, label='vanilla_train', color='darkblue')
    plt.plot(epochs, valid_loss_4, label='vanilla_valid', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
# result_plot()

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    saved_dir = os.path.join(BASE_DIR, "results", args.recorded_file)
    recording_matrix_pred = np.load(os.path.join(saved_dir, 'recording_matrix_pred.npy'))
    # recording_matrix_loss = np.load(os.path.join(saved_dir, 'recording_matrix_loss.npy'))
    # print(recording_matrix_loss.shape)
    # print(np.sum(recording_matrix_pred[:,1]==recording_matrix_pred[:,2]))
    # print(recording_matrix_pred[:100,1])
    # print(recording_matrix_pred[:100,2])
    # args.num = np.where((recording_matrix_pred[:, 1] == recording_matrix_pred[:, 2])&(recording_matrix_pred[:,2]==9))[0][1]
    args.num = np.where(recording_matrix_pred[:, 1]!=recording_matrix_pred[:,2])[0][88]
    epoch = np.arange(1, args.epochs+1)
    plt.plot(epoch, recording_matrix_pred[args.num,3:], label="predictions")
    print(recording_matrix_pred[args.num, 3:])
    print(recording_matrix_pred[args.num, 1], recording_matrix_pred[args.num, 2])
    plt.hlines(recording_matrix_pred[args.num,1], 0, 300, colors="green", label="true label")
    plt.hlines(recording_matrix_pred[args.num,2], 0, 300, colors="red", label="target")
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Label')
    plt.show()
