# create imbalanced datasets

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import copy
import numpy as np

def build_dataset(dataset, num_meta):
    '''
    Meta-Weight-Net的代码 但这里得到的training set还不是imbalance的
    有个问题 这里的meta dataset是不是就是相当于validation set啊
    :param dataset: 'cifar10', 'cifar100'
    :param num_meta:
    :return: train meta data(balanced, validation), train data(balanced), test data(no processing)
    '''
    # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    assert dataset in ['cifar10',  'cifar100'] , 'This dataset is not supported yet'
    CIFAR10_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR10_STD = [0.24703233, 0.24348505, 0.26158768]
    CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR100_STD = [0.2673, 0.2564, 0.2762]
    train_transform_CIFAR10 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_transform_CIFAR10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_transform_CIFAR100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    test_transform_CIFAR100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root='./public_data/cifar10', train=True, download=True, transform=train_transform_CIFAR10)
        test_dataset = torchvision.datasets.CIFAR10('./public_data/cifar10', train=False, transform=test_transform_CIFAR10)
        img_num_list = [num_meta] * 10 # [num_meta, num_meta, ... (10)] 因为meta data要求是balance的 所以num_meta一样
        num_classes = 10

    if dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root='./public_data/cifar100', train=True, download=True, transform=train_transform_CIFAR100)
        test_dataset = torchvision.datasets.CIFAR100('./public_data/cifar10', train=False, transform=test_transform_CIFAR100)
        img_num_list = [num_meta] * 100
        num_classes = 100

    data_list_val = {}
    for j in range(num_classes):
        data_list_val[j] = [i for i, label in enumerate(train_dataset.targets) if label == j] # 得出来的是indexes

    idx_to_meta = []
    idx_to_train = []
    # print(img_num_list)

    # 在每一类中选出一些数据来作为meta data
    for cls_idx, img_id_list in data_list_val.items():
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        idx_to_meta.extend(img_id_list[:img_num])
        idx_to_train.extend(img_id_list[img_num:])
    train_data = copy.deepcopy(train_dataset)
    train_data_meta = copy.deepcopy(train_dataset)
    train_data_meta.data = np.delete(train_dataset.data,idx_to_train,axis=0) # 居然是用np.delete来删除的吗
    train_data_meta.targets = np.delete(train_dataset.targets, idx_to_train, axis=0)
    train_data.data = np.delete(train_dataset.data, idx_to_meta, axis=0)
    train_data.targets = np.delete(train_dataset.targets, idx_to_meta, axis=0)

    return train_data_meta,train_data,test_dataset


def cal_num_per_cls(dataset,imb_factor=1,num_meta=None):
    """这个计算不同label的数据量的函数其实还是有问题的
    比如当imb_factor==1的时候 经过noisy扰动之后有些数据的数目会超过原来的 按这样算会舍弃掉一些数据
    用asymmetric noise的时候 需要有更加合理的做法
    Get a list of image numbers for each class, given cifar version
    Num of images follows exponential distribution
    args:
      cifar_version: str, 'cifar10', 'cifar100'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if getting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    assert dataset in ['cifar10',  'cifar100'] , 'This dataset is not supported yet'
    assert imb_factor<=1 , 'imbalance factor should be in [0,1]'
    if dataset == 'cifar10':
        img_max = (50000-num_meta)/10  # training dataset的构建就保证了最开始的时候每一类的数目都是达到了img_max的
        cls_num = 10

    if dataset == 'cifar100':
        img_max = (50000-num_meta)/100
        cls_num = 100

    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))  # 计算公式
        img_num_per_cls.append(int(num))
    return img_num_per_cls

# 基本可以确定就是这个函数导致的问题了
def build_imbalanced_train_dataset(train_dataset, img_num_list):
    '''
    create imbalanced training dataset, discard some data in the balanced dataset to create imbalanced dataset
    :param dataset_training: the selected training set (besides validation dataset)
    :param img_num_list: a dict, {'class':num_class_imbalanced}
    :return: an imbalanced training dataset
    '''
    # 先通过build_dataset将数据都transform了 就不需要再变化了
    data_list = {}
    num_classes = len(img_num_list)
    for j in range(num_classes):
        data_list[j] = [i for i, label in enumerate(train_dataset.targets) if label == j]
    # img_num_list = cal_num_per_cls(dataset_type, imb_factor, num_meta * num_classes)
    print("Number of training samples of all classes (according to labels 0-9/99):", img_num_list)
    im_data = {}
    idx_to_del = []
    print(img_num_list)
    for cls_idx, img_id_list in data_list.items():
        np.random.shuffle(img_id_list)
        img_num = img_num_list[int(cls_idx)]
        im_data[cls_idx] = img_id_list[img_num:]
        idx_to_del.extend(img_id_list[img_num:])
    print("Number of all training samples after discarding:", sum(img_num_list), ", discard ", len(idx_to_del), "samples")
    imbalanced_train_dataset = copy.deepcopy(train_dataset)
    imbalanced_train_dataset.targets = np.delete(train_dataset.targets, idx_to_del, axis=0)
    imbalanced_train_dataset.true_label = np.delete(train_dataset.true_label, idx_to_del, axis=0) # 加上这一句可以对吗呜呜呜
    imbalanced_train_dataset.data = np.delete(train_dataset.data, idx_to_del, axis=0)
    return imbalanced_train_dataset

if __name__ == '__main__':
    train_data_meta, train_data, test_dataset = build_dataset('cifar10', 10)
    img_num_list = cal_num_per_cls('cifar10', 0.1, 10 * 10)
    imbalanced_train_dataset = build_imbalanced_train_dataset(train_data, img_num_list)
    imbalanced_train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=128, shuffle=True, num_workers = 4, pin_memory=True)
