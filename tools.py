
class MiniBatch(object):
    def __init__(self):
        self.ids = []
        self.images = []
        self.labels = []

    def append(self, id, image, label):
        self.ids.append(id)
        self.images.append(image)
        self.labels.append(label)

    def get_size(self):
        return len(self.ids)

class Sample(object):
    def __init__(self, id, image, true_label):
        # image id
        self.id = id
        # image pixels
        self.image = image
        # image true label
        self.true_label = true_label
        # image corrupted label
        self.label = true_label

        ## for logging ###
        self.last_corrected_label = None
        self.corrected = False

    def toString(self):
        return "Id: " + str(self.id) + ", True Label: ", + str(self.true_label) + ", Corrupted Label: " + str(self.label)

def cal_emaentropy(past_ema, gamma, new_entropy):
    new_ema = gamma*past_ema+(1-gamma)*new_entropy
    return new_ema

def cal_emaloss(past_loss, gamma, new_loss):
    new_ema = gamma*past_loss+(1-gamma)*new_loss
    return new_ema

# visualize cifar10/100
import os
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_labels_name(filename):
    """使用pickle反序列化labels文件，得到存储内容
        cifar10的label文件为“batches.meta”，cifar100则为“meta”
        反序列化之后得到字典对象，可根据key取出相应内容
        filename = './public_data/cifar-10-batches-py/batches.meta' if cifar10
                   './public_data/cifar100/cifar-100-python/meta' if cifar100
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def load_data_cifar(filename, mode='cifar10'):
    """ load data and labels information from cifar10 and cifar100
    cifar10 keys(): dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    cifar100 keys(): dict_keys([b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data'])
    """
    with open(filename, 'rb') as f:
        dataset = pickle.load(f, encoding='bytes')
        if mode == 'cifar10':
            data = dataset[b'data']
            labels = dataset[b'labels']
            img_names = dataset[b'filenames']
        elif mode == 'cifar100':
            data = dataset[b'data']
            labels = dataset[b'fine_labels']
            img_names = dataset[b'filenames']
        else:
            print("mode should be in ['cifar10', 'cifar100']")
            return None, None, None

    return data, labels, img_names

def load_cifar10(cifar10_path, mode='train'):
    if mode == "train":
        data_all = np.empty(shape=[0, 3072], dtype=np.uint8)
        labels_all = []
        img_names_all = []
        for i in range(1, 6):
            filename = os.path.join(cifar10_path, 'data_batch_' + str(i)).replace('\\', '/')
            print("Loading {}".format(filename))
            data, labels, img_names = load_data_cifar(filename, mode='cifar10')
            data_all = np.vstack((data_all, data))
            labels_all += labels
            img_names_all += img_names
        return data_all, labels_all, img_names_all
    elif mode == "test":
        filename = os.path.join(cifar10_path, 'test_batch').replace('\\', '/')
        print("Loading {}".format(filename))
        return load_data_cifar(filename, mode='cifar10')


def load_cifar100(cifar100_path, mode='train'):
    if mode == "train":
        filename = os.path.join(cifar100_path, 'train')
        print("Loading {}".format(filename))
        data, labels, img_names = load_data_cifar(filename, mode='cifar100')
    elif mode == "test":
        filename = os.path.join(cifar100_path, 'test')
        print("Loading {}".format(filename))
        data, labels, img_names = load_data_cifar(filename, mode='cifar100')
    else:
        print("mode should be in ['train', 'test']")
        return None, None, None

    return data, labels, img_names


def to_pil(data):
    r = Image.fromarray(data[0])
    g = Image.fromarray(data[1])
    b = Image.fromarray(data[2])
    pil_img = Image.merge('RGB', (r, g, b))
    return pil_img


def random_visualize(imgs, labels, label_names):
    figure = plt.figure(figsize=(len(label_names), 10))
    idxs = list(range(len(imgs)))
    np.random.shuffle(idxs)
    count = [0] * len(label_names)
    for idx in idxs:
        label = labels[idx]
        if count[label] >= 10:
            continue
        if sum(count) > 10 * len(label_names):
            break

        img = to_pil(imgs[idx])
        label_name = label_names[label]

        subplot_idx = count[label] * len(label_names) + label + 1
        print(label, subplot_idx)
        plt.subplot(10, len(label_names), subplot_idx)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        if count[label] == 0:
            plt.title(label_name)

        count[label] += 1

    plt.show()

