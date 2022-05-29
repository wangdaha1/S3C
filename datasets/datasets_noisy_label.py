# create noisy datasets, use the code of Active-Passive-Loss

import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from numpy.testing import assert_array_almost_equal
import mlconfig


# When we use some standard datasets such as MNIST, CIFAR10, we have some common sense of which imgs are confusing
# For example, in MNIST, 7 -> 1, 2 -> 7, 5<->6, 3->8
# In CIFAR10, automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
# While in CIFAR100, we dont have. These can be used in creating asymmetric noise.
# different paper adopt different mechanisms of generating noisy labels


# Pair noise transition matrix
def build_for_cifar100(num_classes, noise_rate):
    """
    Create asymmetric noise--pair noise transition matrix.
    """
    assert(noise_rate >= 0.) and (noise_rate <= 1.)

    P = (1. - noise_rate) * np.eye(num_classes)
    for i in np.arange(num_classes - 1):
        P[i, i+1] = noise_rate

    # adjust last row
    P[num_classes-1, 0] = noise_rate

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

# Flip classes according to transition probability matrix P.
def multiclass_noisify(y, P, random_state=0):
    '''
    Flip classes according to transition probability matrix P.
    It expects a number between 0 and the number of classes - 1.
    :param y: original labels
    :param P: Transition matrix
    :param random_state: random seed
    :return: new_y: new labels
    '''

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

# used in symmetric noise flip. The label are randomly flipped into other labels since the probs are all the same
def other_class(num_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param num_classes: number of classes
    :param current_class: the original class index
    :return: one random class index that != class_ind
    """
    if current_class < 0 or current_class >= num_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(num_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

# MNIST noisy dataset
# class MNISTNoisy(datasets.MNIST):
#     def __init__(self, root, train=True, transform=None, target_transform=None, download=True, noisy_rate=0.0, asym=False, seed=0):
#         super(MNISTNoisy, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
#         self.targets = self.targets.numpy()
#         if asym:
#             P = np.eye(10)
#             n = noisy_rate
#             # 7 -> 1
#             P[7, 7], P[7, 1] = 1. - n, n
#             # 2 -> 7
#             P[2, 2], P[2, 7] = 1. - n, n
#             # 5 <-> 6
#             P[5, 5], P[5, 6] = 1. - n, n
#             P[6, 6], P[6, 5] = 1. - n, n
#             # 3 -> 8
#             P[3, 3], P[3, 8] = 1. - n, n
#
#             y_train_noisy = multiclass_noisify(self.targets, P=P, random_state=seed)
#             actual_noise = (y_train_noisy != self.targets).mean()
#             assert actual_noise > 0.0
#             print('Actual noise %.2f' % actual_noise)
#             self.targets = y_train_noisy
#
#         else: # symmetric noise
#             n_samples = len(self.targets)
#             n_noisy = int(noisy_rate * n_samples)
#             print("%d Noisy samples" % (n_noisy))
#             class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
#             class_noisy = int(n_noisy / 10)
#             noisy_idx = []
#             for d in range(10):
#                 noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
#                 noisy_idx.extend(noisy_class_index)
#                 print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
#             for i in noisy_idx:
#                 self.targets[i] = other_class(num_classes=10, current_class=self.targets[i])
#             print(len(noisy_idx))
#
#         print("Print noisy label generation statistics:")
#         for i in range(10):
#             n_noisy = np.sum(np.array(self.targets) == i)
#             print("Noisy class %s, has %s samples." % (i, n_noisy))
#
#         return


# CIFAR10 noisy dataset
class cifar10Noisy(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, noisy_rate=0.0, asym=False):
        super(cifar10Noisy, self).__init__(root, download=download, transform=transform, target_transform=target_transform, train=train)
        self.true_label = copy.deepcopy(self.targets)

        if asym:
            # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            for s, t in zip(source_class, target_class):
                cls_idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(noisy_rate * cls_idx.shape[0])
                noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                for idx in noisy_sample_index:
                    self.targets[idx] = t
            return

        elif noisy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(noisy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(num_classes=10, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return

    def __getitem__(self, index): # 相当于是重写getitem函数 需要再把true label, index都加进来
        original_tuple = super(cifar10Noisy, self).__getitem__(index)
        true_labels = self.true_label[index]
        new_tuple = (original_tuple+(true_labels, )+(index, ))
        return new_tuple


# CIFAR100 noisy dataset
class cifar100Noisy(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, noisy_rate=0.0, asym=False, seed=0):
        super(cifar100Noisy, self).__init__(root, download=download, transform=transform, target_transform=target_transform, train=train)
        self.true_label = copy.deepcopy(self.targets)

        if asym:
            """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
            """
            nb_classes = 100
            P = np.eye(nb_classes)
            n = noisy_rate
            nb_superclasses = 20
            nb_subclasses = 5

            if n > 0.0:
                for i in np.arange(nb_superclasses):
                    init, end = i * nb_subclasses, (i+1) * nb_subclasses
                    P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

                    y_train_noisy = multiclass_noisify(np.array(self.targets), P=P, random_state=seed)
                    actual_noise = (y_train_noisy != np.array(self.targets)).mean()
                assert actual_noise > 0.0
                print('Actual noise %.2f' % actual_noise)
                self.targets = y_train_noisy.tolist()
            return

        elif noisy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(noisy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(100)]
            class_noisy = int(n_noisy / 100)
            noisy_idx = []
            for d in range(100):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(num_classes=100, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(100):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return

    def __getitem__(self, index): # 需要再把true label, index都加进来
        original_tuple = super(cifar100Noisy, self).__getitem__(index)
        true_label = self.true_label[index]
        new_tuple = (original_tuple+(true_label, )+(index, ))
        return new_tuple


@mlconfig.register
class NoisyDataloader():
    '''对于MNIST, cifar10, cifar100的dataloader函数
    '''
    def __init__(self,
                 train_batch_size=128,
                 eval_batch_size=128,
                 data_path='./public_data',
                 seed=123,
                 num_of_workers=4,
                 asym=False,
                 dataset_type='CIFAR10',
                 cutout_length=16,
                 noise_rate=0.1):
        self.seed = seed
        np.random.seed(seed)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_path = data_path
        self.num_of_workers = num_of_workers
        self.cutout_length = cutout_length
        self.noise_rate = noise_rate
        self.dataset_type = dataset_type
        assert self.dataset_type in ['CIFAR10', 'CIFAR100'], \
            'This dataset is not supported for creating noisy&imbalanced dataset yet'
        self.asym = asym
        self.data_loaders = self.loadData()
        return

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        # if self.dataset_type == 'MNIST':
        #     MEAN = [0.1307]
        #     STD = [0.3081]
        #     train_transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(MEAN, STD)])
        #
        #     test_transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(MEAN, STD)])
        #
        #     # 这里用的dataset是之前创建的noisy dataset
        #     train_dataset = MNISTNoisy(root=self.data_path,
        #                                train=True,
        #                                transform=train_transform,
        #                                download=True,
        #                                asym=self.asym,
        #                                seed=self.seed,
        #                                noisy_rate=self.noise_rate)
        #
        #     test_dataset = datasets.MNIST(root=self.data_path,
        #                                   train=False,
        #                                   transform=test_transform,
        #                                   download=True)

        if self.dataset_type == 'CIFAR100':
            CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
            CIFAR_STD = [0.2673, 0.2564, 0.2762]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            train_dataset = cifar100Noisy(root=self.data_path,
                                          train=True,
                                          transform=train_transform,
                                          download=True,
                                          asym=self.asym,
                                          seed=self.seed,
                                          noisy_rate=self.noise_rate)

            test_dataset = datasets.CIFAR100(root=self.data_path,
                                             train=False,
                                             transform=test_transform,
                                             download=True)

        elif self.dataset_type == 'CIFAR10':
            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            train_dataset = cifar10Noisy(root=self.data_path,
                                         train=True,
                                         transform=train_transform,
                                         download=True,
                                         asym=self.asym,
                                         noisy_rate=self.noise_rate)

            test_dataset = datasets.CIFAR10(root=self.data_path,
                                            train=False,
                                            transform=test_transform,
                                            download=True)
        else:
            raise("This Dataset is not supported yet")

        data_loaders = {}

        data_loaders['train_dataloader'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=self.num_of_workers)

        data_loaders['test_dataloader'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_of_workers)

        return data_loaders


if __name__ == '__main__':
    dataloaders = NoisyDataloader(dataset_type='CIFAR10', train_batch_size=64).data_loaders
    traindataloader = dataloaders['train_dataloader']
    for epoch in range(0,3):
        for i, data in enumerate(traindataloader):
            (input, target, true_label, index) = data
            if i == 1:
                print(true_label)
                print(index)

