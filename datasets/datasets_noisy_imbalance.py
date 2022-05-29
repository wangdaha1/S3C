# to combine the case of class imbalance and noisy label
import sys
from datasets.datasets_class_imbalance import *
from datasets.datasets_noisy_label import *

@mlconfig.register
class NoisyAndImbalancedDataloader():
    # 这里先不分validation set出来了 直接用test调试吧
    # MNIST还没有写class imbalanced的代码
    def __init__(self,
                 train_batch_size=128,
                 eval_batch_size=128,
                 data_path='./public_data',
                 seed=123,
                 num_of_workers=4,
                 asym=False,
                 dataset_type='CIFAR10',
                 cutout_length=16,  # 这个其实没用
                 noise_rate=0.1,
                 imb_factor=1,
                 ):
        self.seed = seed
        np.random.seed(seed)  # 这个是在每次在用np.random.xxx的时候 产生的随机数都相同 在创建noisy dataset的时候用到了
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
        self.imb_factor = imb_factor
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
        #     valid_transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(MEAN, STD)])
        #
        #     # 这里用的train dataset是之前创建的noisy dataset
        #     train_dataset = MNISTNoisy(root=self.data_path,
        #                                train=True,
        #                                transform=train_transform,
        #                                download=True,
        #                                asym=self.asym,
        #                                seed=self.seed,
        #                                noisy_rate=self.noise_rate)
        #
        #     valid_dataset = datasets.MNIST(root=self.data_path,
        #                                   train=False,
        #                                   transform=test_transform,
        #                                   download=True)

        if self.dataset_type == 'CIFAR10':
            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            train_dataset = cifar10Noisy(root=self.data_path,
                                         train=True,
                                         transform=train_transform,
                                         download=True,
                                         asym=self.asym,
                                         noisy_rate=self.noise_rate)

            # 在noisy dataset的基础上加入class imbalance
            img_num_per_cls = cal_num_per_cls('cifar10', self.imb_factor, 0) # calculate number of samples for each class
            imbalanced_noisy_train_dataset = build_imbalanced_train_dataset(train_dataset, img_num_per_cls)

            valid_dataset = datasets.CIFAR10(root=self.data_path,
                                            train=False,
                                            transform=valid_transform,
                                            download=True)

        elif self.dataset_type == 'CIFAR100':
            CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
            CIFAR_STD = [0.2673, 0.2564, 0.2762]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            train_dataset = cifar100Noisy(root=self.data_path,
                                          train=True,
                                          transform=train_transform,
                                          download=True,
                                          asym=self.asym,
                                          seed=self.seed,
                                          noisy_rate=self.noise_rate)

            img_num_per_cls = cal_num_per_cls('cifar100', self.imb_factor, 0)
            imbalanced_noisy_train_dataset = build_imbalanced_train_dataset(train_dataset, img_num_per_cls)

            valid_dataset = datasets.CIFAR100(root=self.data_path,
                                             train=False,
                                             transform=valid_transform,
                                             download=True)

        else:
            raise("Unknown Dataset")

        data_loaders = {}

        data_loaders['train_dataloader'] = DataLoader(dataset=imbalanced_noisy_train_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.num_of_workers)

        data_loaders['valid_dataloader'] = DataLoader(dataset=valid_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_of_workers)

        print("Final information of the training data of each class after class imbalance and noisy label creation:")
        for i in range(10):
            num_each_cls = np.sum(imbalanced_noisy_train_dataset.targets==i)
            print("class %d has %d training samples (labels are noisy)" %(i, num_each_cls))

        return data_loaders


if __name__ == '__main__':
    dataloaders = NoisyAndImbalancedDataloader(dataset_type='CIFAR10', train_batch_size=64, eval_batch_size=64).data_loaders
    traindataloader = dataloaders['train_dataloader']
    print(len(traindataloader.dataset))
    for i, data in enumerate(traindataloader):
        (input, target, true_label, index) = data
        if i == 0:
            print(input)
            print(target)
            print(true_label)
            print(index)
