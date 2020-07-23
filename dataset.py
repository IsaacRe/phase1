import os.path as path
from torch.utils.data.dataset import Dataset
from torchvision.datasets import CIFAR100, CIFAR10  #, ImageNet
from torchvision.transforms import Resize, Compose, RandomCrop, RandomHorizontalFlip
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader
import torch
import numpy as np
from PIL import Image
DATASETS = {'cifar10': CIFAR10, 'cifar100': CIFAR100}


def get_dataset_cifar(data_dir='../data', num_classes=100, train=False, download=False):
    dataset = 'cifar10' if num_classes == 10 else 'cifar100'
    ExtendedDataset = extend_dataset(dataset)
    dataset = ExtendedDataset(data_dir, num_classes=num_classes, train=train, download=download)
    return dataset


def get_dataloader_cifar(batch_size=100, data_dir='../data', num_classes=100, train=True, shuffle=True,
                         download=False):
    dataset = get_dataset_cifar(data_dir=data_dir, num_classes=num_classes, train=train,
                                download=download)
    if not train:
        shuffle = False
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def extend_dataset(base_dataset):
    assert base_dataset in DATASETS, "Dataset '%s' is not in list of accepted datasets for extension" % base_dataset

    class ExtendedDataset(DATASETS[base_dataset]):

        def __init__(self, *args, num_classes=100, keep_index=True, img_size=224, train=True, **kwargs):
            super(ExtendedDataset, self).__init__(*args, train=train, **kwargs)
            self.train = train
            self.num_classes = num_classes
            self.data_index = None
            self.keep_index = keep_index
            if keep_index:
                self.data_index = np.arange(len(self))
            self.mean_image = None
            self.resize = Resize(img_size)
            self.augment = Compose([RandomCrop(img_size, padding=8), RandomHorizontalFlip()])
            self._set_mean_image()
            self._set_data()

        def __getitem__(self, index):
            x_arr, y = self.data[index], self.targets[index]

            # convert to PIL Image
            img = Image.fromarray(x_arr)

            # resize image
            img = self.resize(img)

            # apply data augmentation
            if self.train:
                img = self.augment(img)

            # convert to tensor
            x = to_tensor(img)

            if self.keep_index:
                index = self.data_index[index]

            return index, x, y

        def _set_mean_image(self):
            mean_image_path = '%s/%s_mean_image.npy' % (self.root, base_dataset)
            if path.exists(mean_image_path):
                mean_image = np.load(mean_image_path)
            else:
                mean_image = np.mean(self.data, axis=0)
                np.save(mean_image_path, mean_image)
            self.mean_image = mean_image.astype(np.uint8)
            self.data -= self.mean_image[None]

        def _get_data_mask(self, label_arr):
            mask = label_arr == 0
            for i in range(1, self.num_classes):
                mask = np.logical_or(mask, label_arr == i)
            return mask

        def _set_data(self):
            label_arr = np.array(self.targets)
            mask = self._get_data_mask(label_arr)
            self.data = self.data[mask]
            self.targets = list(label_arr[mask])
            if self.keep_index:
                self.data_index = self.data_index[mask]

    return ExtendedDataset
