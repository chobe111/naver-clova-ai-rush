from torch.utils import data
from torchvision import datasets, transforms
import torch
import os
from tqdm import tqdm
from PIL import Image
import sklearn
from typing import List
from sklearn.model_selection import train_test_split
import random


def encode_index(index: int):
    if 0 <= index <= 6:
        return index - 1
    if 6 <= index <= 11:
        return index - 2
    if 11 <= index <= 17:
        return index - 3
    if 17 <= index <= 23:
        return index - 4
    if 23 <= index <= 29:
        return index - 5
    if 29 <= index <= 35:
        return index - 6
    if 35 <= index <= 39:
        return index - 7
    if 39 <= index <= 40:
        return index - 8
    if 40 <= index <= 47:
        return index - 9
    if 47 <= index <= 54:
        return index - 10
    if 54 <= index <= 57:
        return index - 11
    if 57 <= index <= 58:
        return index - 12
    if 58 <= index <= 60:
        return index - 13
    if 60 <= index <= 63:
        return index - 14
    if 63 <= index <= 65:
        return index - 15
    if 65 <= index <= 69:
        return index - 16
    if 69 <= index <= 74:
        return index - 17
    if 74 <= index <= 82:
        return index - 18
    if 82 <= index <= 86:
        return index - 19
    if 86 <= index <= 90:
        return index - 20
    if 90 <= index <= 93:
        return index - 21
    if 93 <= index <= 100:
        return index - 22
    if 100 <= index:
        return index - 23


def pil_loader(path, img_size=224):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except FileNotFoundError as e:
        raise FileNotFoundError(e)


def get_transform(random_crop=True):
    # normalize = transforms.Normalize(
    #     mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    #     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform = []
    transform.append(transforms.Resize(256))
    if random_crop:
        transform.append(transforms.RandomResizedCrop(224))
        transform.append(transforms.RandomHorizontalFlip())
    else:
        transform.append(transforms.CenterCrop(224))
    transform.append(transforms.ToTensor())
    transform.append(normalize)
    return transforms.Compose(transform)


class TestDataset(data.Dataset):
    def __init__(self, root='../1-3-DATA-fin'):

        self.root = root
        self.data_idx = 'data_idx'

        self.sample_dir = 'test'
        self.data_loc = 'test_data'
        self.path = os.path.join(root, self.data_loc, self.data_idx)

        with open(self.path, 'r') as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            idx = line.split("_")[0]
            self.samples.append([line.rstrip('\n'), idx])

        self.transform = get_transform(random_crop=False)

    def __getitem__(self, index):
        '''
        Here, our problem supposes maximum 3 hierarchy
        '''
        path, idx = self.samples[index]
        path = os.path.join(self.root, self.data_loc, path)
        sample = self.transform(pil_loader(path=path))

        return torch.LongTensor([int(idx)]), sample

    def __len__(self):
        return len(self.samples)


class RangeDataset(data.Dataset):
    def __init__(self, data, data_loc, sample_dir, root='../1-3-DATA-fin') -> None:
        lines = data
        self.samples = []
        self.data_loc = data_loc
        self.sample_dir = sample_dir
        random_crop = True
        self.root = root
        for line in lines:
            idx = line.split(" ")[0].split("__")[1].split("_")[0]
            label = [v.rstrip('\n') for v in line.split(' ')[1:]]
            self.samples.append([line.split(' ')[0], label, idx])

        self.transform = get_transform(random_crop=random_crop)

    def __getitem__(self, index):
        path, target, idx = self.samples[index]
        path = os.path.join(self.root, self.sample_dir, self.data_loc, path)
        sample = self.transform(pil_loader(path=path))

        if len(target) == 1:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([encode_index(int(target[0]))]), torch.LongTensor([-1]), torch.LongTensor([-1])
        elif len(target) == 2:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([encode_index(int(target[1]))]), torch.LongTensor([encode_index(int(target[1]))]), torch.LongTensor([-1])
        else:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([encode_index(int(target[2]))]), torch.LongTensor([encode_index(int(target[1]))]), torch.LongTensor([encode_index(int(target[2]))])

    def __len__(self):
        return len(self.samples)


class AccRangeDataset(data.Dataset):
    def __init__(self, data, root='../1-3-DATA-fin') -> None:
        lines = data
        self.samples = []
        random_crop = True
        self.root = root
        for line in lines:
            idx = line.split(" ")[0].split("__")[1].split("_")[0]
            label = [v.rstrip('\n') for v in line.split(' ')[1:]]
            self.samples.append([line.split(' ')[0], label, idx])

        self.transform = get_transform(random_crop=random_crop)

    def __getitem__(self, index):
        path, target, idx = self.samples[index]
        path = os.path.join(self.root, self.sample_dir, self.data_loc, path)
        sample = self.transform(pil_loader(path=path))

        if len(target) == 1:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([encode_index(int(target[0]))]), torch.LongTensor([-1]), torch.LongTensor([-1])
        elif len(target) == 2:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([encode_index(int(target[0]))]), torch.LongTensor([encode_index(int(target[1]))]), torch.LongTensor([-1])
        else:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([encode_index(int(target[0]))]), torch.LongTensor([encode_index(int(target[1]))]), torch.LongTensor([encode_index(int(target[2]))])


class BaggingDataset(data.Dataset):
    def __init__(self, root='../1-3-DATA-fin', split_num: int = 5):
        self.root = root
        self.data_idx = 'data_idx'

        self.sample_dir = 'train'
        self.data_loc = 'train_data'
        self.path = os.path.join(
            root, self.sample_dir, self.data_loc, self.data_idx)

        with open(self.path, 'r') as f:
            lines = f.readlines()

        total_data_len = len(lines)
        self.whole_dataset: List[RangeDataset] = []
        self.whole_acc_dataset: List[AccRangeDataset] = []

        for i in range(split_num):
            self.whole_dataset.append(RangeDataset(
                random.sample(lines, int(total_data_len * 0.95)), data_loc=self.data_loc, sample_dir=self.sample_dir, root=self.root
            ))

    def __call__(self, key='train') -> List[RangeDataset]:
        if key == 'train':
            return self.whole_dataset
        else:
            return self.whole_acc_dataset


class CustomDataset(data.Dataset):
    def __init__(self, is_train=True, root='../1-3-DATA-fin', split=1.0):

        self.root = root
        self.data_idx = 'data_idx'

        self.sample_dir = 'train'
        self.data_loc = 'train_data'
        self.path = os.path.join(
            root, self.sample_dir, self.data_loc, self.data_idx)

        print("Data path = ", self.path)
        with open(self.path, 'r') as f:
            lines = f.readlines()

        split = int(len(lines) * split)
        if is_train:
            random_crop = True
            lines = lines[:split]
        else:
            random_crop = False
            lines = lines[split:]

        self.samples = []
        for line in lines:
            idx = line.split(" ")[0].split("__")[1].split("_")[0]
            label = [v.rstrip('\n') for v in line.split(' ')[1:]]
            self.samples.append([line.split(' ')[0], label, idx])

        self.transform = get_transform(random_crop=random_crop)

    def __getitem__(self, index):
        '''
        Here, our problem supposes maximum 3 hierarchy
        '''
        path, target, idx = self.samples[index]
        path = os.path.join(self.root, self.sample_dir, self.data_loc, path)
        sample = self.transform(pil_loader(path=path))

        if len(target) == 1:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([int(target[0])]), torch.LongTensor([-1]), torch.LongTensor([-1])
        elif len(target) == 2:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([int(target[0])]), torch.LongTensor([int(target[1])]), torch.LongTensor([-1])
        else:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([int(target[0])]), torch.LongTensor([int(target[1])]), torch.LongTensor([int(target[2])])

    def __len__(self):
        return len(self.samples)


class AugumentDataset(CustomDataset):
    def __init__(self, is_train, root, split):
        super().__init__(is_train=is_train, root=root, split=split)

    def __getitem__(self, index):
        path, target, idx = self.samples[index]
        path = os.path.join(self.root, self.sample_dir, self.data_loc, path)

        sample1 = self.transform(pil_loader(path=path))
        sample2 = self.transform(pil_loader(path=path))

        if len(target) == 1:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([encode_index(int(target[0]))]), torch.LongTensor([-1]), torch.LongTensor([-1])
        elif len(target) == 2:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([encode_index(int(target[1]))]), torch.LongTensor([int(target[1])]), torch.LongTensor([-1])
        else:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([encode_index(int(target[2]))]), torch.LongTensor([int(target[1])]), torch.LongTensor([int(target[2])])


class TrainDataset(CustomDataset):
    def __init__(self, is_train, root, split):
        super().__init__(is_train=is_train, root=root, split=split)

    def __getitem__(self, index):
        path, target, idx = self.samples[index]
        path = os.path.join(self.root, self.sample_dir, self.data_loc, path)

        sample = self.transform(pil_loader(path=path))

        if len(target) == 1:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([int(target[0])]), torch.LongTensor([-1]), torch.LongTensor([-1])
        elif len(target) == 2:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([int(target[0])]), torch.LongTensor([int(target[1])]), torch.LongTensor([-1])
        else:
            return torch.LongTensor([int(idx)]), sample, torch.LongTensor([int(target[0])]), torch.LongTensor([int(target[1])]), torch.LongTensor([int(target[2])])


def data_analyze_loader(root, phase='train', batch_size=16, split=1.0):
    is_train = True
    dataset = CustomDataset(is_train=is_train, root=root, split=split)

    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)


def data_loader(root, phase='train', batch_size=16, split=1.0, submit=True):
    if phase == 'train':
        is_train = True
    elif phase == 'test':
        is_train = False
    else:
        raise KeyError

    if submit:
        dataset = TestDataset(root=root)
    else:
        dataset = TrainDataset(is_train=is_train, root=root, split=split)

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=is_train)
