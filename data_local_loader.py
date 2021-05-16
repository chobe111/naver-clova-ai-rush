from torch.utils import data
from torchvision import datasets, transforms
import torch
import os
from tqdm import tqdm
from PIL import Image

def pil_loader(path, img_size=224):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except FileNotFoundError as e:
        raise FileNotFoundError(e)


def get_transform(random_crop=True):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
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

class CustomDataset(data.Dataset):
    def __init__(self, is_train=True, root='../1-3-DATA-fin', split=1.0):

        self.root = root
        self.data_idx = 'data_idx'

        self.sample_dir = 'train'
        self.data_loc = 'train_data'
        self.path = os.path.join(root, self.sample_dir, self.data_loc, self.data_idx)


        with open(self.path , 'r') as f:
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
        dataset = CustomDataset(is_train=is_train, root=root, split=split)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=is_train)


