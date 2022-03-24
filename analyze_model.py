import os
import math
import datetime

import numpy as np

import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse

from data_local_loader import data_loader, data_analyze_loader

from tqdm import tqdm

from typing import List
import timm
from pprint import pprint
from utils import AverageMeter
from ptflops import get_model_complexity_info


try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML

except:
    IS_ON_NSML = False
    DATASET_PATH = '../1-3-DATA-fin'

data_name = ['food', 'pasta', 'salad', 'pizza', 'burger', 'steak', 'drink', 'tea', 'coffee', 'juice', 'alchol', 'koreanfood', 'bibimbab', 'meat', 'gook', 'side', 'noodle', 'dessert', 'bread', 'cake', 'macaroon', 'icecream', 'fruit', 'animal', 'dog', 'cat', 'hamster', 'Hedgehog', 'parrot', 'bird', 'penguine', 'ostrich', 'swan', 'chicken', 'duck', 'reptile', 'snake', 'turtle', 'lizard', 'nature', 'beach', 'see', 'cape', 'seaside', 'sky', 'sunsetsunrize', 'nightview', 'snowscene', 'snow', 'mountain', 'lakeRiver', 'grassland', 'fall', 'forest', 'plant', 'base',
             'flower', 'anniversary', 'birthday', 'present', 'wedding', 'tucksido', 'weddingdress', 'graduation', 'bachelorcap', 'christmas', 'christmassocks', 'christmastree', 'santa', 'objects', 'creditcard', 'passport', 'driverlicense', 'citizencard', 'toy', 'doll', 'robot', 'lego', 'book', 'magazine', 'comics', 'library', 'site', 'zoo', 'aquarium', 'pool', 'playground', 'buffet', 'restaurant', 'caffe', 'fashion', 'hat', 'shoes', 'excersize', 'soccer', 'baseball', 'basketball', 'tennis', 'scuba', 'ski', 'board', 'car', 'plane', 'bike', 'autobike', 'ship', 'train']
case = [[0, -1, -1],
        [0, 1, -1],
        [0, 2, -1],
        [0, 3, -1],
        [0, 4, -1],
        [0, 5, -1],
        [0, 6, -1],
        [0, 6, 7],
        [0, 6, 8],
        [0, 6, 9],
        [0, 6, 10],
        [0, 11, -1],
        [0, 11, 12],
        [0, 11, 13],
        [0, 11, 14],
        [0, 11, 15],
        [0, 11, 16],
        [0, 17, -1],
        [0, 17, 18],
        [0, 17, 19],
        [0, 17, 20],
        [0, 17, 21],
        [0, 17, 22],
        [23, -1, -1],
        [23, 24, -1],
        [23, 25, -1],
        [23, 26, -1],
        [23, 27, -1],
        [23, 28, -1],
        [23, 29, -1],
        [23, 29, 30],
        [23, 29, 31],
        [23, 29, 32],
        [23, 29, 33],
        [23, 29, 34],
        [23, 35, -1],
        [23, 35, 36],
        [23, 35, 37],
        [23, 35, 38],
        [39, -1, -1],
        [39, 40, -1],
        [39, 40, 41],
        [39, 40, 42],
        [39, 40, 43],
        [39, 44, -1],
        [39, 44, 45],
        [39, 46, -1],
        [39, 47, -1],
        [39, 47, 48],
        [39, 49, -1],
        [39, 50, -1],
        [39, 51, -1],
        [39, 52, -1],
        [39, 53, -1],
        [39, 54, -1],
        [39, 54, 55],
        [39, 54, 56],
        [57, -1, -1],
        [57, 58, -1],
        [57, 58, 59],
        [57, 60, -1],
        [57, 60, 61],
        [57, 60, 62],
        [57, 63, -1],
        [57, 63, 64],
        [65, -1, -1],
        [65, 66, -1],
        [65, 67, -1],
        [65, 68, -1],
        [69, -1, -1],
        [69, 70, -1],
        [69, 70, 71],
        [69, 70, 72],
        [69, 70, 73],
        [69, 74, -1],
        [69, 74, 75],
        [69, 74, 76],
        [69, 74, 77],
        [69, 78, -1],
        [69, 78, 79],
        [69, 78, 80],
        [69, 78, 81],
        [82, -1, -1],
        [82, 83, -1],
        [82, 84, -1],
        [85, -1, -1],
        [86, -1, -1],
        [82, 87, -1],
        [0, 88, -1],
        [82, 89, -1],
        [90, -1, -1],
        [90, 91, -1],
        [90, 92, -1],
        [93, -1, -1],
        [93, 94, -1],
        [93, 95, -1],
        [93, 96, -1],
        [93, 97, -1],
        [93, 98, -1],
        [93, 99, -1],
        [100, -1, -1],
        [100, 101, -1],
        [100, 102, -1],
        [100, 103, -1],
        [100, 104, -1],
        [100, 105, -1],
        [100, 106, -1]]


def _infer(model, root_path, test_loader=None):

    if test_loader is None:
        test_loader = data_loader(
            root=root_path, phase='test', split=0.0, batch_size=1, submit=True)

    model.eval()
    ret_id = []
    ret_cls = []
    for idx, (data_id, image) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        fc = model(image)
        fc = fc.squeeze().detach().cpu().numpy()
        data_id = data_id[0].item()

        res_cls = np.argmax(fc)
        results = get_parent([], res_cls).reverse()
        while len(results) < 3:
            results.append(-1)

        assert len(results) == 3

        res_id = data_id
        ret_cls.append(results)
        ret_id.append(res_id)

    return [ret_id, ret_cls]


def bind_nsml(model, optimizer, scheduler):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        print('loaded')

    def infer(root_path, top_k=1):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer, use_nsml_legacy=False)


def init_weight(model):
    # TODO Modify init weight algorithm
    print("init weight")
    for m in model.modules():
        print("module = ", m)
        print("module type= ", (type(m)))
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def get_model():
    # model = timm.create_model('mnasnet_100', pretrained=True)
    model = timm.create_model('mnasnet_100', pretrained=True)
    model.classifier = nn.Linear(
        in_features=1280, out_features=num_classes, bias=True)

    return model


def update_loss(pred, loss_fn1, loss_fn2, loss_fn3, label_0, label_1, label_2):

    loss = loss_fn1(pred, label_0) + loss_fn2(pred,
                                              label_1) + loss_fn3(pred, label_2)

    return loss


def check_data_set(config):
    print("check data start !!")
    train_split = config.train_split
    batch_size = config.batch_size

    tr_loader = data_loader(root=DATASET_PATH, phase='train',
                            split=train_split, batch_size=batch_size, submit=False)

    arr = [0 for i in range(107)]

    cuda = config.cuda

    for i, data in enumerate(tr_loader):
        print(i, f"Start Total Data = {i * 64}")
        _, x, label_0, label_1, label_2 = data

        label_0 = label_0.cuda()[:, 0]
        label_0.tolist()

        for value in label_0:
            arr[value] += 1

    print("Data Count")

    for i in range(len(arr)):
        print(
            f"{i} {data_name[i]} count = {arr[i]} percentage = {(arr[i] /7232) * 100}%")


if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=107)
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--train_split", type=float, default=1.0)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=10)
    args.add_argument("--print_iter", type=int, default=10)

    # reserved for nsml
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)

    config = args.parse_args()

    # get configurations
    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    mode = config.mode
    train_split = config.train_split
    batch_size = config.batch_size
    # # initialize model using timm
    # check_data_set(config)

    total_models = ['efficientnet_b3_pruned',
                    'efficientnet_b3', 'efficientnet_b2_pruned', 'res2net101_26w_4s', 'tf_efficientnet_b7_ap', 'tf_efficientnet_b7_ns']

    # b1_pruned 0.31 b2_pruned 0.46
    model = timm.create_model('dm_nfnet_f0', pretrained=True)

    # print("Models State Dict")

    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
                                             verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print(f"{model.classifier.in_features}")
