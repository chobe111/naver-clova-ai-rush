import os
import math
import datetime
from os.path import split

import numpy as np

import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse

from data_local_loader2 import data_loader, data_analyze_loader
from model import EnsembleModelA

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

parent_info = [set() for i in range(107)]

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


def get_parent(parent_list: List, cur_index: int):
    parent_list.append(cur_index)
    if len(parent_info[cur_index]) == 0:
        return parent_list

    return get_parent(parent_list, list(parent_info[cur_index])[0])


def set_parent(config):
    train_split = config.train_split
    batch_size = config.batch_size

    tr_loader = data_analyze_loader(root=DATASET_PATH, phase='train',
                                    split=train_split, batch_size=batch_size)

    cuda = config.cuda

    for i, data in enumerate(tr_loader):
        _, x, label_0, label_1, label_2 = data

        label_0 = label_0.cuda()[:, 0]
        label_1 = label_1.cuda()[:, 0]
        label_2 = label_2.cuda()[:, 0]

        label0_list = label_0.tolist()
        label1_list = label_1.tolist()
        label2_list = label_2.tolist()

        for i in range(len(label0_list)):
            if label1_list[i] != -1:
                parent_info[label1_list[i]].add(label0_list[i])

            if label2_list[i] != -1:
                parent_info[label2_list[i]].add(label1_list[i])


def _infer(model, root_path, test_loader=None):
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

    if test_loader is None:
        test_loader = data_loader(
            root=root_path, phase='test', split=0.0, batch_size=1, submit=True)

    modelA = model.modelA
    modelB = model.modelB
    modelC = model.modelC

    # model.eval()

    modelA.eval()
    modelB.eval()
    modelC.eval()

    dp1 = [0 for i in range(107)]
    dp2 = [0 for i in range(107)]
    dp3 = [0 for i in range(107)]

    ret_id = []
    ret_cls = []
    for idx, (data_id, image) in enumerate(tqdm(test_loader)):
        print("idx = ", idx)

        image = image.cuda()
        fc1 = modelA(image)
        fc1 = torch.softmax(fc1, dim=1)
        fc1 = fc1.squeeze().detach().cpu().numpy()

        res_fc1 = np.argmax(fc1)
        res_fc1_per = fc1[res_fc1]
        dp1[res_fc1] += res_fc1_per

        fc2 = modelB(image)
        fc2 = torch.softmax(fc2, dim=1)
        fc2 = fc2.squeeze().detach().cpu().numpy()

        res_fc2 = np.argmax(fc2)
        res_fc2_per = fc2[res_fc2]

        parent1 = case[res_fc2][0]
        dp1[parent1] += res_fc2_per
        dp2[res_fc2] += res_fc2_per

        fc3 = modelC(image)
        fc3 = torch.softmax(fc3, dim=1)
        fc3 = fc3.squeeze().detach().cpu().numpy()

        res_fc3 = np.argmax(fc3)
        res_fc3_per = fc3[res_fc3]

        parent1 = case[res_fc3][0]
        parent2 = case[res_fc3][1]

        dp1[parent1] += res_fc3_per
        dp2[parent2] += res_fc3_per
        dp3[res_fc3] += res_fc3_per

        child1 = np.argmax(dp1)
        child2 = np.argmax(dp2)
        child3 = np.argmax(dp3)

        print("============== total result percentage ==============",
              res_fc1_per, res_fc2_per, res_fc3_per)

        print("res_fc1_per = ", res_fc1_per)
        print("res_fc2_per = ", res_fc2_per)
        print("res_fc3_per = ", res_fc3_per)

        print("total child = ", child1, child2, child3)
        res_id = data_id
        ret_cls.append([child1, child2, child3])
        ret_id.append(res_id)

    return [ret_id, ret_cls]


def bind_ensemble_nsml(model):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'modelA': model.modelA.state_dict(),
            'modelB': model.modelB.state_dict(),
            'modelC': model.modelC.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.modelA.load_state_dict(state['modelA'])
        model.modelB.load_state_dict(state['modelB'])
        model.modelC.load_state_dict(state['modelC'])
        print('loaded')

    def infer(root_path, top_k=1):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer, use_nsml_legacy=False)


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
    # # TODO Modify init weight algorithm
    # print("init weight")
    # for m in model.modules():
    #     print("module = ", m)
    #     print("module type= ", (type(m)))
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #     elif isinstance(m, nn.BatchNorm2d):
    #         m.weight.data.fill_(1)
    #         m.bias.data.zero_()
    pass


def get_model():
    # model = timm.create_model('mnasnet_100', pretrained=True)
    model = timm.create_model('mnasnet_100', pretrained=True)

    # freeze layer
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=640, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(in_features=640, out_features=num_classes)
    )

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

    arr = [set() for i in range(107)]

    cuda = config.cuda

    for i, data in enumerate(tr_loader):
        if i == 1:
            break

        print(i, "start")
        _, x, label_0, label_1, label_2 = data

        label_0 = label_0.cuda()[:, 0]
        label_1 = label_1.cuda()[:, 0]
        label_2 = label_2.cuda()[:, 0]

        print("label data length = ", label_0.tolist())

    for index, s in enumerate(arr):
        print(f'{index} class number of parent = {len(s)}')


def train_base(model, loss_fn, how):
    print("Model {} Inference Start!!".format(how))

    tr_loader = data_loader(root=DATASET_PATH, phase='train',
                            split=train_split, batch_size=batch_size, submit=False)

    optimizer = Adam([param for param in model.parameters()
                      if param.requires_grad], lr=base_lr, weight_decay=1e-4)

    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    time_ = datetime.datetime.now()
    num_batches = len(tr_loader)
    train_stat = AverageMeter()
    global_iter = 0
    for epoch in range(num_epochs):
        print(f"{epoch} epoch start!!")
        model.train()
        for iter_, data in enumerate(tr_loader):
            global_iter += iter_
            _, x, label_0, label_1, label_2 = data
            if cuda:
                x = x.cuda()
                label_0 = label_0.cuda()[:, 0]
                label_1 = label_1.cuda()[:, 0]
                label_2 = label_2.cuda()[:, 0]

                label_list = [label_0, label_1, label_2]

            pred = model(x)
            # very naive loss function given
            # I don't know but I will concern just first one.
            loss = loss_fn(pred, label_list[how])
            train_stat.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iter_ + 1) % print_iter == 0:
                elapsed = datetime.datetime.now() - time_
                expected = elapsed * (num_batches / print_iter)
                _epoch = epoch + ((iter_ + 1) / num_batches)
                print('[{:.3f}/{:d}] loss({}) '
                      'elapsed {} expected per epoch {}'.format(
                          _epoch, num_epochs, train_stat.avg, elapsed, expected))

                time_ = datetime.datetime.now()

                if IS_ON_NSML:
                    report_dict = dict()
                    report_dict["train__loss"] = float(train_stat.avg)
                    report_dict["train__lr"] = optimizer.param_groups[0]["lr"]
                    nsml.report(step=global_iter, **report_dict)

        scheduler.step()
        time_ = datetime.datetime.now()
        elapsed = datetime.datetime.now() - time_
        print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))


def print_model_info(model):
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
                                             verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=107)
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--train_split", type=float, default=1.0)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=20)
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

    # check_data_set(config)
    # # initialize model using timm
    # pprint(timm.list_models(pretrained=True))

    modelA = get_model()
    modelB = get_model()
    modelC = get_model()

    ensemble_model = EnsembleModelA(modelA, modelB, modelC)
    # get flops of model

    # print_model_info(modelA)
    loss_fn_0 = nn.CrossEntropyLoss(ignore_index=-1)
    loss_fn_1 = nn.CrossEntropyLoss(ignore_index=-1)
    loss_fn_2 = nn.CrossEntropyLoss(ignore_index=-1)
    # init_weight(model)

    if cuda:
        modelA = modelA.cuda()
        modelB = modelB.cuda()
        modelC = modelC.cuda()

        loss_fn_0 = loss_fn_1.cuda()
        loss_fn_1 = loss_fn_1.cuda()
        loss_fn_2 = loss_fn_1.cuda()

    if IS_ON_NSML:
        bind_ensemble_nsml(ensemble_model)

        if config.pause:
            nsml.paused(scope=locals())

    if mode == 'train':
        train_base(modelA, loss_fn_0, 0)
        train_base(modelB, loss_fn_1, 1)
        train_base(modelC, loss_fn_2, 2)

        if IS_ON_NSML:
            nsml.save('ensemble_best')
