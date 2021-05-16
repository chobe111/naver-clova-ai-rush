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

from data_local_loader import data_loader

from tqdm import tqdm

import timm
from pprint import pprint
from utils import AverageMeter
from ptflops import get_model_complexity_info


try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML

except:
    IS_ON_NSML=False
    DATASET_PATH = '../1-3-DATA-fin'


def _infer(model, root_path, test_loader=None):

    if test_loader is None:
        test_loader = data_loader(root=root_path, phase='test', split=0.0, batch_size=1, submit=True)

    model.eval()
    ret_id = []
    ret_cls= []
    for idx, (data_id, image) in enumerate(tqdm(test_loader)):
        image = image.cuda()
        fc = model(image)
        fc = fc.squeeze().detach().cpu().numpy()
        data_id = data_id[0].item()

        res_cls = np.argmax(fc)
        res_id = data_id

        ret_cls.append([res_cls, -1, -1])
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
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=107)
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--train_split", type=float, default=0.8)
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

    #initialize model using timm
    pprint(timm.list_models(pretrained=True))
    model = timm.create_model('mnasnet_100', pretrained=True)
    model.classifier = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
                                             verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    loss_fn_0 = nn.CrossEntropyLoss()
    loss_fn_1 = nn.CrossEntropyLoss()
    loss_fn_2 = nn.CrossEntropyLoss()

    init_weight(model)

    if cuda:
        model = model.cuda()
        loss_fn_0 = loss_fn_1.cuda()
        loss_fn_1 = loss_fn_1.cuda()
        loss_fn_2 = loss_fn_1.cuda()

    optimizer = Adam([param for param in model.parameters() if param.requires_grad], lr=base_lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    if IS_ON_NSML:
        bind_nsml(model, optimizer, scheduler)

        if config.pause:
            nsml.paused(scope=locals())


    if mode == 'train':
        tr_loader = data_loader(root=DATASET_PATH, phase='train', split=train_split, batch_size=batch_size, submit=False)

        time_ = datetime.datetime.now()
        num_batches = len(tr_loader)

        train_stat = AverageMeter()

        global_iter = 0
        for epoch in range(num_epochs):

            model.train()
            for iter_, data in enumerate(tr_loader):
                global_iter += iter_
                _, x, label_0, label_1, label_2 = data
                if cuda:
                    x = x.cuda()
                    label_0 = label_0.cuda()[:,0]
                    label_1 = label_1.cuda()[:,0]
                    label_2 = label_2.cuda()[:,0]

                pred = model(x)

                #very naive loss function given
                loss = loss_fn_0(pred, label_0) # I don't know but I will concern just first one.

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

            if IS_ON_NSML:
                nsml.save(str(epoch + 1))

            time_ = datetime.datetime.now()
            elapsed = datetime.datetime.now() - time_
            print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))