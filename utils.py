from posixpath import realpath
import torch
import numpy as np

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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_acc(pred: torch.Tensor, gt: torch.Tensor):
    '''
    pred shape is [batch_size, num_classes]
    gt shape is [batch_size]
    '''
    pred = pred.squeeze().detach().cpu().numpy()
    gt = gt.cpu().numpy()

    total_data_len = 0
    total_score = 0

    pred = np.argmax(pred, axis=1)

    for real, syn in zip(gt, pred):
        total_data_len += 1
        real_list = case[real]
        syn_list = case[syn]

        score = 0
        denom = 0
        for real2, syn2 in zip(real_list, syn_list):
            if real2 > -1:
                denom += 1.0
                if real2 == syn2:
                    score += 1.0
            else:
                break
        score = score / denom
        total_score += score

    assert len(pred) == len(gt)

    return total_score / len(pred)
