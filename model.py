import torch
import torch.nn as nn

try:
    import nsml
except:
    IS_ON_NSML = False
    DATASET_PATH = '../1-3-DATA-fin'


class XgboostModel(nn.Module):
    def __init__(self, base_model, xgboost_model):
        super(XgboostModel, self).__init__()
        self.base_model = base_model
        self.xgboost_model = xgboost_model


class EnsembleModelA(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        # model 1 is efficient net with fine tuning until -10 paramters plus fc
        super(EnsembleModelA, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC

        # self.fc1 = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)

        out = out1 + out2 + out3
        # x = self.fc1(out)
        return torch.softmax(out, dim=1)


class EnsembleModelB(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelD, modelE):
        # model 1 is efficient net with fine tuning until -10 paramters plus fc
        super(EnsembleModelB, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.modelD = modelD
        self.modelE = modelE

        # self.fc1 = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)
        out4 = self.modelD(x)
        out5 = self.modelE(x)

        out = out1 + out2 + out3 + out4 + out5
        # x = self.fc1(out)
        return out
