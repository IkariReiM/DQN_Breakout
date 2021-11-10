import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        #初始化网络，将Q分为v和a
        self.__fc_v = nn.Linear(64 * 7 * 7, 512)
        self.__v = nn.Linear(512, 1)
        self.__fc_a = nn.Linear(64 * 7 * 7, 512)
        self.__a = nn.Linear(512, action_dim)
        self.__device = device

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        #计算当前的v值
        v = F.relu(self.__fc_v(x.view(x.size(0), -1)))
        v = self.__v(v)
        #计算当前的a值
        a = F.relu(self.__fc_a(x.view(x.size(0), -1)))
        a = self.__a(a)
        return v + (a-a.mean(dim = 1,keepdim = True))

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
