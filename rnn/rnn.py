import torch
from d2l import torch as d2l
import math
from torch import nn
from torch.nn import functional as F

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
X = torch.arange(10).reshape(2, 5)
print(F.one_hot(X.T, 28))


def get_model_params(vocab_size, num_hiddens, device):
    """
    初始化模型参数
    :param vocab_size:字典大小，即输入层和输出层大小
    :param num_hiddens:隐藏层大小
    :param device: GPU或者CPU
    :return:初始化参数List
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device = device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal(num_hiddens, num_hiddens)
    b_h  = torch.zeros(num_hiddens, device=device)
    # 输出层
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad(True)
    return params


# 在初始化时返回隐状态
def init_rnn_state(batch_size, num_hiddens, device):
    # 隐状态包含多个变量的情况， 而使用元组可以更容易地处理些
    return (torch.zeros(batch_size, num_hiddens, device=device),)


def rnn(inputs, state, params):
    """
    定义RNN如何在一个时间步内计算隐状态和输出
    :param inputs:RNN的输入X
    :param state: 隐状态
    :param params:模型参数
    :return:输出Y的List以及隐状态H
    """
    # inpput的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # 通过inputs最外层的维度实现循环， 以便逐时间步更新小批量数据的隐状态H
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, )



