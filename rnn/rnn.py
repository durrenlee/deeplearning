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
    W_hh = normal((num_hiddens, num_hiddens))
    b_h  = torch.zeros(num_hiddens, device=device)
    # 输出层
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
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
    # 通过inputs(5*2*28独热编码)最外层的维度实现循环， 以便逐时间步更新小批量数据的隐状态H
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        # X: 2*28独热编码
        # W_xh: 28*512矩阵X
        # H: 2*512 隐状态矩阵
        # W_hh: 512*512隐状态参数矩阵
        # b_h: 512的一维偏置向量,使用广播机制都加到2*512中
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # W_hq: 512*28矩阵X
        # b_q: 28的一维偏置向量，使用广播机制都加到2*28中
        Y = torch.mm(H, W_hq) + b_q
        # Y: 2*28输出结果矩阵
        outputs.append(Y)
        # outputs: 长度为5的2*28矩阵列表
        # cat：将长度为5的2*28矩阵列表按照纵轴方向叠加成10*28的矩阵
        # 同时输出包含2*512 只有一个隐状态的矩阵的tuple
    return torch.cat(outputs, dim=0), (H, )


class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # 5*2*28独热编码
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # 调用rnn进行前向网络计算，返回10*28的输出结果矩阵
        # 同时返回包含2*512 只有一个隐状态的矩阵的tuple
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


# 检查输出是否具有正确的形状。 例如，隐状态的维数是否保持不变。
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_model_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
# state: 2*512的0矩阵
Y, new_state = net(X.to(d2l.try_gpu()), state)
# Y.shape输出形状是（时间步数批量大小，词表大小）：torch.Size([10, 28]
print(Y.shape)
# len(new_state):1
print(len(new_state))
# 隐状态形状保持不变，即（批量大小，隐藏单元数）:torch.Size([2, 512])
print(new_state[0].shape)
