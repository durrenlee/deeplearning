"""
 Sequence data has time sequence structure.
 Sequence data is changed while time is changed.
 Sequence data examples: music, language,  video, stock price.

 x1, x2, ..., xt: non independent random variables
 p(a, b) = p(a)p(b|a) = p(b)p(a|b)
 p(x) = p(x1)p(x2|x1)P(x3|x1, x2)...p(xt|x1, x2,...xt-1)
      = p(xt)p(xt-1|xt)P(xt-2|xt-1, xt)...p(x1|x2,...xt-1, xt)

 To model conditional probability:
 p(xt|x1, x2,...xt-1) = p(xt|f(x1, x2,...xt-1))
 f(x1, x2,...xt-1):Autoregressive model, model for history sequence data
 
 To predict xt with history sequence data:
 solution 1:
    Markov Assumption: current data is just related to T history data, older data than T are careless
    p(xt|x1, x2,...xt-1) = p(xt|f(xt-T,...xt-1))
    f(xt-T,...xt-1): can be trained via MLP model, linear regression or other models
 solution 2:
    Latent variable model: latent variable ht is given to represent history information:
    ht = f(x1, ..., xt-1) -> xt = p(xt|ht), ht = f(ht-1, xt-1)
    such as RNN

 sample code:
    sin function and noise data are used to generate sequence data, then Markov Assumption and MLP
    to train data, finally the different predictions will be executed.
"""

import torch
from torch import nn
from d2l import torch as d2l

# sin function and noise data are used to generate sequence data
T = 1000
# to generate a vector from 1 to T
time = torch.arange(1, T + 1, dtype=torch.float32)
# to use sin function with normal noise data to generate a vector X
x = torch.sin(0.01 * time + torch.normal(0, 0.2, (T,)))
d2l.plot(time, x, 'time', 'x', xlim=[1, 1000], figsize=(8, 5))
d2l.use_svg_display()
d2l.plt.show()

# Markov assumption
tau = 4
# features shape: [996, 4]
features = torch.zeros((T - tau, tau))
for i in range(tau):
    # truncate X into 4 parts: 0~996, 1~997, 2~998, 3~999
    # 0~996 is put into the 1st column of features which row number is 996
    # other parts follow same above
    features[:, i] = x[i: T - tau + i]
# truncate x from index:4 to the end(999): the last 996 numbers and given to labels
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# features[:n_train], labels[:n_train]: 600 rows
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)


# Glorot initialization
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


# MLP
def get_net():
    # input layer: 4 units, hidden layer: 10 units, output: 1 (scalar)
    mlp_net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    mlp_net.apply(init_weights)
    return mlp_net


# MSE Loss, note that there is no 1/2
loss = nn.MSELoss()


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            # Gradient of the Optimizer to be 0
            trainer.zero_grad()
            l = loss(net(X), y)
            # calculate gradient
            l.backward()
            # update all parameters
            trainer.step()
        print(f'epoch{epoch + 1}, ', f'loss:{d2l.evaluate_loss(net, train_iter, loss):f}')


net = get_net()
train(net, train_iter, loss, 5, 0.01)

# predict next one step with train dataset:features
onestep_preds = net(features)
d2l.plot(
    [time, time[tau:]],
    [x.detach().numpy(), onestep_preds.detach().numpy()],
    'time',
    'x',
    legend=['data', '1-step preds'],
    xlim=[1, 1000],
    figsize=(8,5)
)
d2l.use_svg_display()
d2l.plt.show()

# predict multiple steps
multistep_preds = torch.zeros(T)
multistep_preds[:n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    # multistep_preds[i - tau: i].shape = torch.Size([4])
    # multistep_preds[i - tau: i].reshape((1, -1)).shape = torch.Size([1, 4])
    # use 600(real data), 601(real data), 602(real data), 603(real data)---predict 604(prediction data)
    # then use 604(prediction data), 603(real data), 602(real data), 601(real data) --predict 605(prediction data)
    # then use 605(prediction data), 604(prediction data),603(real data), 602(real data) --predict 606(prediction data)
    # and so on...to predict 1000(prediction data)
    multistep_preds[i] = net(multistep_preds[i - tau: i].reshape((1, -1)))
d2l.plot(
    [time, time[tau:], time[n_train + tau:]],
    [x.detach().numpy(), onestep_preds.detach().numpy(), multistep_preds[n_train + tau:].detach().numpy()],
    'time',
    'x',
    legend=['data', '1-step preds', 'multistep preds'],
    xlim=[1, 1000],
    figsize=(8, 5)
)
d2l.use_svg_display()
d2l.plt.show()

max_steps = 64


features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot(
    [time[tau + i - 1: T - max_steps + i] for i in steps],
    [features[:, (tau + i - 1)].detach().numpy() for i in steps],
    'time',
    'x',
    legend=[f'{i}-step preds' for i in steps],
    xlim=[5, 1000],
    figsize=(8, 5)
)
d2l.use_svg_display()
d2l.plt.show()