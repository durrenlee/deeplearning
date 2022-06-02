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
time = torch.arange(1, T+1, dtype=torch.float32)
# to use sin function with normal noise data to generate a vector X
x = torch.sin(0.01*time + torch.normal(0, 0.2, (T, )))
d2l.plot(time, x, 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
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