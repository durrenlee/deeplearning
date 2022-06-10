import random
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])

freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log', figsize=(8, 5))
d2l.use_svg_display()
d2l.plt.show()
"""
从词频图中看出词频以一种明确的方式迅速衰减,
将前几个单词作为例外消除后，剩余的所有单词大致遵循双对数坐标图上的一条直线,
单词的频率满足齐普夫定律,
这告诉我们想要通过计数统计和平滑来建模单词是不可行的， 
因为这样建模的结果会大大高估尾部单词的频率，也就是所谓的不常用单词。
"""


# 我们来看看二元语法的频率是否与一元语法的频率表现出相同的行为方式
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])
# 在十个最频繁的词对中，有九个是由两个停用词组成的
# 再进一步看看三元语法的频率是否表现出相同的行为方式
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

# 对比三种模型中的词元频率：一元语法、二元语法和三元语法
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'], figsize=(8, 5))
d2l.use_svg_display()
d2l.plt.show()
# 单词序列似乎也遵循齐普夫定律,元组的数量并没有那么大, 很多元组很少出现

"""
序列变得太长而不能被模型一次性全部处理时， 我们可能希望拆分这样的序列方便模型读取
模型中的网络一次处理具有预定义长度（例如n个时间步）的一个小批量序列
现在的问题是如何随机生成一个小批量数据的特征和标签以供读取
任意长的序列可以被我们划分为具有相同时间步数的子序列
当训练神经网络时，这样的小批量子序列将被输入到模型中
如果我们只选择一个偏移量， 那么用于训练网络的、所有可能的子序列的覆盖范围将是有限的
因此，我们可以从随机偏移量开始划分序列， 以同时获得覆盖性（coverage）和随机性（randomness）
"""


"""
随机采样：每个样本都是在原始的长序列上任意捕获的子序列
在迭代过程中，来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
目标是基于到目前为止我们看到的词元来预测下一个词元
因此标签是移位了一个词元的原始序列
"""


def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    使用随机抽样生成一个小批量子序列,
    num_steps是每个子序列中预定义的时间步数
    从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    0<=随机开始位置<时间步长数num_steps
    """
    corpus = corpus[random.randint(0, num_steps - 1):]
    """
    减去1，原因是在得到标签Y时，
    当initial_indices_per_batch有最后一个起始索引时，
    data[j+1]可能不存在导致数组越界:
    ValueError: expected sequence of length 4 at dim 1 (got 5)
    """
    num_subseqs = (len(corpus)-1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    """
    迭代器：分割初始化索引列表initial_indices，
    分割大小为batch_size
    如：
    总序列为batch_size(2个序列) * num_batches(3个批次)=6
    shuffle过之后的初始化索引列表[25, 20, 0, 15, 10, 5]
    分割为[25, 20],[0, 15],[10, 5]
    然后依次得到样本X为[25, 26, 27, 28, 29, 30], [20, 21, 22, 23, 24]
    依次得到标签Y为[26, 27, 28, 29, 30, 31], [21, 22, 23, 24, 25]
    """
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)


# 两个相邻的小批量中的子序列，在原始序列上也是相邻的
# 每个mini batch中是相互独立的
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


print("使用顺序分区生成一个小批量子序列")
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)