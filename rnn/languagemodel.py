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


