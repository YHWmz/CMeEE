from fastNLP.embeddings.embedding import TokenEmbedding
from fastNLP.core import Vocabulary
from fastNLP.io.file_utils import PRETRAIN_STATIC_FILES, _get_embedding_url, cached_path
import os
import warnings
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

from fastNLP.core import logger
from fastNLP.io.file_utils import PRETRAINED_BERT_MODEL_DIR

class MyDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0<=p<=1
        self.p = p

    def forward(self, x):
        if self.training and self.p>0.001:
            # print('mydropout!')
            mask = torch.rand(x.size())
            # print(mask.device)
            mask = mask.to(x)
            # print(mask.device)
            mask = mask.lt(self.p)
            x = x.masked_fill(mask, 0)/(1-self.p)
        return x


class StaticEmbedding(TokenEmbedding):
    """
    StaticEmbedding组件. 给定预训练embedding的名称或路径，根据vocab从embedding中抽取相应的数据(只会将出现在vocab中的词抽取出来，
    如果没有找到，则会随机初始化一个值(但如果该word是被标记为no_create_entry的话，则不会单独创建一个值，而是会被指向unk的index))。
    当前支持自动下载的预训练vector有以下的几种(待补充);
    Example::
        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import StaticEmbedding
        >>> vocab = Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-50d')
        >>> vocab = Vocabulary().add_word_lst(["The", 'the', "THE"])
        >>> embed = StaticEmbedding(vocab, model_dir_or_name="en-glove-50d", lower=True)
        >>> # "the", "The", "THE"它们共用一个vector，且将使用"the"在预训练词表中寻找它们的初始化表示。
        >>> vocab = Vocabulary().add_word_lst(["The", "the", "THE"])
        >>> embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=5, lower=True)
        >>> words = torch.LongTensor([[vocab.to_index(word) for word in ["The", "the", "THE"]]])
        >>> embed(words)
        >>> tensor([[[ 0.5773,  0.7251, -0.3104,  0.0777,  0.4849],
                     [ 0.5773,  0.7251, -0.3104,  0.0777,  0.4849],
                     [ 0.5773,  0.7251, -0.3104,  0.0777,  0.4849]]],
                   grad_fn=<EmbeddingBackward>)  # 每种word的输出是一致的。
    """

    def __init__(self, 
    vocab: Vocabulary, 
    model_dir_or_name: str = 'en', 
    requires_grad: bool = False, 
    init_method=None,
    dropout=0, 
    word_dropout=0, 
    **kwargs ):
        """
        :param vocab: Vocabulary. 若该项为None则会读取所有的embedding。
        :param model_dir_or_name: 可以有两种方式调用预训练好的static embedding：第一种是传入embedding文件夹(文件夹下应该只有一个
            以.txt作为后缀的文件)或文件路径；第二种是传入embedding的名称，第二种情况将自动查看缓存中是否存在该模型，没有的话将自动下载。
            如果输入为None则使用embedding_dim的维度随机初始化一个embedding。
        :param bool requires_grad: 是否需要gradient. 默认为True
        :param callable init_method: 如何初始化没有找到的值。可以使用torch.nn.init.*中各种方法, 传入的方法应该接受一个tensor，并
            inplace地修改其值。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param dict kwarngs: only_train_min_freq, 仅对train中的词语使用min_freq筛选; only_norm_found_vector是否仅对在预训练中找到的词语使用normalize。
        """
        super(StaticEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)

        # 得到cache_path
        if model_dir_or_name is None:
            assert embedding_dim >= 1, "The dimension of embedding should be larger than 1."
            embedding_dim = int(embedding_dim)
            model_path = None
        elif model_dir_or_name.lower() in PRETRAIN_STATIC_FILES:
            model_url = _get_embedding_url('static', model_dir_or_name.lower())
            model_path = cached_path(model_url, name='embedding')
            # 检查是否存在
        elif os.path.isfile(os.path.abspath(os.path.expanduser(model_dir_or_name))):
            model_path = os.path.abspath(os.path.expanduser(model_dir_or_name))
        else:
            raise ValueError(f"Cannot recognize {model_dir_or_name}.")


        # 读取embedding
        if model_path:
            embedding = self._load_with_vocab(model_path, vocab=vocab, init_method=init_method)
        else:
            print("Lack of model, random generalize embedding!")
            embedding = self._randomly_init_embed(len(vocab), embedding_dim, init_method)
            self.register_buffer('words_to_words', torch.arange(len(vocab)).long())

        self.embedding = nn.Embedding(num_embeddings=embedding.shape[0], embedding_dim=embedding.shape[1],
                                      padding_idx=vocab.padding_idx,
                                      max_norm=None, norm_type=2, scale_grad_by_freq=False,
                                      sparse=False, _weight=embedding)
        self._embed_size = self.embedding.weight.size(1)
        self.requires_grad = requires_grad
        self.dropout = MyDropout(dropout)

    def _randomly_init_embed(self, num_embedding, embedding_dim, init_embed=None):
        """
        :param int num_embedding: embedding的entry的数量
        :param int embedding_dim: embedding的维度大小
        :param callable init_embed: 初始化方法
        :return: torch.FloatTensor
        """
        embed = torch.zeros(num_embedding, embedding_dim)

        if init_embed is None:
            nn.init.uniform_(embed, -np.sqrt(3 / embedding_dim), np.sqrt(3 / embedding_dim))
        else:
            init_embed(embed)

        return embed

    def _load_with_vocab(self, embed_filepath, vocab, dtype=np.float32, padding='<pad>', unknown='<unk>',
                         error='strict', init_method=None):
        """
        从embed_filepath这个预训练的词向量中抽取出vocab这个词表的词的embedding。EmbedLoader将自动判断embed_filepath是
        word2vec(第一行只有两个元素)还是glove格式的数据。
        :param str embed_filepath: 预训练的embedding的路径。
        :param vocab: 词表 :class:`~fastNLP.Vocabulary` 类型，读取出现在vocab中的词的embedding。
            没有出现在vocab中的词的embedding将通过找到的词的embedding的正态分布采样出来，以使得整个Embedding是同分布的。
        :param dtype: 读出的embedding的类型
        :param str padding: 词表中padding的token
        :param str unknown: 词表中unknown的token
        :param str error: `ignore` , `strict` ; 如果 `ignore` ，错误将自动跳过; 如果 `strict` , 错误将抛出。
            这里主要可能出错的地方在于词表有空行或者词表出现了维度不一致。
        :param init_method: 如何初始化没有找到的值。可以使用torch.nn.init.*中各种方法。默认使用torch.nn.init.zeros_
        :return torch.tensor:  shape为 [len(vocab), dimension], dimension由pretrain的embedding决定。
        """
        assert isinstance(vocab, Vocabulary), "Only fastNLP.Vocabulary is supported."
        if not os.path.exists(embed_filepath):
            raise FileNotFoundError("`{}` does not exist.".format(embed_filepath))
        with open(embed_filepath, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts) == 2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts) - 1
                f.seek(0)
            matrix = {}
            if vocab.padding:
                matrix[vocab.padding_idx] = torch.zeros(dim)
            if vocab.unknown:
                matrix[vocab.unknown_idx] = torch.zeros(dim)
            found_count = 0
            found_unknown = False
            for idx, line in enumerate(f, start_idx):
                try:
                    parts = line.strip().split()
                    word = ''.join(parts[:-dim])
                    nums = parts[-dim:]
                    # 对齐unk与pad
                    if word == padding and vocab.padding is not None:
                        word = vocab.padding
                    elif word == unknown and vocab.unknown is not None:
                        word = vocab.unknown
                        found_unknown = True
                    if word in vocab:
                        index = vocab.to_index(word)
                        matrix[index] = torch.from_numpy(np.fromstring(' '.join(nums), sep=' ', dtype=dtype, count=dim))
                        found_count += 1
                except Exception as e:
                    if error == 'ignore':
                        warnings.warn("Error occurred at the {} line.".format(idx))
                    else:
                        logger.error("Error occurred at the {} line.".format(idx))
                        raise e
            logger.info("Found {} out of {} words in the pre-training embedding.".format(found_count, len(vocab)))
            for word, index in vocab:
                if index not in matrix and not vocab._is_word_no_create_entry(word):
                    if found_unknown:  # 如果有unkonwn，用unknown初始化
                        matrix[index] = matrix[vocab.unknown_idx]
                    else:
                        matrix[index] = None
            # matrix中代表是需要建立entry的词
            vectors = self._randomly_init_embed(len(matrix), dim, init_method)

            if vocab.unknown is None:  # 创建一个专门的unknown
                unknown_idx = len(matrix)
                vectors = torch.cat((vectors, torch.zeros(1, dim)), dim=0).contiguous()
            else:
                unknown_idx = vocab.unknown_idx
            self.register_buffer('words_to_words', torch.full((len(vocab),), fill_value=unknown_idx).long())
            for index, (index_in_vocab, vec) in enumerate(matrix.items()):
                if vec is not None:
                    vectors[index] = vec
                self.words_to_words[index_in_vocab] = index

            return vectors

    def drop_word(self, words):
        """
        按照设定随机将words设置为unknown_index。
        :param torch.LongTensor words: batch_size x max_len
        :return:
        """
        if self.word_dropout > 0 and self.training:
            mask = torch.rand(words.size())
            mask = mask.to(words.device)
            mask = mask.lt(self.word_dropout)
            # mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float)
            #             # mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
            #             # mask = mask.to(words.device)
            pad_mask = words.ne(self._word_pad_index)
            mask = mask.__and__(pad_mask)
            words = words.masked_fill(mask, self._word_unk_index)
        return words

    def forward(self, words):
        """
        传入words的index
        :param words: torch.LongTensor, [batch_size, max_len]
        :return: torch.FloatTensor, [batch_size, max_len, embed_size]
        """
        if hasattr(self, 'words_to_words'):
            words = self.words_to_words[words]
        words = self.drop_word(words)
        words = self.embedding(words)
        words = self.dropout(words)
        return words

if __name__ == "__main__":
    unimodel_path = '../pretrain_model/gigaword_chn.all.a2b.uni.ite50.vec'
    bimodel_path = '../pretrain_model/gigaword_chn.all.a2b.bi.ite50.vec'
    wordmodel_path = '../pretrain_model/ctb.50d.vec'
    vocab = Vocabulary()
    vocab.add_word_lst(['一','二','我','是','弱','智'])
    uni_embedding = StaticEmbedding(vocab=vocab,model_dir_or_name=unimodel_path)
    test_text = ['我','是','弱','智']
    words = torch.LongTensor([vocab.to_index(w) for w in test_text]).reshape(-1,1)
    print(words)
    print(uni_embedding.forward(words=words).shape)
    print(uni_embedding.forward(words=words))
     