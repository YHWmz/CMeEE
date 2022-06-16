import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from char_featurizer import Featurizer
import pdb
import jieba

NER_PAD, NO_ENT = '[PAD]', 'O'

LABEL1 = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'dis', 'bod']  # 按照出现频率从低到高排序
LABEL2 = ['sym']

LABEL  = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'] 

# 标签出现频率映射，从低到高
_LABEL_RANK = {L: i for i, L in enumerate(['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'])}

EE_id2label1 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL1 for P in ("B", "I")]
EE_id2label2 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL2 for P in ("B", "I")]
EE_id2label  = [ L for L in LABEL]

EE_label2id1 = {b: a for a, b in enumerate(EE_id2label1)}
EE_label2id2 = {b: a for a, b in enumerate(EE_id2label2)}
EE_label2id  = {b: a for a, b in enumerate(EE_id2label)}

EE_NUM_LABELS1 = len(EE_id2label1)
EE_NUM_LABELS2 = len(EE_id2label2)
EE_NUM_LABELS  = len(EE_id2label)

def load_data(path):
    D = []
    for d in json.load(open(path)):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                D[-1].append((start, end, EE_label2id[label]))
    return D

class EntDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item

class CollateForEnt:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encoder(self, item):
        text = item[0]
        token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)["offset_mapping"]
        start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        #将raw_text的下标 与 token的start和end下标对应
        encoder_txt = self.tokenizer.encode_plus(text, max_length=self.max_length, truncation=True)
        input_ids = encoder_txt["input_ids"]
        token_type_ids = encoder_txt["token_type_ids"]
        attention_mask = encoder_txt["attention_mask"]

        return text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []
        for item in examples:
            raw_text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask = self.encoder(item)

            labels = np.zeros((len(EE_label2id), self.max_length, self.max_length))
            for start, end, label in item[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[label, start, end] = 1
            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()
        batch_labels_ = torch.zeros((batch_labels.shape[0],batch_labels.shape[1],1))

        inputs = {
            "input_ids":batch_inputids,
            "attention_mask":batch_attentionmask,
            "token_type_ids":batch_segmentids,
            "labels": batch_labels,
            "labels_": batch_labels_,
        }
        return inputs

    def __call__(self, examples):
        return self.collate(examples=examples)

def get_dict(vocab_list):
    return { vocab_list[i]: i+1 for i in range(len(vocab_list)) }

FEATURE_TYPES = 8

class CollateForEnt_Chaizi:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.featurizer = Featurizer()
        vocab = self.featurizer.get_vocabulary()
        vocab[-1] = ['0','1','2','3','4','5','6','7','8','9']
        self.feature_size = [ len(v) for v in vocab]
        self.unk_index = sum(self.feature_size) + 1
        add_num = 0
        self.feature_ids_adder = []
        for sz in self.feature_size:
            self.feature_ids_adder.append(add_num)
            add_num += sz
        self.feature_dict = [ get_dict(vocab[i]) for i in range(len(vocab))]

    def handle_raw_text(self, raw_text):
        new_text = ""
        for t in raw_text:
            if '\u4e00' <= t <= '\u9fff':
                new_text = new_text + t
            else:
                new_text = new_text + '草'
        return new_text

    def handle_chaizi(self, text, max_len, mapping):
        text = self.handle_raw_text(text)
        result = self.featurizer.featurize(text)
        range_length = min(len(result[0]),self.max_length)
        feature_ids = np.zeros((max_len,FEATURE_TYPES))
        for i in range(range_length):
            for j in range(FEATURE_TYPES):
                if i not in mapping:
                    continue 
                feature = result[j][i]
                pos = mapping[i]
                if isinstance(feature, list):
                    feature = feature[0]
                if feature in self.feature_dict[j]:
                    feature_ids[pos,j] = self.feature_ids_adder[j] + self.feature_dict[j][feature]
                else:
                    feature_ids[pos,j] = self.unk_index
        return feature_ids

    def encoder(self, item):
        text = item[0]
        token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)["offset_mapping"]
        start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        #将raw_text的下标 与 token的start和end下标对应
        encoder_txt = self.tokenizer.encode_plus(text, max_length=self.max_length, truncation=True)
        input_ids = encoder_txt["input_ids"]
        token_type_ids = encoder_txt["token_type_ids"]
        attention_mask = encoder_txt["attention_mask"]

        return text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []
        batch_chaizi_ids = []
        for item in examples:
            raw_text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask = self.encoder(item)
            chaizi_ids = self.handle_chaizi(raw_text, max_len=len(input_ids), mapping=start_mapping)
            labels = np.zeros((len(EE_label2id), self.max_length, self.max_length))
            for start, end, label in item[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[label, start, end] = 1
            raw_text_list.append(raw_text)
            batch_chaizi_ids.append(chaizi_ids)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
        batch_chaiziids = torch.tensor(self.sequence_padding(batch_chaizi_ids)).long()
        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()
        batch_labels_ = torch.zeros((batch_labels.shape[0],batch_labels.shape[1],1))

        inputs = {
            "input_ids":batch_inputids,
            "chaizi_ids":batch_chaiziids,
            "attention_mask":batch_attentionmask,
            "token_type_ids":batch_segmentids,
            "labels": batch_labels,
            "labels_": batch_labels_,
        }
        return inputs

    def __call__(self, examples):
        return self.collate(examples=examples)

class CollateForEnt_fenci:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encoder(self, item):
        text = item[0]
        token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_length, truncation=True)["offset_mapping"]
        start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
        #将raw_text的下标 与 token的start和end下标对应
        encoder_txt = self.tokenizer.encode_plus(text, max_length=self.max_length, truncation=True)
        input_ids = encoder_txt["input_ids"]
        token_type_ids = encoder_txt["token_type_ids"]
        attention_mask = encoder_txt["attention_mask"]

        return text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def align_token(self, fenci_idx, start_mapping, end_mapping):
        new_idx = [0]
        for i in range(len(fenci_idx)):
            if i not in start_mapping:
                continue

            new_idx.append(fenci_idx[i])
        new_idx.append(0)
        return new_idx

    def collate(self, examples):
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels, batch_segment_ids = [], [], [], [], []
        batch_fenci_idx = []
        for item in examples:
            raw_text, start_mapping, end_mapping, input_ids, token_type_ids, attention_mask = self.encoder(item)
            fenci_idx = self.align_token(item[1],start_mapping,end_mapping)
            labels = np.zeros((len(EE_label2id), self.max_length, self.max_length))
            for start, end, label in item[2:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[label, start, end] = 1
            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_segment_ids.append(token_type_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
            batch_fenci_idx.append(fenci_idx)
        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_segmentids = torch.tensor(self.sequence_padding(batch_segment_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).long()
        batch_labels_ = torch.zeros((batch_labels.shape[0],batch_labels.shape[1],1))
        batch_fenci_idx = torch.tensor(self.sequence_padding(batch_fenci_idx)).long()
        inputs = {
            "input_ids":batch_inputids,
            "attention_mask":batch_attentionmask,
            "token_type_ids":batch_segmentids,
            "labels": batch_labels,
            "labels_": batch_labels_,
            "fenci_ids": batch_fenci_idx
        }
        return inputs

    def __call__(self, examples):
        return self.collate(examples=examples)


def load_data_fenci(path):
    D = []
    for d in json.load(open(path)):
        D.append([d['text']])
        
        seg_list = jieba.cut(d['text'], cut_all=False) 
        cut_char_idx = np.zeros((len(d['text'])))
        cnt = 0
        for word in seg_list:
            if len(word) == 1:
                cut_char_idx[cnt] = 0 #S
                cnt += 1
            else:
                cut_char_idx[cnt] = 1 #B
                cnt += 1
                for i in range(1, len(word)-1):
                    cut_char_idx[cnt] = 2 #M
                    cnt += 1
                cut_char_idx[cnt] = 3 #E
                cnt += 1
        D[-1].append(cut_char_idx)

        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                D[-1].append((start, end, EE_label2id[label]))
    return D
    

if __name__ == '__main__':
    import os
    from os.path import expanduser
    from transformers import BertTokenizer
    from transformers import BertTokenizerFast

   
    MODEL_NAME = "/dssg/home/acct-stu/stu928/zsy/CMEEE/RoBERTa_zh_Large_PyTorch"
    CBLUE_ROOT = "/dssg/home/acct-stu/stu928/zsy/CMEEE/data/CBLUEDatasets/CMeEE"
    BATCH_SIZE = 4
    data_path = CBLUE_ROOT + "/CMeEE_dev.json"

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    data = load_data(data_path)
    dataset = EntDataset(data=data)
    data_loader = DataLoader(dataset=dataset , 
    batch_size=BATCH_SIZE, 
    collate_fn=CollateForEnt_Chaizi(tokenizer=tokenizer,max_length=100), 
    shuffle=False)
    for i,input in enumerate(data_loader):
        print(input['raw_text'])
        print(input['chaizi_ids'].shape,input['input_ids'].shape)
        break