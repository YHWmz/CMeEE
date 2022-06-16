
from inspect import stack
import json
import logging
import pickle
import re
from itertools import repeat
from os.path import join, exists
from typing import List

import torch
from torch.utils.data import Dataset
import pdb
from lexicon_tree import lexicon_tree
from fastNLP.core import Vocabulary
from transformers import BertModel
import warnings
import pdb
logger = logging.getLogger(__name__)

NER_PAD, NO_ENT = '[PAD]', 'O'

LABEL1 = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'dis', 'bod']  # 按照出现频率从低到高排序
LABEL2 = ['sym']

LABEL  = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'] 

# 标签出现频率映射，从低到高
_LABEL_RANK = {L: i for i, L in enumerate(['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'])}

EE_id2label1 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL1 for P in ("B", "I")]
EE_id2label2 = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL2 for P in ("B", "I")]
EE_id2label  = [NER_PAD, NO_ENT] + [f"{P}-{L}" for L in LABEL  for P in ("B", "I")]

EE_label2id1 = {b: a for a, b in enumerate(EE_id2label1)}
EE_label2id2 = {b: a for a, b in enumerate(EE_id2label2)}
EE_label2id  = {b: a for a, b in enumerate(EE_id2label)}

EE_NUM_LABELS1 = len(EE_id2label1)
EE_NUM_LABELS2 = len(EE_id2label2)
EE_NUM_LABELS  = len(EE_id2label)

class InputExample:
    def __init__(self, sentence_id: str, text: str,entities: List[dict] = None):
        self.sentence_id = sentence_id
        self.text = text
        self.entities = entities

    def to_ner_task(self, lex_tree, for_nested_ner: bool = False):    
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        """这个通过label的位置生成整句话的label
        """
        start_pos, end_pos, text, sen_len = lex_tree.get_lattice(self.text)
        self.text = text

        if self.entities is None:
            return self.sentence_id, self.text, start_pos, end_pos, sen_len
        else:
            if not for_nested_ner:
                label = [NO_ENT] * sen_len
            else:
                label1 = [NO_ENT] * sen_len
                label2 = [NO_ENT] * sen_len

            def _write_label(_label: list, _type: str, _start: int, _end: int):
                for i in range(_start, _end + 1):
                    if i == _start:
                        _label[i] = f"B-{_type}"
                    else:
                        _label[i] = f"I-{_type}"

            for entity in self.entities:
                start_idx = entity["start_idx"]
                end_idx = entity["end_idx"]
                entity_type = entity["type"]

                # assert entity["entity"] == self.text[start_idx: end_idx + 1], f"{entity} mismatch: `{self.text}`"

                if not for_nested_ner:
                    _write_label(label, entity_type, start_idx, end_idx)
                else:
                    if entity_type in LABEL1:
                        _write_label(label1, entity_type, start_idx, end_idx)
                    else:
                        _write_label(label2, entity_type, start_idx, end_idx)

            if not for_nested_ner:
                return self.sentence_id, self.text, label, start_pos, end_pos, sen_len
            else:
                return self.sentence_id, self.text, label1, label2, start_pos, end_pos, sen_len

class EEDataloader:
    def __init__(self, cblue_root: str):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")

    @staticmethod
    def _load_json(filename: str) -> List[dict]:
        with open(filename, encoding="utf8") as f:
            return json.load(f)

    @staticmethod
    def _parse(cmeee_data: List[dict]) -> List[InputExample]:
        return [InputExample(sentence_id=str(i),**data) for i, data in enumerate(cmeee_data)]

    def get_data(self, mode: str):
        if mode not in ("train", "dev", "test"):
            raise ValueError(f"Unrecognized mode: {mode}")
        return self._parse(self._load_json(join(self.data_root, f"CMeEE_{mode}.json")))


class EEDataset(Dataset):
    def __init__(self, cblue_root: str, mode: str, max_length: int, tokenizer, for_nested_ner: bool):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")
        self.max_length = max_length
        self.for_nested_ner = for_nested_ner

        # This flag is used in CRF
        self.no_decode = mode.lower() == "train"

        # Processed data can vary from using different tokenizers
        _tk_class = re.match("<class '.*\.(.*)'>", str(type(tokenizer))).group(1)
        _head = 2 if for_nested_ner else 1
        cache_file = join(self.data_root, f"cache_{mode}_{max_length}_{_tk_class}_{_head}head.pkl")

        if False and exists(cache_file):
            with open(cache_file, "rb") as f:
                self.examples, self.data = pickle.load(f)
            logger.info(f"Load cached data from {cache_file}")
        else:
            self.examples = EEDataloader(cblue_root).get_data(mode) # get original data
            self.data = self._preprocess(self.examples, tokenizer) # preprocess
            with open(cache_file, 'wb') as f:
                pickle.dump((self.examples, self.data), f)
            logger.info(f"Cache data to {cache_file}")

    def _preprocess(self, examples: List[InputExample], tokenizer) -> list:
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        """这个是将text和label转为id
        """
        is_test = examples[0].entities is None
        data = []

        if self.for_nested_ner:
            label2id1 = EE_label2id1
            label2id2 = EE_label2id2
        else:
            label2id = EE_label2id

        for example in examples:
            if is_test:
                sentence_id, text, start_pos, end_pos, sen_len = example.to_ner_task(self.for_nested_ner)
                if self.for_nested_ner:
                    label1 = repeat(None, len(text))
                    label2 = repeat(None, len(text))
                else:
                    label = repeat(None, len(text))
            else:
                if self.for_nested_ner:
                    _sentence_id, text, label1, label2, start_pos, end_pos, sen_len = example.to_ner_task(self.for_nested_ner)
                else:
                    _sentence_id, text, label, start_pos, end_pos, sen_len = example.to_ner_task(self.for_nested_ner)


            tokens = []
            if self.for_nested_ner:
                label_ids1 = None if is_test else []
                label_ids2 = None if is_test else []
            else:
                label_ids = None if is_test else []
            
            if self.for_nested_ner:
                for word, L1, L2 in zip(text, label1, label2):
                    token = tokenizer.tokenize(word)
                    if not token:
                        token = [tokenizer.unk_token]
                    tokens.extend(token)

                    if not is_test:
                        label_ids1.extend([label2id1[L1]] + [tokenizer.pad_token_id] * (len(token) - 1))
                        label_ids2.extend([label2id2[L2]] + [tokenizer.pad_token_id] * (len(token) - 1))
            else:
                for word, L in zip(text, label):
                    token = tokenizer.tokenize(word)
                    if not token:
                        token = [tokenizer.unk_token]
                    tokens.extend(token)

                    if not is_test:
                        label_ids.extend([label2id[L]] + [tokenizer.pad_token_id] * (len(token) - 1))

            
            tokens = [tokenizer.cls_token] + tokens[: self.max_length - 2] + [tokenizer.sep_token]
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            if not is_test:
                if self.for_nested_ner:
                    label_ids1 = [label2id1[NO_ENT]] + label_ids1[: self.max_length - 2] + [label2id1[NO_ENT]]
                    label_ids2 = [label2id2[NO_ENT]] + label_ids2[: self.max_length - 2] + [label2id2[NO_ENT]]
                    data.append((token_ids, label_ids1, label_ids2, start_pos, end_pos, sen_len))
                else:
                    label_ids = [label2id[NO_ENT]] + label_ids[: self.max_length - 2] + [label2id[NO_ENT]]
                    data.append((token_ids, label_ids, start_pos, end_pos, sen_len))
            else:
                data.append((token_ids, start_pos, end_pos, sen_len,))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.no_decode

class FlatDataset(Dataset):
    def __init__(self, cblue_root: str, mode: str, max_length: int, unipath: str, wordpath: str, for_nested_ner: bool, use_bert = False):
        self.cblue_root = cblue_root
        self.data_root = join(cblue_root, "CMeEE")
        self.max_length = max_length
        self.for_nested_ner = for_nested_ner
        self.unipath = unipath
        self.wordpath = wordpath
        self.lexicon_words = []
        self.lexicon_tree = lexicon_tree()
        self.use_bert = use_bert
        self.used_words = []

        # This flag is used in CRF
        self.no_decode = mode.lower() == "train"
        self.uni_vocab = Vocabulary()
        self.word_vocab = Vocabulary()
        self.build_vocab(self.uni_vocab, self.unipath)
        self.build_vocab(self.word_vocab, self.wordpath, self.lexicon_words)
        self.uni_num = len(self.uni_vocab)

        unigram, bigram =  0,0
        for word in self.lexicon_words:
            if len(word) == 1:
                unigram +=1
            if len(word) == 2 :
                bigram += 1
        print(f"unigram num {unigram}, bigram num {bigram}")

        if not use_bert:
            self.lexicon_tree.insert_word(self.lexicon_words)
        print("Finish Vocab Construction~")

        self.examples = EEDataloader(cblue_root).get_data(mode) # get original data
        self.data = self._preprocess(self.examples) # preprocess

    @staticmethod
    def build_vocab(self, vocab, model_path, lexicon_words=None, error='ignore'):
        with open(model_path, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts) == 2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts) - 1
                f.seek(0)
            for idx, line in enumerate(f, start_idx):
                try:
                    parts = line.strip().split()
                    word = ''.join(parts[:-dim])
                    vocab.add_word(word)
                    if lexicon_words is not None:
                        lexicon_words.append(word)
                except Exception as e:
                    if error == 'ignore':
                        warnings.warn("Error occurred at the {} line.".format(idx))
                    else:
                        logger.error("Error occurred at the {} line.".format(idx))
                        raise e

    def _preprocess(self, examples: List[InputExample]) -> list:
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        """这个是将text和label转为id
        """
        is_test = examples[0].entities is None
        data = []

        if self.for_nested_ner:
            label2id1 = EE_label2id1
            label2id2 = EE_label2id2
        else:
            label2id = EE_label2id

        sen_counter = 0

        for example in examples:
            if is_test:
                sentence_id, text, start_pos, end_pos, sen_len = example.to_ner_task(self.lexicon_tree,self.for_nested_ner)
                if self.for_nested_ner:
                    label1 = repeat(None, len(text))
                    label2 = repeat(None, len(text))
                else:
                    label = repeat(None, len(text))
            else:
                if self.for_nested_ner:
                    _sentence_id, text, label1, label2, start_pos, end_pos, sen_len = example.to_ner_task(self.lexicon_tree,self.for_nested_ner)
                else:
                    _sentence_id, text, label, start_pos, end_pos, sen_len = example.to_ner_task(self.lexicon_tree,self.for_nested_ner)


            char_ids = []
            word_ids = []
            if self.for_nested_ner:
                label_ids1 = None if is_test else []
                label_ids2 = None if is_test else []
            else:
                label_ids = None if is_test else []
            
            for i in range(sen_len):
                char_id = self.uni_vocab.to_index(text[i])
                char_ids.append(char_id)

            for i in range(sen_len,len(text)):
                word_id = self.word_vocab.to_index(text[i]) + self.uni_num
                if text[i] not in self.used_words:
                    self.used_words.append(text[i])
                word_ids.append(word_id)

            if self.for_nested_ner:
                for word, L1, L2 in zip(text, label1, label2):
                    if not is_test:
                        label_ids1.extend([label2id1[L1]])
                        label_ids2.extend([label2id2[L2]])
            else:
                for word, L in zip(text,label):
                    if not is_test:
                        label_ids.extend([label2id[L]])

            sentence_ids = char_ids + word_ids
            sentence_ids = sentence_ids[:self.max_length]
            start_pos = start_pos[:self.max_length]
            end_pos = end_pos[:self.max_length]
            sen_len = min(sen_len,self.max_length)
            if len(char_ids) + len(word_ids) <= self.max_length:
                lat_len = len(word_ids)
            else:
                lat_len = max(self.max_length - len(char_ids),0)
            char_ids = torch.LongTensor(char_ids).reshape(1,-1)
            
            if not is_test:
                if self.for_nested_ner:
                    label_ids1 = label_ids1[: self.max_length]
                    label_ids2 = label_ids2[: self.max_length]
                    data.append((sentence_ids, start_pos, end_pos, sen_len, lat_len, label_ids1, label_ids2))
                else:
                    label_ids = label_ids[: self.max_length]
                    data.append((sentence_ids, start_pos, end_pos, sen_len, lat_len, label_ids))
            else:
                data.append((sentence_ids, start_pos, end_pos, sen_len, lat_len))
        
        print("finish loading {} sentences".format(sen_counter))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.no_decode


class CollateFnForEE:
    def __init__(self, pad_token_id: int, label_pad_token_id: int = EE_label2id[NER_PAD], for_nested_ner: bool = False):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.for_nested_ner = for_nested_ner
       
    def __call__(self, batch) -> dict:
        '''NOTE: This function is what you need to modify for Nested NER.
        '''
        """这里是将每一句话进行pad到固定长度
        """
        inputs = [x[0] for x in batch]
        no_decode_flag = batch[0][1]

        input_ids = [x[0] for x in inputs]
        start_pos = [x[1] for x in inputs]
        end_pos = [x[2] for x in inputs]
        sen_len = [x[3] for x in inputs]
        lat_len = [x[4] for x in inputs]

        if self.for_nested_ner:
            labels1 = [x[6]  for x in inputs] if len(inputs[0]) > 6 else None
            labels2 = [x[7]  for x in inputs] if len(inputs[0]) > 7 else None
        else:
            labels  = [x[6]  for x in inputs] if len(inputs[0]) > 6 else None
    
        max_len = max(map(len, input_ids))
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, _ids in enumerate(input_ids):
            attention_mask[i][:sen_len[i]] = 1
            _delta_len = max_len - sen_len[i]
            if not self.for_nested_ner and labels is not None:
                labels[i] += [self.label_pad_token_id] * _delta_len
            elif self.for_nested_ner and labels1 is not None and labels2 is not None:
                labels1[i] += [self.label_pad_token_id] * _delta_len
                labels2[i] += [self.label_pad_token_id] * _delta_len

            _delta_len = max_len - len(_ids)

            input_ids[i] += [self.pad_token_id] * _delta_len
            start_pos[i] += [0] * _delta_len
            end_pos[i] += [0] * _delta_len

        if not self.for_nested_ner:
            inputs = {
                "input_ids": torch.tensor(input_ids, dtype= torch.long),
                "sen_len": torch.tensor(sen_len, dtype=torch.long),
                "lat_len": torch.tensor(lat_len, dtype=torch.long),
                "start_pos": torch.tensor(start_pos, dtype=torch.long),
                "end_pos": torch.tensor(end_pos, dtype=torch.long),
                "attention_mask": attention_mask,
                "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
                "no_decode": no_decode_flag
            }
        else:
            inputs = {
                "input_ids": torch.tensor(input_ids, dtype= torch.long),
                "start_pos": torch.tensor(start_pos, dtype=torch.long),
                "end_pos": torch.tensor(end_pos, dtype=torch.long),
                "lat_len": torch.tensor(lat_len, dtype=torch.long),
                "sen_len": torch.tensor(sen_len, dtype=torch.long),
                "attention_mask": attention_mask,
                "labels": torch.tensor(labels1, dtype=torch.long) if labels1 is not None else None,
                "labels2": torch.tensor(labels2, dtype=torch.long) if labels2 is not None else None,
                "no_decode": no_decode_flag
            }

        return inputs


if __name__ == '__main__':
    # words = ["儿童", "研究", "成人"]
    # lex_tree.insert_word(words)
    # text = "成人在研究儿童"
    # start_pos, end_pos, text, sen_len = lex_tree.get_lattice(text)
    # print(start_pos, end_pos, text, sen_len)

    from os.path import expanduser
    from transformers import BertTokenizer
   
    MODEL_NAME = "../bert-base-chinese"
    CBLUE_ROOT = "../data/CBLUEDatasets"
    unimodel_path = '../pretrain_model/gigaword_chn.all.a2b.uni.ite50.vec'
    bimodel_path = '../pretrain_model/gigaword_chn.all.a2b.bi.ite50.vec'
    wordmodel_path = '../pretrain_model/ctb.50d.vec'

    dataset = FlatDataset(CBLUE_ROOT, mode="train", max_length=100, unipath=unimodel_path, wordpath=wordmodel_path, for_nested_ner=False, use_bert= False)
    

    # tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    # dataset = EEDataset(CBLUE_ROOT, mode="dev", max_length=100, tokenizer=tokenizer, for_nested_ner=True)
    batch = [dataset[0], dataset[1], dataset[2]]
    inputs = CollateFnForEE(pad_token_id=Vocabulary().padding_idx, for_nested_ner=False)(batch)
    print(len(dataset.used_words))

