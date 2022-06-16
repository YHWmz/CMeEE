from GP_dataloader import FEATURE_TYPES
from transformers import BertModel, BertTokenizerFast, BertForSequenceClassification
from GP_module import GlobalPointer, GlobalPointer_Chaizi
import json
import torch
import numpy as np
from tqdm import  tqdm

MODEL_NAME = "../RoBERTa_zh_Large_PyTorch"
CBLUE_ROOT = "../data/CBLUEDatasets/CMeEE/CMeEE_test.json"

bert_model_path = MODEL_NAME #your RoBert_large path
save_model_path = '/dssg/home/acct-stu/stu928/zsy/CMEEE/ckpts/bert_Chaizi_2022/checkpoint-18000/pytorch_model.bin'
device = torch.device("cuda:0")

LABEL  = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'] 
EE_id2label  = [ L for L in LABEL]
EE_label2id  = {b: a for a, b in enumerate(EE_id2label)}

max_len = 100
ent2id, id2ent = EE_label2id, {}
for k, v in ent2id.items(): id2ent[v] = k

tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
encoder = BertModel.from_pretrained(bert_model_path)
model = GlobalPointer_Chaizi(encoder, 9 , 64).to(device)
model.load_state_dict(torch.load(save_model_path, map_location='cuda:0'))
model.eval()

from char_featurizer import Featurizer
FEATURE_TYPES = 8

def get_dict(vocab_list):
    return { vocab_list[i]: i+1 for i in range(len(vocab_list)) }

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

def NER_RELATION(text, tokenizer, ner_model,  max_len=100, collator=None):
    assert collator is not None
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=max_len)["offset_mapping"]
    start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
    
    new_span, entities= [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])

    encoder_txt = tokenizer.encode_plus(text, max_length=max_len)
    chaizi_ids = collator.handle_chaizi(text, len(encoder_txt["input_ids"]), start_mapping)
    chaizi_ids = torch.tensor(chaizi_ids).long().unsqueeze(0).cuda()
    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).cuda()
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).cuda()
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).cuda()
    scores = ner_model.get_pred(input_ids, attention_mask, token_type_ids, chaizi_ids)[0].data.cpu().numpy()
    scores[:, [0, -1]] -= np.inf
    scores[:, :, [0, -1]] -= np.inf
    for l, start, end in zip(*np.where(scores > 0)):
        entities.append({"start_idx":new_span[start][0], "end_idx":new_span[end][-1], "type":id2ent[l]})
    return {"text":text, "entities":entities}

if __name__ == '__main__':
    all_ = []
    collator = CollateForEnt_Chaizi(tokenizer=tokenizer, max_length=max_len)
    for d in tqdm(json.load(open(CBLUE_ROOT))):
        all_.append(NER_RELATION(d["text"], tokenizer= tokenizer, ner_model=model, collator = collator))
    json.dump(
        all_,
        open('CMeEE_test.json', 'w'),
        indent=4,
        ensure_ascii=False
    )