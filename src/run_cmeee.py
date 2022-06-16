import json
from lib2to3.pgen2.token import NOTEQUAL
from os.path import join
import pdb
from typing import List
from fastNLP.core import Vocabulary
from sklearn.metrics import precision_recall_fscore_support
from transformers.models.bert.modeling_bert import BertEmbeddings, BertAttention
from transformers import set_seed, BertTokenizer, Trainer, HfArgumentParser, TrainingArguments, BertLayer
from transformers import BertTokenizerFast, BertModel

from args import ModelConstructArgs, CBLUEDataArgs, FLATConstructArgs
from logger import get_logger
from ee_data import EE_label2id2, EEDataset, EE_NUM_LABELS1, EE_NUM_LABELS2, EE_NUM_LABELS, CollateFnForEE, \
    EE_label2id1, NER_PAD, EE_label2id, FlatDataset
from GP_module import GlobalPointer
from GP_dataloader import EntDataset, CollateForEnt, load_data
from model import BertForCRFHeadNER, BertForLinearHeadNER,  BertForLinearHeadNestedNER, BertForCRFHeadNestedNER, Lattice_Transformer
from metrics import ComputeMetricsForNER, ComputeMetricsForNestedNER, extract_entities, ComputeMetricsForGP
from torch.nn import LSTM
from NewTrainer import Trainer_lr_decay
import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"

MODEL_CLASS = {
    'linear': BertForLinearHeadNER, 
    'linear_nested': BertForLinearHeadNestedNER,
    'crf': BertForCRFHeadNER,
    'crf_nested':BertForCRFHeadNestedNER,
    'FLAT': Lattice_Transformer,
    'GP': GlobalPointer,
}

def get_logger_and_args(logger_name: str, _args: List[str] = None):
    parser = HfArgumentParser([TrainingArguments, ModelConstructArgs, CBLUEDataArgs, FLATConstructArgs])
    train_args, model_args, data_args, flat_args = parser.parse_args_into_dataclasses(_args)

    # ===== Get logger =====
    logger = get_logger(logger_name, exp_dir=train_args.logging_dir, rank=train_args.local_rank)
    for _log_name, _logger in logger.manager.loggerDict.items():
        # 在4.6.0版本的transformers中无效
        if _log_name.startswith("transformers.trainer"):
            # Redirect other loggers' output
            _logger.addHandler(logger.handlers[0])

    logger.info(f"==== Train Arguments ==== {train_args.to_json_string()}")
    logger.info(f"==== Model Arguments ==== {model_args.to_json_string()}")
    logger.info(f"==== Data Arguments ==== {data_args.to_json_string()}")
    logger.info(f"==== FLAT Arguments ==== {flat_args.to_json_string()}")

    return logger, train_args, model_args, data_args, flat_args


def get_model_with_tokenizer(model_args, data_args, flat_args):
    model_class = MODEL_CLASS[model_args.head_type]


    if model_class == Lattice_Transformer:
        dropout = {
        'attn' : 0.02, # Attention 层的dropout
        'res_1' : 0.02, # residual 层的dropout
        'res_2' : 0.02, # 因为每个encode模块有两个残差链接
        'ff_1' : 0.02, # FFN层的dropout
        'ff_2' : 0.02, # FFN层的第二个dropout
        }
        model = model_class(flat_args.hidden_size,
                                     flat_args.ff_size,
                                     EE_NUM_LABELS,
                                     flat_args.num_layers,
                                     flat_args.num_heads,
                                     data_args.max_length,
                                     dropout,
                                     flat_args.shared_pos_encoding
                                     )
    elif model_class == GlobalPointer:
        model = model_class(
            encoder = BertModel.from_pretrained(model_args.model_path), 
            ent_type_size = 9, 
            inner_dim = 64,
        )
    else:      
        if 'nested' not in model_args.head_type:
            model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS)
        else:
            model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS1, num_labels2=EE_NUM_LABELS2)
    
    tokenizer = BertTokenizer.from_pretrained(model_args.model_path)
    return model, tokenizer


def generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=False):
    assert len(predictions) == len(test_dataset.examples), \
        f"Length mismatch: predictions({len(predictions)}), test examples({len(test_dataset.examples)})"

    if not for_nested_ner:
        pred_entities1 = extract_entities(predictions[:, 1:], for_nested_ner=False)
        pred_entities2 = [[]] * len(pred_entities1)
    else:
        pred_entities1 = extract_entities(predictions[:, 1:, 0], for_nested_ner=True, first_labels=True)
        pred_entities2 = extract_entities(predictions[:, 1:, 1], for_nested_ner=True, first_labels=False)

    final_answer = []

    for p1, p2, example in zip(pred_entities1, pred_entities2, test_dataset.examples):
        text = example.text
        entities = []
        for start_idx, end_idx, entity_type in p1 + p2:
            entities.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "type": entity_type,
                "entity": text[start_idx: end_idx + 1],
            })
        final_answer.append({"text": text, "entities": entities})

    with open(join(train_args.output_dir, "CMeEE_test.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(f"`CMeEE_test.json` saved")



def main(_args: List[str] = None):
    # ===== Parse arguments =====
    logger, train_args, model_args, data_args, flat_args = get_logger_and_args(__name__, _args)

    # ===== Set random seed =====
    set_seed(train_args.seed)

    # ===== Get models =====
    model, tokenizer = get_model_with_tokenizer(model_args, data_args, flat_args)
    for_nested_ner = 'nested' in model_args.head_type

    lr_decay_rate = model_args.lr_decay_rate

    # ===== Get datasets =====
    if train_args.do_train:
        if isinstance(model, GlobalPointer):
            train_dataset = EntDataset(load_data(data_args.cblue_root + "/CMeEE/CMeEE_train.json"))
            dev_dataset = EntDataset(load_data(data_args.cblue_root + "/CMeEE/CMeEE_dev.json"))
        else:
            train_dataset = FlatDataset(data_args.cblue_root, "train", data_args.max_length, unipath=data_args.unimodel_path, wordpath=data_args.wordmodel_path, for_nested_ner=False)
            dev_dataset = FlatDataset(data_args.cblue_root, "dev", data_args.max_length, unipath=data_args.unimodel_path, wordpath=data_args.wordmodel_path, for_nested_ner=False)
        logger.info(f"Trainset: {len(train_dataset)} samples")
        logger.info(f"Devset: {len(dev_dataset)} samples")
    else:
        train_dataset = dev_dataset = None

    # ===== Trainer =====
    print("begin train")
    if isinstance(model ,GlobalPointer):
        compute_metrics = ComputeMetricsForGP()
    else:
        compute_metrics = ComputeMetricsForNestedNER() if for_nested_ner else ComputeMetricsForNER()

    # ===== Data_Collator =====
    if isinstance(model, GlobalPointer):
        data_collator = CollateForEnt(tokenizer=BertTokenizerFast.from_pretrained(model_args.model_path),max_length=data_args.max_length)
    else:
        data_collator = CollateFnForEE(pad_token_id=Vocabulary().padding_idx, for_nested_ner=False)

    print("This is the model for {}".format(for_nested_ner))
    
    # trainer = Trainer_lr_decay(
    #     model=model,
    #     args=train_args,
    #     data_collator=data_collator,
    #     train_dataset=train_dataset,
    #     eval_dataset=dev_dataset,
    #     compute_metrics=compute_metrics,
    #     lr_decay_rate = lr_decay_rate,
    #     swa = False, # 记得改成参数输入
    # )

    trainer = Trainer(
        model=model,
        tokenizer=BertTokenizerFast.from_pretrained(model_args.model_path),
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )
    
    print(model)
    
    if train_args.do_train:
        try:
            trainer.train()
            # if model_args.adv_train == "None":
            #     trainer.train_swa()
            #     # trainer.train()
            # elif model_args.adv_train == "fgm":
            #     trainer.train_fgm()
            # elif model_args.adv_train == "pdg":
            #     print('*'*100)
            #     trainer.train_pgd()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")

    # trainer.train_swa()

    if train_args.do_predict:
        test_dataset = FlatDataset(data_args.cblue_root, "test", data_args.max_length, unipath=data_args.unimodel_path, wordpath=data_args.wordmodel_path, for_nested_ner=for_nested_ner)
        logger.info(f"Testset: {len(test_dataset)} samples")

        # np.ndarray, None, None
        predictions, _labels, _metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=for_nested_ner)


if __name__ == '__main__':
    main()
