
import numpy as np

from typing import List, Union, NamedTuple, Tuple, Counter

from sklearn.metrics import recall_score
from ee_data import EE_label2id, EE_label2id1, EE_label2id2, EE_id2label1, EE_id2label2, EE_id2label, NER_PAD, _LABEL_RANK


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray

class ComputeMetricsForGP:
    def __call__(self, eval_pred) -> dict:
        return {"f1" : -1.0}
class ComputeMetricsForNER: # training_args  `--label_names labels `
    def __call__(self, eval_pred) -> dict:
        predictions, labels = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels[labels == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        #'''NOTE: You need to finish the code of computing f1-score.
        pred_entity = extract_entities(predictions)
        lab_entity = extract_entities(labels)
        TP = 0
        all_ent = 0
        for pred, lab in zip(pred_entity, lab_entity):
            TP += len(set(pred) & set(lab))
            all_ent += (len(pred) + len(lab))
        
        f1 = 2 * TP / all_ent
        #'''

        return { "f1": f1 }


class ComputeMetricsForNestedNER: # training_args  `--label_names labels labels2`
    def __call__(self, eval_pred) -> dict:
        predictions, (labels1, labels2) = eval_pred
        
        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD] # [batch, seq_len, 2]
        labels1[labels1 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        labels2[labels2 == -100] = EE_label2id[NER_PAD] # [batch, seq_len]
        
        # '''NOTE: You need to finish the code of computing f1-score.
        pred_entity = extract_entities(predictions[:,:,0], True, True)
        lab_entity = extract_entities(labels1, True, True)
        TP = 0
        all_ent = 0
        for pred, lab in zip(pred_entity, lab_entity):
            TP += len(set(pred) & set(lab))
            all_ent += (len(pred) + len(lab))
        
        pred_entity = extract_entities(predictions[:,:,1], True, False)
        lab_entity = extract_entities(labels2, True, False)
        for pred, lab in zip(pred_entity, lab_entity):
            TP += len(set(pred) & set(lab))
            all_ent += (len(pred) + len(lab))
            
        f1 = 2 * TP / all_ent
        # '''
        return { "f1": f1 }


def extract_entities(batch_labels_or_preds: np.ndarray, for_nested_ner: bool = False, first_labels: bool = True) -> List[List[tuple]]:
    """
    本评测任务采用严格 Micro-F1作为主评测指标, 即要求预测出的 实体的起始、结束下标，实体类型精准匹配才算预测正确。
    
    Args:
        batch_labels_or_preds: The labels ids or predicted label ids.  
        for_nested_ner:        Whether the input label ids is about Nested NER. 
        first_labels:          Which kind of labels for NestNER.
    """
    batch_labels_or_preds[batch_labels_or_preds == -100] = EE_label2id1[NER_PAD]  # [batch, seq_len]

    if not for_nested_ner:
        id2label = EE_id2label
    else:
        id2label = EE_id2label1 if first_labels else EE_id2label2

    batch_entities = []  # List[List[(start_idx, end_idx, type)]]
    
    # '''NOTE: You need to finish this function of extracting entities for generating results and computing metrics.
    
    # '''
    ids = len(id2label)
    H, W = batch_labels_or_preds.shape
    
    def get_entity(start_idx, finish_idx):
        ent = batch_labels_or_preds[i, start_idx : finish_idx+1]
        ent[0] += 1
        freq = np.bincount(ent, minlength = ids)[::-1]
        max_id = ids - np.argmax(freq) - 1

        label = id2label[max_id].split('-')[1]
        return (start_idx, finish_idx, label)
    
    for i in range(H):
        start_idx = -1
        finish_idx = 0
        entities = []
        for j in range(W):
            if batch_labels_or_preds[i,j] in [0, 1]:
                if start_idx == -1: # 没有需要处理的实体
                    continue
                else: # 实体结束             
                    entities.append(get_entity(start_idx, finish_idx))
                    start_idx = -1
            elif int(batch_labels_or_preds[i,j]) % 2 == 0: # 实体开头
                if start_idx != -1: # 前面有一个实体待处理
                    entities.append(get_entity(start_idx, finish_idx))
                start_idx = j
                finish_idx = j
            else:
                if start_idx != -1: # 存在实体开头
                    finish_idx += 1
        batch_entities.append(entities)
                    
    return batch_entities


if __name__ == '__main__':

    # Test for ComputeMetricsForNER
    predictions = np.load('../test_files/predictions.npy')
    labels = np.load('../test_files/labels.npy')

    metrics = ComputeMetricsForNER()(EvalPrediction(predictions, labels))

    if abs(metrics['f1'] - 0.606179116) < 1e-5:
        print('You passed the test for ComputeMetricsForNER.')
    else:
        print('The result of ComputeMetricsForNER is not right.')
    
    # Test for ComputeMetricsForNestedNER
    predictions = np.load('../test_files/predictions_nested.npy')
    labels1 = np.load('../test_files/labels1_nested.npy')
    labels2 = np.load('../test_files/labels2_nested.npy')

    metrics = ComputeMetricsForNestedNER()(EvalPrediction(predictions, (labels1, labels2)))

    if abs(metrics['f1'] - 0.60333644) < 1e-5:
        print('You passed the test for ComputeMetricsForNestedNER.')
    else:
        print('The result of ComputeMetricsForNestedNER is not right.')
    