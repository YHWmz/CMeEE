# CMEEE
## Requirements
```bash
pip install -r requirements.txt
```

## 数据
将下载好的CMeEE_dev.json，CMeEE_test.json, CMeEE_train.json放入./data/CBLUEDatasets/CMeEE中。

## 预训练模型
我们在预训练模型部分分别尝试使用了BERT，RoBERTa，MedBERT和GigaWord，其对应的下载链接如下：

    BERT：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    RoBERTa：https://pan.baidu.com/s/1MRDuVqUROMdSKr6HD9x1mw
    MedBERT：https://huggingface.co/trueto/medbert-base-chinese/tree/main
    GigaWord：https://pan.baidu.com/s/1pLO6T9D

## 模型部分
新使用到的模型为GlobalPointer模型和FLAT模型，对应的模型结构在./src文件夹中

## 优化训练技巧
使用到的对抗训练代码实现在./src/adversarial.py中，随机参数平均和逐层学习率下降优化都是通过对Trainer进行重载后得到，新重载的Trainer在./src/NewTrainer.py文件中

## 训练模型
修改./src/run_cmeee.sh文件中的MODEL_PATH路径为预训练模型所在路径，然后运行
```bash
cd ./src
bash run_cmeee.sh
```

## 测试模型
选择ckpts中需要测试的模型文件（例如ckpts/Roberta_GP_2022/checkpoint-1000/pytorch_model.bin)，然后运行
```bash
cd ./src
python GP_predict.py --MODEL_PATH {预训练模型路径} --CBLUE_ROOT {CMeEE_test.json所在路径} --SAVE_MODEL_PATH {测试模型文件路径}
```
运行完后便会在./src中生成

