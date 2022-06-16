import json
from os.path import join
from typing import Optional

from dataclasses import dataclass, field, asdict


@dataclass
class _Args:
    def to_dict(self):
        return asdict(self)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ModelConstructArgs(_Args):
    model_type: str = field(metadata={"help": "Pretrained model path"})
    head_type: str = field(metadata={"choices": ["linear", "linear_nested", "crf", "crf_nested", "FLAT", "GP"], "help": "Type of head"})
    adv_train: str = field(default="None", metadata={"choices": ["None", "fgm", "pdg"], "help": "Type of Adversarial training"})
    model_path: Optional[str] = field(default=None, metadata={"help": "Pretrained model path"})
    init_model: Optional[int] = field(default=0, metadata={"choices": [0, 1], "help": "Init models' parameters"})
    lr_decay_rate : Optional[float] = field(default=0.9, metadata={"help" : "learning rate layer-wise decay rate, -1 means don`t use layer-wise decay"})
    

@dataclass
class CBLUEDataArgs(_Args):
    cblue_root: str = field(metadata={"help": "CBLUE data root"})
    max_length: Optional[int] = field(default=128, metadata={"help": "Max sequence length"})
    unimodel_path : str = field(default='../pretrain_model/gigaword_chn.all.a2b.uni.ite50.vec', metadata={"help": "CBLUE data root"})
    bimodel_path : str = field(default='../pretrain_model/gigaword_chn.all.a2b.bi.ite50.vec', metadata={"help": "CBLUE data root"})
    wordmodel_path : str = field(default='../pretrain_model/ctb.50d.vec', metadata={"help": "CBLUE data root"})

@dataclass
class FLATConstructArgs(_Args):
    hidden_size : Optional[int] = field(default=200, metadata={"help" : "learning rate layer-wise decay rate, -1 means don`t use layer-wise decay"})
    ff_size : Optional[int] = field(default=800, metadata={"help" : "learning rate layer-wise decay rate, -1 means don`t use layer-wise decay"})
    num_layers : Optional[int] = field(default=8, metadata={"help" : "learning rate layer-wise decay rate, -1 means don`t use layer-wise decay"})
    num_heads : Optional[int] = field(default=1, metadata={"help" : "learning rate layer-wise decay rate, -1 means don`t use layer-wise decay"})
    shared_pos_encoding : Optional[bool] = field(default=True, metadata={"help" : "learning rate layer-wise decay rate, -1 means don`t use layer-wise decay"})

