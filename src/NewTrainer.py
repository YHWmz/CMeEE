from typing_extensions import runtime
from transformers import Trainer
from torch import nn
import torch
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader,IterableDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


class Trainer_lr_decay(Trainer):
    def __init__(self, 
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks  = None,
        optimizers = (None, None),
        lr_decay_rate = None,
        swa = False,
        swa_update_epoches = 50) :
        super(Trainer_lr_decay, self).__init__(model,args ,
                                                data_collator,
                                                train_dataset,
                                                eval_dataset,
                                                tokenizer,
                                                model_init,
                                                compute_metrics,
                                                callbacks,
                                                optimizers)
        # if Trainer.__version__ != '4.11.0':
        #     raise RuntimeError(f"要使用Trainer版本为4.11.0,当前版本为{Trainer.__version__}")

        self.lr_decay_rate = lr_decay_rate # 假如decay_rate为-1，则就不是用lr_layer_size_decay

        self.swa = swa # 如果是True，则使用参数平均
        self.swa_update_epoches = swa_update_epoches # 参数平均的训练轮数
        self.swa_scheduler = None
        self.swa_model = None
   
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method (or :obj:`create_optimizer`
        and/or :obj:`create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)
        self.create_swa_scheduler(self.optimizer, self.model)
        print("完成创建优化器和scheduler")

    def create_optimizer(self):
        """
        Override
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        assert self.lr_decay_rate < 1, "lr decay rate 应该在0,1之间，或者是-1"

        if self.lr_decay_rate == -1:
            print("没有使用layer-wise lr decay")
        else:
            print(f"使用了layer-wise lr decay ， decay rate 是 {self.lr_decay_rate}")

        print(self.args.device)

        lr_decay_rate = 1 if self.lr_decay_rate == -1 else self.lr_decay_rate

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # 先找到逐层的名字，后面按照层的类型设置lr
            layer_names = []
            for name, _ in self.model.named_parameters():
                layer_names.append(name)

            # 越底层的layer学习率要越大
            layer_names.reverse()

            optimizer_grouped_parameters = []
            prev_name = layer_names[0].split('.')[0]
            lr = self.args.learning_rate

            for name in layer_names:
                cur_name = name.split('.')[0]

                if cur_name != prev_name:
                    lr *= lr_decay_rate
                prev_name = cur_name
                
                if name in decay_parameters:
                    weight_decay = self.args.weight_decay
                else:
                    weight_decay = 0.0
                optimizer_grouped_parameters += [{'params': 
                                    [p for n, p in opt_model.named_parameters() if n == name and p.requires_grad], 
                                'lr': lr,
                                "weight_decay": weight_decay,
                                }]

            optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }

            self.optimizer = torch.optim.AdamW(
                    params=optimizer_grouped_parameters,
                    **optimizer_kwargs)
        return self.optimizer

    def create_swa_scheduler(self, optimizer: torch.optim.Optimizer = None, model : torch.nn.Module = None):
        """ 创建stochatic weight average scheduler
        """
        if self.swa and self.swa_scheduler is None:
            self.swa_model = AveragedModel(model)
            self.swa_scheduler = SWALR(optimizer, swa_lr=0.05)


    def train_swa(self):
        if not self.swa:
            self.swa_model = self.swa
            return self.model

        train_dataloader = self.get_train_dataloader()

        args = self.args

        for epoch in range(self.swa_update_epoches):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDataset):
                train_dataloader.dataset.set_epoch(epoch)
            self.optimizer.zero_grad()
            
            tr_loss = torch.tensor(0.0).to(args.device)
            for step, input in tqdm(enumerate(train_dataloader)):
                loss = self.compute_loss(self.model, input)
                loss.backward()
                tr_loss += loss.detach()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.swa_model.update_parameters(self.model)
                    self.swa_scheduler.step()
                    self.optimizer.zero_grad()
            tr_loss /= step
            print(f"train loss is {tr_loss}")

        # Update bn statistics for the swa_model at the end
        torch.optim.swa_utils.update_bn(train_dataloader, self.swa_model)
        # Use swa_model to make predictions on test data 
        return self.swa_model

    def set_model_as_swa_model(self):
        """将模型设置为参数平均后的模型，方便测试
        """
        self.model = self.swa_model
