from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import GPT2PreTrainedModel,GPT2Model

class GPT2LMHeadModel(GPT2PreTrainedModel):
    #__xxx__ 是python默认的方法
    def __init__(self,config):
        """
        初始化函数
        :param config: 配置参数
        """
        super().__init__(config)
