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
        # __init__ 初始化自己， super().__init__ 调用父类的初始化
        super().__init__(config)
        # transformer 是 GPT2Model 类
        self.transformer=GPT2Model(config)
        # 调用config.json
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.init_weights()
    def forward(self,input_ids=None,past=None,token_type_ids=None,labels=None,title_id=None):
        """
        forward propagation
        :param input_ids: the index of one-hot for each token, size:[batch_size,sequence_length]
        :param past: 包含由模型预先计算好的隐藏状态，防止重复计算token，加速顺序解码，一般使用在预测阶段
        :param token_type_ids: list of lists, use to identify content and title of sequence, size:[batch_size,sequence_length]
        :param labels: 标签序列，一般与 input_ids 相同。每一行是不同sequence
        :param title_id: the index of title of one-hot
        :return:
        """

        # get the output of GPT2
        transformers_output=self.transformer(input_ids,token_type_ids=token_type_ids)
        # get the state of latest layer of hidden layers
        hidden_states=transformers_output[0]
        # after input tokens go through GPT2 one-by-one, the output the predicted tokens one-by-one
        lm_logits=self.lm_head(hidden_states)
        # 拼接输出结果，但不太懂为什么这样写
        #todo
        outputs=(lm_logits,)+transformers_output[1:]
        if labels is not None:
            # when calculate loss, title_id can't be None, because titile_id need to find corresponding title
            if title_id is None or token_type_ids is None:
                raise  Exception("when label not None, title_id and token_type_ids need to be not None,too")
            # token_type_ids中等于title_id的部分需要计算loss，标记为1; content 不需要，为0
            mask=(token_type_ids==title_id).long()
            labels=labels*mask
            # 对预测结果lm_logits进行翻转，它是从最后一个预测出的token往第一个排，而原始labels(等于input_ids)不是
            # [...,XXX,XXX] 从第二、三个维度取
            # 不加contiguous(), shift_logits改变会影响lm_logits
            shift_logits=lm_logits[...,:-1,:].contiguous()
            shift_labels=labels[...,1:].contiguous()

            # ignore the loss that shift_labels=0, sum each token of loss to optimize
            loss_fct=CrossEntropyLoss(ignore_index=0,reduction="sum")
            #torch 中 view() 相当于 numpy 的 reshape
            loss=loss_fct(shift_logits.view(-1,shift_logits.size(-1)),shift_labels.view(-1))
            # the length of title
            #ne(0) 相当于 boolean(shift_labels!=0), 一个矩阵
            #使用item()取出的元素值精度更高
            num=shift_labels.ne(0).long().sum().item()
            loss=loss/num
            outputs=(loss,)+outputs
        return outputs
