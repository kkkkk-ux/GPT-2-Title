import torch
import os
import random
import numpy as np
import argparse
import logging
from transformers import GPT2Config
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
#进度条模块 tqdm
from tqdm import tqdm, trange
from model import GPT2LMHeadModel
from data_set import GPT2NewsTitleDataSet,collate_func
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter

# Do basic configuration for the logging system.
#format:Use the specified format string for the handler.
# datefmt:Use the specified date/time format.
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger=logging.getLogger(__name__)

def train(model,device,train_data,test_data,args):
    """

    :param model:
    :param device: imformation of device
    :param train_data:
    :param test_data:
    :param args: 训练参数配置信息
    :return:
    """

    #SummaryWriter():Writes entries directly to event files in the log_dir to be consumed by TensorBoard.
    tb_write=SummaryWriter()
    # don't allow update model just after 1 batch
    if args.gradient_accumulation_steps<1:
        raise  ValueError("Parameter \'gradient_accumulation_steps\' is invalid, it must >= 1")
    # calculate the truly batch size during trainning
    train_batch_size=int(args.train_batch_size/args.gradient_accumulation_steps)
    train_samper=RandomSampler(train_data)
    train_data_loader=DataLoader(train_data,sampler=train_samper,
                                 batch_size=train_batch_size,collate_fn=collate_func)
    total_steps=int(len(train_data_loader)*args.num_train_epochs/args.gradient_accumulation_steps)
    logger.info("total trainning steps:{}".format(total_steps))
    #将模型加载到相应设备中
    model.to(device)
    # get all the config of model
    param_optimizer=list(model.named_parameters())
    no_decay=['bias','LayerNorm.bias','LayerNorm.weight']
    #any(),()中全为 False返回 False
    # 取出p;从n中取出nd，若nd中一个都不是no_decay的元素，则不取出p,并设置weight_decay 0.01
    # nd中任何一个是no_decay的元素，设置weight_decay 0.0
    optimizer_grouped_parameters=[
        {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay':0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer=AdamW(optimizer_grouped_parameters,
                    lr=args.learning_rate,eps=args.adam_epsilon)
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(args.warmup_proportion*total_steps),
                                              num_training_steps=total_steps)
    # clean the cache of cuda
    torch.cuda.empty_cache()
    # 将模型调至训练状态
    # model 是 GPT2LMHeadModel类，GPT2LMHeadModel父类又是GPT2Model类
    model.train()
    title_id=train_data.title_id
    tr_loss,logging_loss,min_loss=0.0,0.0,0.0
    global_step=0

    # trange():A shortcut for range in tqdm =tqdm(range())
    #desc: Prefix for the progressbar.progressbar:进度条
    for iepoch in trange(0,int(args.num_train_epochs),desc="Epoch",disable=False):
        # iter_bar 对象，用于显示进度条相关
        iter_bar=tqdm(train_data_loader,desc="Iter (loss=X.XXX)",disable=False)
        #enumerate()将可遍历对象(如列表)组合为索引序列，同时列出数据和对应index
        for step,batch in enumerate(iter_bar):
            input_ids=batch["input_ids"].to(device)
            token_type_ids=batch["token_type_ids"].to(device)
            outputs=model.forward(input_ids=input_ids,token_type_ids=token_type_ids,labels=input_ids,title_id=title_id)
            loss=outputs[0]
            # .item() 增加精度
            tr_loss+=loss.item()
            # put loss into processbar, facilitate to obzervation
            iter_bar.set_description("Iter (loss=%5.3f)"%loss.item())
            # gradient accumulation: 多过几个 batches 后总的梯度再更新
            # if accumulation, divide loss by accumulation step
            if args.gradient_accumulation_steps>1:
                loss=loss/args.gradient_accumulation_steps

            loss.backward()
            # in order to prevent gradient explode, clip grad if it is larger than a threshold
            torch.nn.utils.clip_grad_norm(model.parameters(),args.max_grad_norm)

            # 梯度累积步数的整数更新model
            if(step+1)%args.gradient_accumulation_steps==0:
                optimizer.step()
                scheduler.step()
                # reset grad to zero for accumulation
                optimizer.zero_grad()
                global_step+=1
                # 1 sequence's loss; loss: average of 1 batch's loss; tr_loss: average of 1 accumulation loss; train_loss: average of 1 logging loss;
                #1 logging step=20 accumulation steps, 1 accumulation step(global step) =4 batches, 1 batch=16 sequences
                if args.logging_steps>0 and global_step%args.logging_steps:
                    tb_write.add_scalar("lr",scheduler.get_lr()[0],global_step)
                    #todo: 有问题，(tr_loss-logging_loss)=0只能说模型梯度为零，不能说明loss小
                    tb_write.add_scalar("train_loss",(tr_loss-logging_loss)/
                                        (args.logging_steps*args.gradient_accumulation_steps),global_step)
                    logging_loss=tr_loss
                # every 4*eval_steps batches test test set
                if args.eval_steps>0 and global_step%args.eval_steps==0:
                    eval_loss=evaluate(model,device,test_data,args)
                    tb_write.add_scalar("test_loss",eval_loss,global_step)

        # after an epoch(all sequences have put into model)
        outputs_dir=os.path.join(args.output_dir,"checkopint-{}".format(global_step))
        # hasattr(): Return whether the object has an attribute with the given name. attribute: variable in class.
        # GPT2 是多个一样的module组成，能存module就存，省空间
        model_to_save=model.module if hasattr(model,"module") else model
        model_to_save.save_pretrained(outputs_dir)
        # clean cache of cuda of another epoch
        torch.cuda.empty_cache()

def evaluate(model,device,test_data,args):
    """
    test test set
    :param model:
    :param device:
    :param test_data:
    :param args:
    :return:
    """
    test_sampler=SequentialSampler(test_data)
    test_data_loader=DataLoader(test_data,sampler=test_sampler,
                                batch_size=args.test_batch_size,collate_fn=collate_func)
    iter_bar=tqdm(test_data_loader,desc="Iter",disable=False)
    title_id=test_data.title_id
    total_loss,total=0.0,0.0
    for step,batch in enumerate(iter_bar):
        # set model to the test state
        model.eval()
        with torch.no_grad():
            input_ids=batch["input_ids"].to(device)
            token_type_ids=batch["token_type_ids"].to(device)
            outputs=model.forward(input_ids=input_ids,token_type_ids=token_type_ids,
                                  labels=input_ids,title_id=title_id)
            loss=outputs[0]
            #1 batch's loss
            loss=loss.item()
            # lots batches loss
            total_loss+=loss*len(batch["input_ids"])
            # number of sequences
            total+=len(batch["input_ids"])
    test_loss=total_loss/total
    return test_loss


def set_args():
    #ArgumentParser 用于将命令行字符串解析为Python对象。
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='0',type=str,help='设置训练或者测试时使用的显卡')
    parser.add_argument('--config_path',default='./config/config.json',type=str,help='从config.json中提取模型参数配置信息')
    parser.add_argument('--vocab_path', default='./vocab/vocab.txt', type=str, help='词表，小型，增加了一些新标记')
    parser.add_argument('--train_file_path', default='./data_dir/weibo_data.json', type=str)
    parser.add_argument('--test_file_path', default='./data_dir/weibo_data.json', type=str)
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='预训练GPT2模型路径')
    parser.add_argument('--data_dir', default='./data_dir', type=str)
    parser.add_argument('--num_train_epochs', default=5, type=int)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--test_batch_size', default=8, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    # 当训练过一段时间，模型看过所有数据有了先验知识，加大学习率可以提升收敛速度
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='训练总步长的百分之多少进行warm up')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--logging_steps', default=20, type=int, help='训练多少步保存一次日志')
    parser.add_argument('--eval_steps', default=4000, type=int, help='训练时，多少步测试一次')
    # during training, takes 16 sequences once as input, do it 4 times to accumulate grad
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--output_dir', default='./output_dir/', type=str)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--max_len', default=512, type=int, help='输入模型的sequence的最大长度,<n_ctx in config.json')
    parser.add_argument('--title_max_len', default=32, type=int, help='the max length of title generated')
    return parser.parse_args()

def main():
    # set the trainning model's parameters
    args=set_args()
    # 设置显卡信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    # get the imformation of divce, use to train model
    device=torch.device("cuda" if torch.cuda.is_available() and int(args.device)>=0 else "cpu")

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # load the config.json
    model_config=GPT2Config.from_json_file(args.config_path)


    if args.pretrained_model_path:
        model=GPT2LMHeadModel.from_pretrained(args.pretrained_model_path)
    # 没有6层的预训练模型，所以从随机初始化开始训练
    else:
        model=GPT2LMHeadModel(config=model_config)

    # 实例化tokenizer, tokenizer 包含vocabulary的信息
    tokenizer=BertTokenizer.from_pretrained(args.vocab_path,do_lower_case=True)
    #"我爱[Space]中国。"，使用原始tokenizer分词结果为
    #"['我', '爱', '[', 'Space', ']', '中', '国', '。']";
    #增加分割符号后的结果为"['我', '爱', '[Space]', '中', '国', '。']"
    tokenizer.add_tokens("[Space]",special_tokens=True)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


    train_data=GPT2NewsTitleDataSet(tokenizer,args.max_len,args.title_max_len,args.data_dir,"train",args.train_file_path)
    test_data=GPT2NewsTitleDataSet(tokenizer,args.max_len,args.title_max_len,args.data_dir,"test",args.test_file_path)
    train(model,device,train_data,test_data,args)
if __name__ == '__main__':
    main()