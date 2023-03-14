import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import logging
from torch.nn.utils.rnn import pad_sequence

logger=logging.getLogger(__name__)  #从__name__开始日志记录

class GPT2NewsTitleDataSet(Dataset):
    """模型所需要的数据类"""
    # __init__ runs when class is Instantiate (实例化) into object
    def __init__(self,tokenizer,max_len,title_max_len,data_dir,data_set_name,path_file=None,is_overwrite=None):
        """

        :param tokenizer: 分词器
        :param max_len: 数据最大长度
        :param title_max_len: generated title's max length
        :param data_dir: 缓存路径
        :param data_set_name: data set name
        :param path_file: original data file path
        :param is_overwrite: 是否重新生成缓存文件
        """
        self.tokenizer=tokenizer
        # convert token to corresponding one-hot index of vacabulary
        self.content_id=self.tokenizer.convert_tokens_to_ids("[Content]")
        self.title_id=self.tokenizer.convert_tokens_to_ids("[Title]")
        # 为了保留标题中的空格，也对空格保留训练
        self.space_id=self.tokenizer.convert_tokens_to_ids("[Space]")
        self.max_len=max_len
        self.title_max_len=title_max_len
        cached_feature_file=os.path.join(data_dir,"cached_{}_{}".format(data_set_name,max_len))
        # 若缓存文件存在且不可重新生成
        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info("cache file {} aready exists, load it directly.".format(cached_feature_file))
            self.data_set=torch.load(cached_feature_file)["data_set"]
        else:
            logger.info("cache file {} do not exist, do the pretreatment.".format(cached_feature_file))
            self.data_set=self.load_data(path_file)
            logger.info("data pretreatment finished, saved into {} for cache".format(cached_feature_file))
            #Saves an object to a disk file.
            torch.save({"data_set":self.data_set},cached_feature_file)

    def load_data(self,path_file):
        """
        load original data, pretreat it
        :param self:
        :param path_file: original data path
        :return:
        """
        self.data_set=[]
        with open(path_file,"r",encoding="utf-8") as fh:
            data=json.load(fh)
            # enumerate is a function that add counter into iterable. idx come from enumerate.
            for idx,sample in enumerate(tqdm(data,desc="iter",disable=False)):
                # 使用 convert_feature 函数 对 正文和标题 索引化， 生成模型训练所需格式
                #input_ids 代表one hot 的 index， token_type_ids:different lists corresponds to different sequence, each list has 0 or 1 which means the mask on corresponding token
                input_ids,token_type_ids=self.convert_feature(sample)
                self.data_set.append({"input_ids":input_ids,"token_type_ids":token_type_ids})
        return self.data_set

    def convert_feature(self,sample):
        """
        data treatment function
        sample: a dic, contains context and title, format:{"content":content,"title":title}
        """
        input_ids=[]
        token_type_ids=[]
        # 对content 进行分词
        content_tokens=self.tokenizer.tokenize(sample["content"])
        # 对title 进行分词, 并把空格替换为表示空格的字符
        title_tokens = self.tokenizer.tokenize(sample["title"].replace(" ","[Space]"))
        # if title is longer than generated title max length, cut it
        if len(title_tokens) > self.title_max_len:
            title_tokens=title_tokens[:self.title_max_len]
        # if content is too long, cut it
        if len(content_tokens) > self.max_len-len(title_tokens)-3:
            content_tokens=content_tokens[:self.max_len-len(title_tokens)-3]

        # generate the input_ids
        # the class of input ids, eg:int, float
        input_ids.append(self.tokenizer.cls_token_id)
        # add the truly input ids(index of one hot list). append add one element, extend can add a list.
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(content_tokens))
        # add the separator token id
        input_ids.append(self.tokenizer.sep_token_id)
        # add the truly input ids of title
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(title_tokens))
        # add the separator token id again.
        input_ids.append(self.tokenizer.sep_token_id)

        # 这边有点奇怪，照理说 token_type_ids 是为了标示不同句子，这里却append content_id=convert_tokens_to_ids("[Content]")
        # input_ids---convert_tokens_to_ids(content_tokens)局部，token_type_ids---content_id = convert_tokens_to_ids("[Content]")全局
        # generate the token_type_ids, prefix part as 1 and 0 for rest of the token. 有多个句子的content_id，其中convert_tokens_to_ids会自动生成token_type_ids
        token_type_ids.append(self.content_id)
        # content_tokens: a list of tokens, content_id: a list of index of one-hot
        token_type_ids.extend([self.content_id]*len(content_tokens))
        token_type_ids.append(self.content_id)
        token_type_ids.extend([self.title_id]*len(title_tokens))
        token_type_ids.append(self.title_id)

        assert len(input_ids)==len(token_type_ids)
        assert len(input_ids)<=self.max_len

        return input_ids,token_type_ids

    # 定义特殊方法 len(特定对象)=len(对象.data_set), 与原始len()不同
    def __len__(self): return len(self.data_set)
    def __getitem__(self, item): return self.data_set[item]

def collate_func(batch_data):
    """
    将数据处理成tensor形式
    :param batch_data:
    :return:
    """
    batch_size=len(batch_data)
    # if batch_size=0, return empty dic
    if batch_size==0: return {}
    input_ids_list,token_type_ids_list=[],[]
    for instance in batch_data:
        # 按照batch data 中最长的sequence, padding data(填充不够长的数据)
        input_ids_temp=instance["input_ids"]
        token_type_ids_temp=instance["token_type_ids"]
        #转换为tensor
        input_ids_list.append(torch.tensor(input_ids_temp,dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp,dtype=torch.long))

    # use pad_sequance to do padding
    return {"input_ids":pad_sequence(input_ids_list,batch_first=True,padding_value=0),
            "token_type_ids":pad_sequence(token_type_ids_list,batch_first=True,padding_value=0)}
