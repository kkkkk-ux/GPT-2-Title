"""
data Pretreatment
"""
import re
from functools import partial
from multiprocessing import Pool,cpu_count
from tqdm import tqdm
import json
import random

def clean_weibo_title(title:str):
    """
    Args:
        title: the title of weibo data
    :param title:
    :return:
    """

    # remove ## character(一般为微博的话题标记)
    title=re.sub(r"#","",title)
    # remove [] and content in it.
    #(\[{1,2}) 匹配1-2个\[, dotmatches any character, * matches 0 or more repetitions of the preceding word, ? matches 0 or 1 repetitions
    title=re.sub(r"(\[{1,2})(.*?)(\]{1,2})","",title)
    # 合并标题中过多的空格, + matches 1 or more repetitions
    title =re.sub(r"\s+","",title)
    return title
def clean_weibo_content(content:str):
    """
    clean context in weibo data
    :param content:
    :return:
    """
    #remove url
    content = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "", content)
    # remove excessive blank space
    content=re.sub(r"\s+","",content)
    # remove \u200b character
    content=content.replace("\u200b","")
    return content
def clean_data(sample):
    """
    多线程清理函数
    :param sample: a tuple, includes content and title
    :return:
    """
    (content,title)=sample
    sample=dict()  # new empty dictionary
    sample["title"]=clean_weibo_title(title.strip())  #strip() remove the leading and trailing characters.
    sample["content"]=clean_weibo_title(content.strip())
    return sample

def build_new_data(original_data_path,train_save_path,test_save_path):
    """
    clean weibo data and construct trainset and testset

    :param content_path: the path of content
    :param train_save_path: path of trainning set
    :param test_save_path: path of test set
    :return:
    """
    # open the files and zip them
    data=open(original_data_path,"r",encoding="utf-8")
    # 使用多进程处理数据
    # 最多8个进程
    threads=min(8,cpu_count())
    # Pool() 可以将输入数据分配给不同进程处理, Pool流程池
    with Pool(threads) as p:
        # partial(XXX) run XXX function
        annoate_ = partial(clean_data)
        # chunksize like total work tasks, average allocated to 8 threads
        data=list(tqdm(p.imap(annoate_,data,chunksize=8),desc="build data"))

    # 过滤数据，去除重复、content<100 and title<100的 数据
    data_set=set()
    data_new=[]
    for d in data:
        # 去除重复、content<100 and title<100的 数据
        if d["content"] in data_set or len(d["content"]) <100 or len(d["title"])<2:
            continue
        else:
            data_set.add(d["content"])
            data_new.append(d)
    #分割数据，construct trainning set and test set
    random.shuffle(data_new)  # 乱序data_new list
    train_data=data_new[:-3000]
    test_data=data_new[-3000:]
    fin=open(train_save_path,"w",encoding="uft-8")
    fin.write(json.dumps(train_data,indent=4,ensure_ascii=False))  # write .json file
    fin.close()
    fin = open(test_save_path, "w", encoding="uft-8")
    fin.write(json.dumps(test_data, indent=4, ensure_ascii=False))  # write .json file
    fin.close()

if __name__=='__main__':
    weibo_data_path_dir="data_dir/weibo_data.json"
    train_save_path_dir="data_dir/train_data.json"
    test_save_path_dir="data_dir/test_data.json"
    build_new_data(weibo_data_path_dir,train_save_path_dir,test_save_path_dir)

