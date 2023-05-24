import numpy as np
import json
import logging

logger = logging.getLogger('root')

def conver_dataset_to_samples(dataset,max_span_length,ner_label2id=None,context_window=0,split=0):
    """
    split the data into train and dev (for ACE04)
    split == 0: don't split
    split == 1: return first 90% (train)
    split == 2: return last 10% (dev)
    """
    samples = []
    num_ner = 0
    max_len = 0
    max_ner = 0
    num_overlap = 0

    if split == 0:
        data_range = (0,len(dataset))
    elif split == 1:
        data_range = (0,int(len(dataset)*0.9))
    elif split == 2:
        data_range = (int(len(dataset)*0.9),len(dataset))

    for c,doc in enumerate(dataset):  # doc 每一个样本
        if c < data_range[0] or c >= data_range[1]:
            continue
        for i, sent in enumerate(doc):  # 该样本中的每一个句子
            num_ner += len(sent.ner)  # 获取该样本下ner个数
            sample = {
                'doc_key':doc._doc_key,
                'sentence_ix':sent.sentence_ix
            }
            if context_window != 0 and len(sent.text) > context_window:
                logger.info('Long Sentence: {} {}'.format(sample,len(sent.text)))
            sample['tokens'] = sent.text
            sample['sent_length'] = len(sent.text)
            sent_start = 0
            sent_end = len(sample['tokens'])

            # 动态max_len
            max_len = max(max_len,len(sent.text))
            max_ner = max(max_ner,len(sent.ner))

            # 添加上下文信息
            if context_window > 0:
                add_left = (context_window - len(sent.text)) // 2  # 向下取整
                add_right = (context_window - len(sent.text)) - add_left

                # add left context
                j = i - 1  # 根据j索引样本中句子
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]  # 拿到上一个句子的后add_left个token
                    sample['tokens'] = context_to_add + sample['tokens']
                    add_left -= len(context_to_add)  #
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                # add right context
                j += 1
                while j < len(doc) and add_right > 0:  # 从第一个句子添加right context开始
                    context_to_add = doc[j].text[:add_right]  # 拿到下一个句子的前add_right个token
                    sample['tokens'] = sample['tokens'] + context_to_add
                    add_right -= len(context_to_add)  # if add_right > len(doc[j].text), 则从doc[j+1]句子继续截取token
                    j += 1

            sample['sent_start'] = sent_start
            sample['sent_end'] = sent_end
            sample['sent_start_in_doc'] = sent.sentence_start

            sent_ner = {}
            for ner in sent.ner:
                sent_ner[ner.span.span_sent] = ner.label

            # 基于实体排列组合的span，span长度不大于8
            # (0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3,..., (实体句子中起始位置索引，实体在句子中的结束位置索引，实体长度)
            span2id = {}
            sample['spans'] = []
            sample['spans_labels'] = []
            for i in range(len(sent.text)):
                for j in range(i,min(len(sent.text),i+max_span_length)):  # 当循环到句子末尾部分的tokens时，i+max_span_length会大于len(sent.text)
                    sample['spans'].append((i+sent_start,j+sent_start,j-i+1))  # j-i+1：max_span_length中的第几个span
                    span2id[(i,j)] = len(sample['spans']) - 1  # 当前span的长度 [1~8]
                    if (i,j) not in sent_ner:  # # 在（i，i~i+7）之间穷举所有索引（范围不大于max_span_length），判断是否有与实体索引相等的值
                        sample['spans_labels'].append(0)  # 不是实体span标签为0
                    else:
                        sample['spans_labels'].append(ner_label2id[sent_ner[(i,j)]])  # 将文本标签转化为数值型标签
            samples.append(sample)  # 将一个句子的信息添加到samples中，此时不在以一个样本(doc)为单位，将dataset中所有的句子都添加到samples中

    avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])
    logger.info('# Overlap: %d'%num_overlap)
    logger.info('Extracted %d samples from %d documents, with %d NER labels, %.3f avg input length, %d max length'%(len(samples),data_range[1]-data_range[0],num_ner,avg_length,max_length))
    logger.info('Max Length: %d, Max NER: %d'%(max_len,max_ner))
    return samples,num_ner

def batchify(samples,batch_size):
    num_samples = len(samples)
    list_samples_batches = []

    # 如果一个句子太长，则单独成一个batch
    to_single_batch = []
    for i in range(0,len(samples)):
        if len(samples[i]['tokens']) > 350:
            to_single_batch.append(i)

    for i in to_single_batch:
        logger.info('Single batch sample: %s-%d', samples[i]['doc_key'],samples[i]['sentence_ix'])
        list_samples_batches.append([samples[i]])

    samples = [sample for i, sample in enumerate(samples) if i not in to_single_batch]

    for i in range(0,len(samples),batch_size):
        list_samples_batches.append(samples[i:i+batch_size])

    assert(sum([len(batch) for batch in list_samples_batches]) == num_samples)

    return list_samples_batches

class NpEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj,np.integer):
            return int(obj)
        elif isinstance(obj,np.floating):
            return float(obj)
        elif isinstance(obj,np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder,self).default(obj)















