import sys
import argparse
import logging
import os
import random
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset
from torch.nn import CrossEntropyLoss
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME,CONFIG_NAME
from transformers import AutoTokenizer,AdamW,get_linear_schedule_with_warmup
from shared.const import task_ner_labels,task_rel_labels
from relation.utils import decode_sample_id
from relation.utils import generate_relation_data
from relation.models import BertForRelation
CLS = '[CLS]'
SEP = '[SEP]'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data"""
    def  __init__(self, input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sub_idx = sub_idx
        self.obj_idx = obj_idx

def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def add_marker_tokens(tokenizer,ner_labels):
    new_tokens = ['<SUBJ_START>','<SUBJ_END>','<OBJ_START>','<OBJ_END>']
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>'%label)
        new_tokens.append('<SUBJ_END=%s>' % label)
        new_tokens.append('<OBJ_START=%s>' % label)
        new_tokens.append('<OBJ_END=%s>' % label)
    for label in ner_labels:
        new_tokens.append('<SUBJ=%s>'%label)
        new_tokens.append('<OBJ=%s>'%label)
    tokenizer.add_tokens(new_tokens)  # 将额外标记符加入到tokenizer中
    logger.info('# vocal after adding marker: %d'%len(tokenizer))

def convert_examples_to_features(examples,label2id,max_seq_len,tokenizer,special_tokens,unused_tokens=True):
    '''
    loads a data file into a list of 'InputBatch's,
    unused_tokens: whether use [unused1] [unused2] as special tokens
    '''
    def get_special_token(w):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = "[unused%d]"%(len(special_tokens) + 1)  # special_tokens[w]: [unused%d]
            else:
                special_tokens[w] = ('<' + w + '>').lower()  # <SUBJ_START=TYPE>
        return special_tokens[w]

    num_tokens = 0
    max_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in tqdm(enumerate(examples),desc='Converting examples into features ...'):
        tokens = [CLS]  # ['[CLS]']

        # 拿到句子中 主客体的实体类型，
        SUBJECT_START_NER = get_special_token("SUBJ_START=%s"%example['subj_type'])  # SUBJ_START=TYPE  主体marker
        SUBJECT_END_NER = get_special_token("SUBJ_END=%s" % example['subj_type'])
        OBJECT_START_NER = get_special_token("OBJ_START=%s" % example['obj_type'])
        OBJECT_END_NER = get_special_token("OBJ_END=%s" % example['obj_type'])

        # 给实体添加marker，并且进行subword分词
        for i, token in enumerate(example['tokens']):
            if i == example['subj_start']:  # 第i个token是不是主体开始
                sub_idx = len(tokens)
                tokens.append(SUBJECT_START_NER)  # 添加主体标记符marker
            if i == example['obj_start']:  # 第i个token是不是客体开始
                obj_idx = len(tokens)
                tokens.append(OBJECT_START_NER)  # 添加客体标记符marker

            for sub_token in tokenizer.tokenize(token):
                tokens.append(sub_token)

            if i == example['subj_end']:
                tokens.append(SUBJECT_END_NER)
            if i == example['obj_end']:
                tokens.append(OBJECT_END_NER)
        tokens.append(SEP)

        num_tokens += len(tokens)
        max_tokens = max(num_tokens,len(tokens))

        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
            if sub_idx >= max_seq_len:
                sub_idx = 0
            if obj_idx >= max_seq_len:
                obj_idx = 0
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(tokens)
        padding = [0] * (max_seq_len - len(input_ids))
        # padding
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        label_id = label2id[example['relation']]

        assert max_seq_len == len(input_ids)
        assert max_seq_len == len(input_mask)
        assert max_seq_len == len(segment_ids)

        if num_shown_examples < 20:
            if (ex_index < 5) or (label_id > 0):  # no_relation : 0
                num_shown_examples += 1
                logger.info('*** Example ***')
                logger.info('guid: %s'%(example['id']))
                logger.info('tokens %s'%' '.join(
                    [str(x) for x in tokens]))
                logger.info('input_ids: %s'% ' '.join([str(x) for x in input_ids]))
                logger.info('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
                logger.info('segment_ids: %s' % ' '.join([str(x) for x in segment_ids]))
                logger.info('label: %s (id = %d)' % (example['relation'],label_id))
                logger.info('sub_idx, obj_idx: %d, %d'%(sub_idx, obj_idx))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          sub_idx=sub_idx,  # 记录该句子中 主体开始位置索引
                          obj_idx=obj_idx))  # 记录该句子中 客体开始位置索引
    logger.info("Average #tokens: %.2f" %(num_tokens * 1.0 / len(examples)))
    logger.info('Max #tokens: %d'%max_tokens)
    logger.info('%d (%.2f %%) examples can fit max_seq_len = %d'%(num_fit_examples,
                                                                  num_fit_examples*100.0 / len(examples), max_seq_len))

    return features

def compute_f1(preds,labels,e2e_ngold):
    n_gold = n_pred = n_correct = 0
    for pred,label in zip(preds,labels):
        if pred != 0:
            n_pred += 1  # 模型给出了预测值，但不一定对
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1  # 除了no_relaton标签外，有预测且预测正确

    if n_correct == 0:
        return {'Precision':0.0, 'Recall':0.0, 'F1':0.0}
    else:
        prec = n_correct*1.0 / n_pred  # 预测正确的占预测的比例
        recall = n_correct*1.0 / n_gold  # 预测正确的占标签的比例
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 = 0.0

    if e2e_ngold is not None:
        e2e_recall = n_correct*1.0 / e2e_ngold
        e2e_f1 = 2.0 * prec *e2e_recall / (prec + e2e_recall)
    else:
        e2e_recall = e2e_f1 = 0.0

    return {'Precision':prec, 'Recall':e2e_recall,'Task_recall':recall,'F1':e2e_f1,'Task_F1':f1,
            'n_correct':n_correct,'n_pred':n_pred,'n_gold':e2e_ngold,'task_ngold':n_gold}

def simple_accuracy(preds,labels):
    return (preds == labels).mean()

def evaluate(model,device,eval_dataloader,eval_label_ids,num_labels,e2e_ngold=None,verbose=True):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        sub_idx = sub_idx.to(device)
        obj_idx = obj_idx.to(device)
        with torch.no_grad():
            logits = model(input_ids,segment_ids,input_mask,labels=None,sub_idx=sub_idx,obj_idx=obj_idx)
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1,num_labels),label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0],logits.detach().cpu().numpy(),axis=0)  # ???????

    eval_loss = eval_loss / nb_eval_steps
    logits = preds[0]
    preds = np.argmax(preds[0],axis=1)
    result = compute_f1(preds,eval_label_ids.numpy(),e2e_ngold=e2e_ngold)
    result['Accuracy'] = simple_accuracy(preds,eval_label_ids.numpy())
    result['Eval_loss'] = eval_loss
    if verbose:
        logger.info('**** Eval results ****')
        for key in sorted(result.keys()):
            logger.info('   %s = %s',key,str(result[key]))
    return preds,result,logits

def save_trained_model(output_dir,model,tokenizer):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger.info('Saving model to %s'%output_dir)
    # 保存模型
    model_to_save = model.module if hasattr(model,' module') else model  # hasattr: model中是否有module
    output_model_file = os.path.join(output_dir,WEIGHTS_NAME)  # 模型保存路径
    output_config_file = os.path.join(output_dir,CONFIG_NAME)
    torch.save(model_to_save.state_dict(),output_model_file)
    output_model_file.config.to_json_file(output_config_file)
    # 保存tokenizer
    tokenizer.save_vocabulary(output_dir)

def print_pred_json(eval_data,eval_examples,preds,id2label,output_file):
    rels = dict()
    for ex, pred in zip(eval_examples,preds):
        doc_sent, sub, obj = decode_sample_id(ex['id'])
        if doc_sent not in rels:
            rels[doc_sent] = []
        if pred != 0:
            rels[doc_sent].append([sub[0],sub[1],obj[0],obj[1],id2label[pred]])

    js = eval_data.js
    for doc in js:
        doc['predicted_relations'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d'%(doc['doc_key'],sid)
            doc['predicted_relations'].append(rels.get(k, []))

    logger.info('Output predictions to %s...'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in js))

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    n_gpu = torch.cuda.device_count()

    if args.do_train:
        train_dataset,train_samples,train_nrel = generate_relation_data(args.train_file,use_gold=True,context_window=args.context_window)
    if (args.do_eval and args.do_train) or (args.do_eval and not(args.eval_test)):
        eval_dataset, eval_samples, eval_nrel = generate_relation_data(os.path.join(args.entity_output_dir,args.entity_predictions_dev),use_gold=args.eval_with_gold,context_window=args.context_window)
    if args.eval_test:
        test_dataset, test_samples, test_nrel = generate_relation_data(os.path.join(args.entity_output_dir,args.entity_predictions_test),use_gold=args.eval_with_gold,context_window=args.context_window)

    setseed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError('At least one of ‘do_train’ or ‘do_eval’ must be true ')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, 'train.log'),'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, 'eval.log'),'w'))
    logger.info(sys.argv)
    logger.info(args)
    logger.info('device: {}, n_gpu: {}'.format(device,n_gpu))

    if os.path.exists(os.path.join(args.output_dir,'label_list.json')):
        with open(os.path.join(args.output_dir,'label_list.json'), 'r') as f:
            label_list = json.load(f)
    else:
        label_list = [args.negative_label] + task_rel_labels[args.task]
        with open(os.path.join(args.output_dir, 'label_list.json'), 'w') as f:
            json.dump(label_list,f)

    label2id = {label:i for i, label in enumerate(label_list)}
    id2label = {i:label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model,do_lower_case=args.do_lower_case)
    if args.add_new_tokens:
        add_marker_tokens(tokenizer,task_ner_labels[args.task])

    if os.path.exists(os.path.join(args.output_dir, 'special_tokens.json')):
        with open(os.path.join(args.output_dir,'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}

    if args.do_eval and (args.do_train or not(args.eval_test)):
        eval_features = convert_examples_to_features(eval_samples,label2id,args.max_seq_len,tokenizer,special_tokens,unused_tokens=args.add_new_tokens)
        logger.info('**** Dev *****')
        logger.info(' Num examples = %d', len(eval_samples))
        logger.info(' Batch size = %d', args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features],dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)  # 拿到所有样本的 主体位置索引
        all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)  # 拿到所有样本的 客体位置索引
        eval_data = TensorDataset(all_input_ids,all_input_mask,all_segment_ids,all_label_ids,all_sub_idx,all_obj_idx)  # 将（）内数据进行Tenosr类型的zip
        eval_dataloader = DataLoader(eval_data,batch_size=args.eval_batch_size)
        eval_label_ids = all_label_ids

    with open(os.path.join(args.output_dir, 'special_tokens.json'), 'w') as f:
        json.dump(special_tokens, f)

    if args.do_train:
        train_features = convert_examples_to_features(train_samples,label2id,args.max_seq_len,tokenizer,special_tokens,args.add_new_tokens)
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key= lambda f:np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in train_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx,all_obj_idx)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]  # 升维吗？

        num_train_optimization_steps =len(train_dataloader) * args.num_train_epochs

        logger.info('**** Training ****')
        logger.info(' Num examples = %d', len(train_samples))
        logger.info(' Batch size = %d', args.train_batch_size)
        logger.info(' Num steps = %d', args.num_train_epochs)

        best_result = None
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)

        model = BertForRelation.from_pretrained(args.model,cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),num_rel_labels=num_labels)
        model.bert.resize_token_embeddings(len(tokenizer))  # 修改token 的embedding的维度

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params':[param for name, param in param_optimizer
                    if not any(nd in name for nd in no_decay)],'weight_decay':0.1},
            {'params': [param for name, param in param_optimizer
                    if any(nd in name for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,correct_bias=args.bertadam)
        scheduler = get_linear_schedule_with_warmup(optimizer,int(num_train_optimization_steps*args.warmup_proportion),num_train_optimization_steps)

        start_time = time.time()
        global_step = 0
        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        for epoch in tqdm(range(args.num_train_epochs)):
            model.train()
            logger.info('Start epoch #{} (lr = {})...'.format(epoch,args.learning_rate))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)
            for step, batch in tqdm(enumerate(train_batches)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, sub_idx,obj_idx = batch
                loss = model(input_ids,segment_ids,input_mask,label_ids,sub_idx,obj_idx)
                if n_gpu > 1:
                    loss = loss.mean()

                loss.backward()
                tr_loss += loss
                nb_tr_steps += 1
                nb_tr_examples += input_ids.shape[0]

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if (step + 1) % eval_step == 0:
                    logger.info('Epoch: {}, Step: {} / {}, Used Time = {:.2f}s, loss = {:.6f}'.format(
                        epoch, step + 1, len(train_batches), time.time()-start_time, tr_loss/nb_tr_steps))

            if args.do_eval:
                preds, result, logits = evaluate(model,device,eval_dataloader,eval_label_ids,num_labels,e2e_ngold=eval_nrel)
                result['global_step'] = global_step
                result['epoch'] = epoch
                result['learning_rate'] = args.learning_rate
                result['batch_size'] = args.train_batch_size

                if best_result is None or result[args.eval_metric] > best_result[args.eval_metric]:
                    best_result = result
                    logger.info('!!! Best dev %s (lr=%s, epoch=%d): %.2f'%(
                        args.eval_metric, str(args.learning_rate), epoch, result[args.eval_metric]))
                    save_trained_model(args.output_dir,model,tokenizer)


    if args.do_eval:
        logger.info("Special Tokens",special_tokens)
        if args.eval_test:
            eval_dataset = test_dataset
            eval_examples = test_samples
            eval_features = convert_examples_to_features(test_samples,label2id,args.max_seq_len,tokenizer,special_tokens,unused_tokens=args.add_new_tokens)
            eval_nrel = test_nrel
            logger.info("Special Tokens",special_tokens)
            logger.info('***** Test *****')
            logger.info('   Num Examples = %d',len(test_samples))
            logger.info('   Batch Size = %d', args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
            all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx,
                                       all_obj_idx)
            eval_dataloader = DataLoader(eval_data, batch_size=args.train_batch_size)
            eval_label_ids = all_label_ids
        model = BertForRelation.from_pretrained(args.output_dir,num_rel_labels=num_labels)
        model.to(device)
        preds, result, logits = evaluate(model,device,eval_dataloader,eval_label_ids,num_labels,e2e_ngold=eval_nrel)

        logger.info('*** Evaluation Results ***')
        for key in sorted(result.keys()):
            logger.info('  %s = %s',key,str(result[key]))

        print_pred_json(eval_dataset,eval_examples,preds,id2label,os.path.join(args.output_dir,args.prediction_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default=r'E:\Code File\Pre-train-Model\torchVersion\bert-base-uncased',type=str,
                        help='将要是用的预训练模型')
    parser.add_argument('--output_dir',default='./relation_output',type=str)
    parser.add_argument('--eval_per_epoch',default=10,type=int)
    parser.add_argument('--max_seq_len',default=128,type=int)
    parser.add_argument('--negative_label',default='no_relation',type=str,
                        help='如果两个span对之间没有关系，则标签为no_relation')
    parser.add_argument('--do_train',default=True,action='store_true')
    parser.add_argument('--train_file',default=r'E:\Code File\Pytorch\NLP\Paper code\Code Reconstr\PURE\processed_data\json\train.json',type=str)
    parser.add_argument('--do_eval',default=False,action='store_true')
    parser.add_argument('--do_lower_case',default=True,action='store_true')
    parser.add_argument('--eval_test',default=False,action='store_true')
    parser.add_argument('--eval_with_gold',default=False,action='store_true',
                        help='是否是用gold标签进行验证模型')
    parser.add_argument('--train_batch_size',default=12,type=int)
    parser.add_argument('--train_mode',type=str,default=False,choices=['random','sorted','random_sorted'])
    parser.add_argument('--eval_batch_size',default=12,type=int)
    parser.add_argument('--eval_metric',default='f1',type=str)
    parser.add_argument('--learning_rate',default=2e-5,type=float)
    parser.add_argument('--num_train_epochs',default=30,type=float)
    parser.add_argument('--warmup_proportion',default=0.1,type=float)
    parser.add_argument('--no_cuda',default=False,action='store_true')
    parser.add_argument('--seed',default=0,type=int)
    parser.add_argument('--bertadam',default=True,action='store_true')
    parser.add_argument('--entity_output_dir',default='./entity_output/')
    parser.add_argument('--entity_predictions_dev',type=str,default='ent_pred_dev.json')
    parser.add_argument('--entity_predictions_test',type=str,default='ent_pred_test.json')
    parser.add_argument('--prediction_file',type=str,default='predictions.json')
    parser.add_argument('--task',type=str,default='scierc')
    parser.add_argument('--context_window',type=int,default=100)
    parser.add_argument('--add_new_tokens',default=True,action='store_true')

    args = parser.parse_args()
    main(args)





























