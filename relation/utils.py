import json
import logging
import os
import time
import random
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from shared.data_structured import Dataset
logger = logging.getLogger(__name__)
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE,WEIGHTS_NAME,CONFIG_NAME

def generate_relation_data(entity_data,use_gold=False,context_window=0):
    '''
    Prepare data for the relation model
    if training, set use_gold=True
    '''
    logging.info('Generate relation data from %s'%(entity_data))
    data = Dataset(entity_data)

    nner, nrel = 0,0
    max_sentsample = 0
    samples = []
    for doc in data:
        for i, sent in enumerate(doc):
            sent_samples = []
            nner += len(sent.ner)
            nrel += len(sent.relatoins)

            if use_gold:
                sent_ner = sent.ner
            else:
                sent_ner = sent.predicted_ner

            gold_rel = {}
            for rel in sent.relatoins:
                gold_rel[rel.pair] = rel.label

            sent_start = 0
            sent_end = len(sent.text)
            tokens = sent.text

            if context_window > 0:
                add_left = (context_window - len(sent.text)) // 2
                add_right = (context_window - len(sent.text)) - add_left

                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    tokens = context_to_add + tokens
                    add_left -= context_window - len(tokens)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    tokens = tokens + context_to_add
                    add_right -= len(tokens)
                    j += 1

            for x in range(len(sent_ner)):
                for y in range(len(sent_ner)):
                    if x == y:
                        continue
                    sub = sent_ner[x]
                    obj = sent_ner[y]
                    label = gold_rel.get((sub.span,obj.span),'no_relation')
                    sample = {}
                    sample['doc_id'] = doc._doc_key
                    sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key,sent.sentence_ix,sub.span.start_doc,sub.span.end_doc,obj.span.start_doc,obj.span.end_doc)
                    sample['relation'] = label
                    sample['subj_start'] = sub.span.start_sent + sent_start  # 该样本下第sent_start个句子的起始位置 + span的起始位置（span.start_sent） = span在整个样本中的位置
                    sample['subj_end'] = sub.span.end_sent + sent_start
                    sample['subj_type'] = sub.label
                    sample['obj_start'] = obj.span.start_sent + sent_start
                    sample['obj_end'] = obj.span.end_doc + sent_start
                    sample['obj_type'] = obj.label
                    sample['tokens'] = tokens
                    sample['sent_start'] = sent_start
                    sample['sent_end'] = sent_end

                    sent_samples.append(sample)

            max_sentsample = max(max_sentsample,len(sent_samples))
            samples += sent_samples

    tot = len(samples)
    logging.info('#samples: %d, max #sent.samples: %d'%(tot,max_sentsample))
    return data,samples,nrel

def decode_sample_id(sample_id):
    doc_sent = sample_id.split('::')[0]
    pair = sample_id('::')[1]
    pair = pair.split('-')
    sub = (int(pair[0][1:-1].split(',')[0]), int(pair[0][1:-1].split(',')[1]))
    obj = (int(pair[1][1:-1].split(',')[0]), int(pair[1][1:-1].split(',')[1]))

    return doc_sent,sub,obj

class InputFeatures(object):
    def __init__(self,input_ids, position_ids, input_mask, labels, sub_obj_ids, sub_obj_masks, meta, max_seq_len):
        # padding
        self.num_labels = len(labels)
        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        position_ids += padding
        input_mask += padding

        label_padding = [0] * (max_seq_len // 4 - len(labels))
        ids_padding = [[0, 0]] * (max_seq_len // 4 - len(labels))
        labels += label_padding
        sub_obj_masks += label_padding
        sub_obj_ids += ids_padding

        # 文本token注意力mask矩阵，文本token的mask，不含有marker标记符
        attention_mask = []
        for _, from_mask in enumerate(input_mask):
            attention_mask_i = []
            for to_mask in input_mask:
                if to_mask <= 1:
                    attention_mask_i.append(to_mask)
                elif from_mask == to_mask and from_mask > 0:  # 1，文本token
                    attention_mask_i.append(1)
                else:  # from_mask=0，则为padding，
                    attention_mask_i.append(0)
            attention_mask.append(attention_mask_i)

        self.input_ids = input_ids
        self.position_ids = position_ids
        self.original_input_mask = input_mask
        self.input_mask = attention_mask
        self.segment_ids = [0] * len(input_ids)
        self.labels = labels
        self.sub_obj_ids = sub_obj_ids
        self.sub_obj_masks = sub_obj_masks
        self.meta = meta

def add_marker_tokens(tokenizer,ner_labels):
    new_tokens = ['<SUBJ_START>','<SUBJ_END>','<OBJ_START>','<OBJ_END>']
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>'%label)
        new_tokens.append('<SUBJ_END=%s>' % label)
        new_tokens.append('<OBJ_START=%s>' % label)
        new_tokens.append('<OBJ_END=%s>' % label)
    for label in ner_labels:
        new_tokens.append('<SUBJ=%s>'%label)
        new_tokens.append('<OBJ=%s>' % label)
    tokenizer.add_tokens(new_tokens)
    logger.info('# vocab after adding markers: %d'%len(tokenizer))

def get_features_from_file(filename, label2id, max_seq_len, tokenizer, special_tokens, use_gold, context_window,
                           batch_computation=False, unused_tokens=True):
    def get_special_token(w):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = '[unused%d]' % (len(special_tokens) + 1)
            else:
                special_tokens[w] = ('<' + w + '>').lower()
        return special_tokens[w]

    CLS = '[CLS]'
    SEP = '[SEP]'

    num_shown_examples = 0
    features = []
    nrel = 0

    data = Dataset(filename)
    for doc in data:
        for i, sent in enumerate(doc):
            sid = i
            nrel += len(sent.relatoins)

            if use_gold:
                sent_ner = sent.ner
            else:
                sent_ner = sent.predicted_ner

            if len(sent_ner) <= 1:
                continue

            text = sent.text
            sent_start = 0
            sent_end = len(text)

            if context_window > 0:
                add_left = (context_window - len(text)) // 2
                add_right = (context_window - len(text)) - add_left

                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    text = context_to_add + text
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)  # 上下文的起始位置
                    sent_end += len(context_to_add)  # 上下文的结束位置
                    j -= 1  # 上一句

                j = i + 1
                while j < len(doc) and add_right > 0:  # 在不超出该文档下，添加context
                    context_to_add = doc[j].text[:add_right]
                    text = text + context_to_add
                    add_right -= len(context_to_add)
                    j += 1  # 下一句

            tokens = [CLS]
            token_start = {}
            token_end = {}
            for i, token in enumerate(text):
                token_start[i] = len(tokens)
                for sub_token in tokenizer.tokenize(token):
                    tokens.append(sub_token)
                token_end[i] = len(tokens) - 1
            tokens.append(SEP)
            # num_tokens = len(tokens)
            tokens = tokens[:max_seq_len-4]
            num_tokens = len(tokens)
            assert (num_tokens + 4 <= max_seq_len)

            position_ids = list(range(len(tokens)))
            marker_mask = 1
            input_mask = [1] * len(tokens)
            labels = []
            sub_obj_ids = []
            sub_obj_masks = []
            sub_obj_pairs = []

            gold_rel = {}
            for rel in sent.relatoins:
                gold_rel[rel.pair] = rel.label

            for x in range(len(sent_ner)):
                for y in range(len(sent_ner)):
                    if x == y:
                        continue
                    sub = sent_ner[x]
                    obj = sent_ner[y]
                    label = label2id[gold_rel.get((sub.span, obj.span), 'no_relation')]
                    # 主客体marker
                    SUBJECT_START_NER = get_special_token('SUBJ_START=%s' % sub.label)
                    SUBJECT_END_NER = get_special_token('SUBJ_END=%s' % sub.label)
                    OBJECT_START_NER = get_special_token('OBJ_START=%s' % obj.label)
                    OBJECT_END_NER = get_special_token('OBJ_END=%s' % obj.label)

                    if (len(tokens) + 4 > max_seq_len) or (not (batch_computation) and len(tokens) > num_tokens):
                        input_ids = tokenizer.convert_tokens_to_ids(tokens)
                        features.append(
                            InputFeatures(input_ids=input_ids,
                                          position_ids=position_ids,
                                          input_mask=input_mask,
                                          labels=labels,
                                          max_seq_len=max_seq_len,
                                          sub_obj_ids=sub_obj_ids,
                                          sub_obj_masks=sub_obj_masks,
                                          meta={'doc_id': doc._doc_key, 'sent_id': sid, 'sub_obj_pairs': sub_obj_pairs}))

                        tokens = tokens[:num_tokens]
                        position_ids = list(range(len(tokens)))
                        marker_mask = 1
                        input_mask = [1] * len(tokens)
                        labels = []
                        sub_obj_ids = []
                        sub_obj_masks = []
                        sub_obj_pairs = []

                    # 在句子后添加主客体marker
                    tokens = tokens + [SUBJECT_START_NER,SUBJECT_END_NER,OBJECT_START_NER,OBJECT_END_NER]
                    position_ids = position_ids + [token_start[sent_start+sub.span.start_sent],  # 主体头marker
                                                   token_end[sent_start+sub.span.end_sent],  # 主体尾marker
                                                   token_start[sent_start+obj.span.start_sent],  # 客体头marker
                                                   token_end[sent_start+obj.span.end_sent]]  # 客体尾marker

                    marker_mask += 1
                    input_mask = input_mask + [marker_mask] * 4  # 主客体mask
                    labels.append(label)
                    sub_obj_ids.append([len(tokens)-4,len(tokens)-2])  # 主客体的索引位置
                    sub_obj_masks.append(1)
                    sub_obj_pairs.append([sub.span,obj.span])

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            features.append(
                InputFeatures(input_ids=input_ids,
                              position_ids=position_ids,
                              input_mask=input_mask,
                              labels=labels,
                              sub_obj_ids=sub_obj_ids,
                              sub_obj_masks=sub_obj_masks,
                              meta={'doc_id': doc._doc_key, 'sent_id': sid, 'sub_obj_pairs': sub_obj_pairs},
                              max_seq_len=max_seq_len)
            )

            if num_shown_examples < 20:
                num_shown_examples += 1
                logger.info('*** Example ***')
                logger.info('guid: %s'%(doc._doc_key))
                logger.info('tokens: %s'%' '.join([str(token) for token in tokens]))
                logger.info('input_ids: %s'%' '.join([str(x) for x in features[-1].input_ids]))
                logger.info('position_ids: %s' % ' '.join([str(x) for x in features[-1].position_ids]))
                logger.info('input_masks: %s' % ' '.join([str(x) for x in features[-1].original_input_mask]))
                logger.info('labels: %s' % ' '.join([str(x) for x in features[-1].labels]))
                logger.info('sub_obj_ids: %s' % ' '.join([str(x) for x in features[-1].sub_obj_ids]))
                logger.info('sub_obj_masks: %s' % ' '.join([str(x) for x in features[-1].sub_obj_masks]))
                logger.info('sub_obj_pairs: %s' % ' '.join([str(x) for x in features[-1].meta['sub_obj_pairs']]))

    max_num_tokens = 0
    max_num_pairs = 0
    num_label = 0
    for feat in features:
        if len(feat.input_ids) > max_num_tokens:
            max_num_tokens = len(feat.input_ids)
        if len(feat.sub_obj_ids) > max_num_pairs:
            max_num_pairs = len(feat.sub_obj_ids)
        num_label += feat.num_labels

    logger.info('Max # tokens: %d'%max_num_tokens)
    logger.info('Max # pairs: %d'%max_num_pairs)
    logger.info('Total labels: %d'%(num_label))
    logger.info('# labels per sample on average: %f'%(num_label / len(features)))

    return data, features, nrel

def compute_f1(preds,labels,e2e_ngold):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds,labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if pred !=0 and label !=0 and pred == label:
            n_correct += 1

    if n_correct == 0:
        return {'Precision':0.0, 'Recall':0.0, 'F1':0.0}
    else:
        prec = n_correct * 1.0 /n_pred
        recall = n_correct * 1.0 / n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec *recall / (prec + recall)
        else:
            f1 = 0.0

    if e2e_ngold is not None:
        e2e_recall = n_correct * 1.0 / e2e_ngold
        e2e_f1 = 2.0 * prec * e2e_recall / (prec + e2e_recall)
    else:
        e2e_recall = e2e_f1 = 0.0

    return {'Precision':prec, 'recall':e2e_recall, 'f1':e2e_f1,
            'task_recall':recall, 'task_f1':f1, 'n_correct':n_correct, 'n_pred':n_pred, 'n_gold':e2e_ngold, 'task_ngold':n_gold}

def simple_accuracy(preds,labels):
    return (preds == labels).mean()

def evaluate(model,device,eval_dataloader,e2e_ngold=None,verbose=True):
    c_time = time.time()
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    all_preds = []
    all_labels = []
    for input_ids, input_position, input_mask, segment_ids, labels, sub_obj_ids, sub_obj_masks in eval_dataloader:
        batch_labels = labels
        batch_masks = sub_obj_masks
        input_ids = input_ids.to(device)
        input_position = input_position.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        sub_obj_ids = sub_obj_ids.to(device)
        sub_obj_masks = sub_obj_masks.to(device)
        with torch.no_grad():
            logits = model(input_ids,segment_ids,input_mask,labels=None,sub_obj_ids=sub_obj_ids,sub_obj_masks=sub_obj_masks,input_position=input_position)
        loss_fct = CrossEntropyLoss()
        active_loss = (sub_obj_masks.view(-1) == 1)
        active_logits = logits.view(-1, logits.shape[-1])
        active_labels = torch.where(
            active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        )
        tmp_eval_loss = loss_fct(active_logits, active_labels)
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        batch_preds = np.argmax(logits.detach().cpu().numpy(),axis=2)

        for i in range(batch_preds.shape[0]):
            for j in range(batch_preds.shape[1]):
                if batch_masks[i][j] == 1:  # sub_obj_masks == 1
                    all_preds.append(batch_preds[i][j])
                    all_labels.append(batch_labels[i][j])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    eval_loss = eval_loss / nb_eval_steps
    result = compute_f1(all_preds,all_labels,e2e_ngold=e2e_ngold)
    result['accuracy'] = simple_accuracy(all_preds,labels)
    result['eval_loss'] = eval_loss
    if verbose:
        logger.info('****** Eval results (used time: %.3f s) ******'%(time.time()-c_time))
        for key in sorted(result.keys()):
            logger.info('  %s = %s',key,str(result[key]))
    return all_preds, result

def save_trained_model(output_dir, model, tokenizer):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger.info('Saving model to %s'%output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir,CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

def print_pred_json(eval_data, eval_features, preds, id2label, output_file):
    rels = dict()
    p = 0
    for feat in eval_features:
        doc_sent = '%s@%d'%(feat.meta['doc_id'],feat.meta['sent_id'])
        if doc_sent not in rels:
            rels[doc_sent] = []
        for pair in feat.meta['sub_obj_pairs']:
            sub = pair[0]
            obj = pair[1]
            # get the next prediction
            pred = preds[p]
            p += 1
            if pred != 0:
                rels[doc_sent].append(sub.start_doc, sub.end_eoc, obj.start_doc, obj.end_doc, id2label[pred])

    js =eval_data.js
    for doc in js:
        doc['predicted_relations'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d'%(doc['doc_key'], sid)
            doc['predicted_relations'].append(rels.get(k, []))

    logger.info('Output predictions to %s'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dump(doc) for doc in js))

def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)














