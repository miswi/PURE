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
from relation.utils import decode_sample_id,get_features_from_file,add_marker_tokens,evaluate,save_trained_model,print_pred_json,setseed
from relation.utils import generate_relation_data
from relation.models import BertForRelationApprox
from shared.const import task_rel_labels, task_ner_labels
from shared.data_structured import Dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default=r'E:\Code File\Pre-train-Model\torchVersion\bert-base-uncased',type=str,
                        help='将要是用的预训练模型')
    parser.add_argument('--output_dir',default='./relation_approx_output',type=str)
    parser.add_argument('--eval_per_epoch',default=10,type=int)
    parser.add_argument('--max_seq_len',default=128,type=int)
    parser.add_argument('--negative_label',default='no_relation',type=str,
                        help='如果两个span对之间没有关系，则标签为no_relation')
    parser.add_argument('--do_train',default=True,action='store_true')
    parser.add_argument('--train_file',default=r'E:\Code File\Pytorch\NLP\Paper code\Code Reconstr\PURE\processed_data\json\train.json',type=str)
    parser.add_argument('--do_eval',default=True,action='store_true')
    parser.add_argument('--do_lower_case',default=True,action='store_true')
    parser.add_argument('--eval_test',default=True,action='store_true')
    parser.add_argument('--eval_with_gold',default=True,action='store_true',
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
    parser.add_argument('--batch_computation',default=True,action='store_true')
    parser.add_argument('--add_new_tokens',default=True,action='store_true')

    args = parser.parse_args()
    # main(args)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    n_gpu = torch.cuda.device_count()

    setseed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, 'train.log'), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, 'eval.log'), 'w'))
    logger.info(sys.argv)
    logger.info(args)
    logger.info("device: {}, N_gpu: {}".format(device, n_gpu))

    if os.path.exists(os.path.join(args.output_dir, 'label_list.json')):
        with open(os.path.join(args.output_dir, 'label_list.json'), 'r') as f:
            label_list = json.load(f)
    else:
        label_list = [args.negative_label] + task_rel_labels[args.task]
        with open(os.path.join(args.output_dir, 'label_list.json'), 'w') as f:
            json.dump(label_list, f)

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    if args.add_new_tokens:
        add_marker_tokens(tokenizer, task_rel_labels[args.task])

    if os.path.exists(os.path.join(args.output_dir, 'special_tokens.json')):
        with open(os.path.join(args.output_dir, 'special_tokens.json'), 'w') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}

    if args.do_eval and (args.do_train or not(args.eval_test)):
        eval_dataset, eval_features, eval_ner = get_features_from_file(
            os.path.join(args.entity_output_dir, args.entity_predictions_dev), label2id, args.max_seq_len, tokenizer, special_tokens, use_gold=args.eval_with_gold,
            context_window=args.context_window, batch_computation=args.batch_computation,
            unused_tokens=args.add_new_tokens)
        logger.info('******* Dev *******')
        logger.info('  Num examples = %d', len(eval_features))
        logger.info('  Batch size = %d', args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.long)
        all_sub_obj_ids = torch.tensor([f.sub_obj_ids for f in eval_features], dtype=torch.long)
        all_sub_obj_masks = torch.tensor([f.sub_obj_masks for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_position_ids, all_input_mask, all_segment_ids, all_labels,
                                   all_sub_obj_ids, all_sub_obj_masks)
        eval_dataloader = DataLoader(eval_data, batch_size=args.train_batch_size)
    with open(os.path.join(args.output_dir, 'special_tokens.json'),'w') as f:
        json.dump(special_tokens, f)

    if args.do_train:
        train_dataset, train_features, train_ner = get_features_from_file(
            args.train_file, label2id, args.max_seq_len, tokenizer, special_tokens, use_gold=True,
            context_window=args.context_window, batch_computation=args.batch_computation,
            unused_tokens=args.add_new_tokens
        )
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_position_ids = torch.tensor([f.position_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)
        all_sub_obj_ids = torch.tensor([f.sub_obj_ids for f in train_features], dtype=torch.long)
        all_sub_obj_masks = torch.tensor([f.sub_obj_masks for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids,all_position_ids,all_input_mask,all_segment_ids,all_labels,all_sub_obj_ids,all_sub_obj_masks)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
        logger.info('***** Training *****')
        logger.info('   Num examples = %d', len(train_features))
        logger.info('   Batch size = %d', args.train_batch_size)
        logger.info('   Num steps = %d', num_train_optimization_steps)

        best_result = None
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)

        lr = args.learning_rate
        model = BertForRelationApprox.from_pretrained(args.model,cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),num_rel_labels=num_labels)
        if hasattr(model,'bert'):
            model.bert.resize_token_embeddings(len(tokenizer))
        else:
            raise TypeError('Unknown model calss')

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer
                        if not any(nd in name for nd in no_decay)], 'weight_decay': 0.1},
            {'params': [param for name, param in param_optimizer
                        if any(nd in name for nd in no_decay)], 'weight_decay': 0.0}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=args.bertadam)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    int(num_train_optimization_steps * args.warmup_proportion),
                                                    num_train_optimization_steps)

        start_time = time.time()
        global_step = 0
        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        for epoch in tqdm(range(args.num_train_epochs)):
            model.train()
            logger.info('Start epoch #{} (lr = {})...'.format(epoch, args.learning_rate))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)
            for step, batch in tqdm(enumerate(train_batches)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_position, input_mask, segment_ids, labels, sub_obj_ids, sub_obj_masks = batch
                loss = model(input_ids, segment_ids, input_mask, labels=labels, sub_obj_ids=sub_obj_ids, sub_obj_masks=sub_obj_masks,input_position=input_position)
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
                        epoch, step + 1, len(train_batches), time.time() - start_time, tr_loss / nb_tr_steps))

            if args.do_eval:
                preds, result, logits = evaluate(model,device,eval_dataloader,e2e_ngold=eval_ner)
                result['global_step'] = global_step
                result['epoch'] = epoch
                result['learning_rate'] = lr
                result['batch_size'] = args.train_batch_size

                if best_result is None or result[args.eval_metric] > best_result[args.eval_metric]:
                    best_result = result
                    logger.info('!!! Best dev %s (lr=%s, epoch=%d): %.2f' % (
                        args.eval_metric, str(lr), epoch, result[args.eval_metric]))
                    save_trained_model(args.output_dir, model, tokenizer)

    if args.do_eval:
        if args.eval_test:
            eval_dataset, eval_features, eval_ner = get_features_from_file(
                os.path.join(args.entity_output_dir, args.entity_predictions_test), label2id, args.max_seq_len, tokenizer, special_tokens, use_gold=args.eval_with_gold,
                context_window=args.context_window, batch_computation=args.batch_computation,
                unused_tokens=args.add_new_tokens)
            logger.info('******* Test *******')
            logger.info('  Num examples = %d', len(eval_features))
            logger.info('  Batch size = %d', args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_position_ids = torch.tensor([f.position_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.long)
            all_sub_obj_ids = torch.tensor([f.sub_obj_ids for f in eval_features], dtype=torch.long)
            all_sub_obj_masks = torch.tensor([f.sub_obj_masks for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_position_ids, all_input_mask, all_segment_ids, all_labels,
                                       all_sub_obj_ids, all_sub_obj_masks)
            eval_dataloader = DataLoader(eval_data, batch_size=args.train_batch_size)
            model = BertForRelationApprox.from_pretrained(args.output_dir, num_rel_labels=num_labels)
            model.to(device)
            preds, result = evaluate(model,device,eval_dataloader,e2e_ngold=eval_ner)
            logger.info(' *** Evaluation(test) Results ***')
            for key in sorted(result.keys()):
                logger.info('  %s = %s',key, str(result[key]))
            print_pred_json(eval_dataset,eval_features,preds,id2label,os.path.join(args.output_dir,args.prediction_file))















