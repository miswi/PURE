import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from allennlp.nn.util import batched_index_select
from allennlp.modules import FeedForward
from transformers import BertTokenizer,BertPreTrainedModel,BertModel
import logging

logger = logging.getLogger('root')

class BertForEntity(BertPreTrainedModel):
    def __init__(self,config,num_ner_labels,head_hidden_dim=150,width_embedding_dim=150,max_span_length=8):
        """
        :param config:
        :param num_ner_labels:
        :param head_hidden_dim: 隐藏层中span的维度
        :param width_embedding_dim: 上下文span的长度
        :param max_span_length: span的长度
        """
        super().__init__(config)
        self.bert = BertModel(config)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(max_span_length+1,width_embedding_dim)

        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=config.hidden_size*2+width_embedding_dim,  # 768*2 + 150
                        num_layers=2,
                        hidden_dims=head_hidden_dim,
                        activations=F.relu,
                        dropout=0.2),
            nn.Linear(head_hidden_dim,num_ner_labels)
        )
        self.init_weights()

    def _get_span_embedding(self,input_ids,spans,token_type_ids=None,attention_mask=None):
        # 先通过bert获得token的contextualized embedding
        sequence_output,pooler_output = self.bert(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        sequence_output = self.hidden_dropout(sequence_output)
        """
        spans_mask。shape = spans.shape: [batch_size, num_spans, 3]; 
        """
        span_starts = spans[:,:,0].view(spans.size(0),-1)  # num_spans * 3
        spans_start_embedding = batched_index_select(sequence_output,span_starts)
        spans_ends = spans[:,:,1].view(spans.size(0),-1)
        spans_end_embedding = batched_index_select(sequence_output,spans_ends)

        spans_width = spans[:,:,2].view(spans.size(0),-1)
        spans_with_embedding = self.width_embedding(spans_width)
        spans_embedding = torch.cat((spans_start_embedding,spans_end_embedding,spans_with_embedding),dim=-1)  # (batch_size, num_spans, hidden_size*2+embedding_dim)
        return spans_embedding

    def forward(self,input_ids,spans,spans_mask,spans_ner_label=None,token_type_ids=None,attention_mask=None):
        # spans_embedding：（spans_start,spans_end,spans_width）
        spans_embedding = self._get_span_embedding(input_ids,spans,token_type_ids=token_type_ids,attention_mask=attention_mask)
        ffnn_hidden = [] # the results of feedforward networks in each layer
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        logits = ffnn_hidden[-1]  # （batch_size,seq_len,ner_label_num）取最后一层的hidden_state做分类

        if spans_ner_label is not None:
            loss_fct = CrossEntropyLoss(reduction='sum')
            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                active_logits = logits.view(-1,logits.shape[-1])  # (-1, num_ner_labels)
                # 没有标签的spans不计算loss
                active_labels = torch.where(
                    active_loss, spans_ner_label.view(-1), torch.tensor(loss_fct.ignore_index).type_as(spans_ner_label)
                )
                loss = loss_fct(active_logits,active_labels)
            else:
                loss = loss_fct(logits.view(-1,logits.shape[-1]), spans_ner_label.view(-1))
            return loss,logits,spans_embedding
        else:
            return logits, spans_embedding, spans_embedding


class EntityModel():
    def __init__(self,args,num_ner_labels):
        super().__init__()
        bert_model_name = args.model
        vocab_name = bert_model_name

        if args.bert_model_dir is not None:
            bert_model_name = str(args.bert_model_dir) + '/'
            # vocab_name = bert_model_name + 'vocab.txt'
            vocab_name = bert_model_name
            logger.info('Loading BERT model from {}'.format(bert_model_name))

        self.tokenizer = BertTokenizer.from_pretrained(vocab_name)
        self.bert_model = BertForEntity.from_pretrained(bert_model_name,num_ner_labels=num_ner_labels,max_span_length=args.max_span_length)

        self._model_device='cuda'
        self.move_model_to_cuda()

    def move_model_to_cuda(self):
        if not torch.cuda.is_available():
            logger.error('No CUDA found')
            exit(-1)
        logger.info('Moving to cuda....')
        self._model_device='cuda'
        self.bert_model.cuda()
        logger.info(' # CPUs = %d'%(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)

    def _get_input_tensors(self,tokens,spans,spans_ner_label):
        """
        :param tokens:一个句子
        :param spans:该句子对应的spans
        :param spans_ner_label: spans标签
        :return: tensor类型下：数值型tokens，spans，spans_ner_label
        """
        start2idx = []  # 文本token在分词之前起始位置
        end2idx = []  # 文本token在分词之后最后一个token的结束位置，一个token可能由多个subword构成，比如'robot-human' --> 'robot','-','human'

        # 对句子进行subword，添加[CLS],[SEP]
        bert_tokens = []  # 文本token --> 数值token
        bert_tokens.append(self.tokenizer.cls_token)  # 添加[CLS]
        for token in tokens:
            start2idx.append(len(bert_tokens))
            sub_tokens = self.tokenizer.tokenize(token)  # 对文本token进行分词（subword）
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens)-1)  # subword后最后一个token的索引位置
        bert_tokens.append(self.tokenizer.sep_token)  # 添加[SEP]

        # 对分词好词的文本token 转化为数值型token
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        token_tensors = torch.tensor([indexed_tokens])  # 升维度，转tensor类型

        # start2idx[i]~end2idx[] --> bert分词后token位置
        bert_spans = [[start2idx[span[0]],end2idx[span[1]],span[2]] for span in spans]
        bert_spans_tensor = torch.tensor([bert_spans])

        spans_ner_label_tensor = torch.tensor([spans_ner_label])
        return token_tensors,bert_spans_tensor,spans_ner_label_tensor

    def _get_input_tensors_batch(self,sample_list,training=True):
        tokens_tensor_list = []  # 3维，[batch_size,sequence_len,embed_dim]
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        sentence_length = []

        # 根据一个batch下最长的样本大小决定padding数量
        max_tokens = 0
        max_spans = 0
        for sample in sample_list:  # 一条句子
            tokens = sample['tokens']
            spans = sample['spans']
            spans_ner_label = sample['spans_labels']

            # tokens_tensor：一个样本的数值型token
            # bert_spans_tensor：句子中由token组成的所有span，[[1, 1, 1],[1, 2, 2],[1, 3, 3],...,[132, 132, 1],[132, 133 ,2],[132, 134 ,3]]
            # spans_ner_label_tensor：span对应的标签
            tokens_tensors,bert_spans_tensor,spans_ner_label_tensor = self._get_input_tensors(tokens,spans,spans_ner_label)
            tokens_tensor_list.append(tokens_tensors)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            assert (bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1])  # the number of spans should be equal to the number of its labels
            if tokens_tensors.shape[1] > max_tokens:
                max_tokens = tokens_tensors.shape[1]  # update the number of tokens in the sentence, find the longest sentence
            if bert_spans_tensor.shape[1] > max_spans:
                max_spans = bert_spans_tensor.shape[1]  # find the maxmium spans
            sentence_length.append(sample['sent_length'])


        # apply padding and concatenate tensors
        final_tokens_tensors = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_mask_tensor = None
        for tokens_tensor,bert_spans_tensor,spans_ner_label_tensor in zip(tokens_tensor_list,bert_spans_tensor_list,spans_ner_label_tensor_list):
            # tokens_tensor：数值型tokens，tokens_tensor.shape：1 x num_tokens
            # bert_spans_tensor：句子中由token组成的所有span，[[1, 1, 1],[1, 2, 2],[1, 3, 3],...,[132, 132, 1],[132, 133 ,2],[132, 134 ,3]]
            # spans_ner_label_tensor：span对应的标签

            # padding for tokens
            num_tokens = tokens_tensor.shape[1]  # 当前句子的token数量
            tokens_pad_length = max_tokens - num_tokens  # 需要padding的个数
            attention_tensor = torch.full([1,num_tokens],1,dtype=torch.long)  # 对非padding token进行attention
            if tokens_pad_length > 0:
                pad = torch.full([1,tokens_pad_length],self.tokenizer.pad_token_id,dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor,pad),dim=1)
                attention_pad = torch.full([1,tokens_pad_length],0,dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor,attention_pad),dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1,num_spans],1,dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full([1,spans_pad_length,bert_spans_tensor.shape[2]],0,dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor,pad),dim=1)
                mask_pad = torch.full([1,spans_pad_length],0,dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor,mask_pad),dim=1)
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor,mask_pad),dim=1)

            # assert (bert_spans_tensor.shape[1] == spans_ner_label_tenosr.shape[1])

            # update final outputs
            if final_tokens_tensors is None:
                final_tokens_tensors = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensors = torch.cat((final_tokens_tensors,tokens_tensor),dim=0)
                final_attention_mask = torch.cat((final_attention_mask,attention_tensor),dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor,bert_spans_tensor),dim=0)
                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor,spans_ner_label_tensor),dim=0)
                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor,spans_mask_tensor),dim=0)

        return final_tokens_tensors,final_attention_mask,final_bert_spans_tensor,final_spans_mask_tensor,final_spans_ner_label_tensor,sentence_length

    def run_batch(self,sample_list,training=True):
        # convert samples to input tensors
        """
            tokens_tensor：数值型文本token
            attention_mask_tensor：对数值型token进行attention，对padding的值进行mask
            bert_spans_tensor：[[1, 1, 1],[1, 2, 2],[1, 3, 3],...[0, 0, 0]]，0为padding
            spans_mask_tensor：a list that consists of 1(the number of spans) and 0(the numebr of masked span) [1,1,1,....,0,0,0]
            spans_ner_label_tensor：spans对应的标签，对于masked span标签为0
            _get_input_tensors_batch：对batch内的数据进行维度统一，保证size相同才可以输入到模型中
        """
        tokens_tenosr, attention_mask_tensor,bert_spans_tensor,spans_mask_tensor, spans_ner_label_tensor,sentence_length = self._get_input_tensors_batch(sample_list,training)

        output_dict = {
            'ner_loss':0,
        }

        if training:
            self.bert_model.train()
            ner_loss, ner_logits, spans_embedding = self.bert_model(
                input_ids = tokens_tenosr.to(self._model_device),
                spans = bert_spans_tensor.to(self._model_device),
                spans_mask = spans_mask_tensor.to(self._model_device),
                spans_ner_label = spans_ner_label_tensor.to(self._model_device),
                attention_mask = attention_mask_tensor.to(self._model_device)
            )
            output_dict['ner_loss'] = ner_loss.sum()
            output_dict['ner_llh'] = F.log_softmax(ner_logits,dim=-1)
        else:
            self.bert_model.eval()
            with torch.no_grad():
                ner_logits,spans_embedding,last_hidden = self.bert_model(
                    input_ids = tokens_tenosr.to(self._model_device),
                    spans = bert_spans_tensor.to(self._model_device),
                    spans_mask = spans_mask_tensor.to(self._model_device),
                    spans_ner_label = None,
                    attention_mask = attention_mask_tensor.to(self._model_device)
                )
            _, predicted_label = ner_logits.max(2)
            predicted_label = predicted_label.cpu().numpy()
            last_hidden = last_hidden.cpu().numpy()

            predicted = []
            pred_prob = []
            hidden = []
            for i, sample in enumerate(sample_list):
                ner = []
                prob = []
                lh = []
                for j in range(len(sample['spans'])):
                    ner.append(predicted_label[i][j])
                    prob.append(ner_logits[i][j].cpu().numpy())
                    lh.append(last_hidden[i][j])
                predicted.append(ner)
                pred_prob.append(prob)
                hidden.append(lh)
            output_dict['pred_ner'] = predicted
            output_dict['ner_probs'] = pred_prob
            output_dict['ner_last_hidden'] = hidden

        return output_dict













