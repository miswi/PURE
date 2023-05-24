import json
import copy
from collections import Counter
import numpy as np

def fields_to_batches(d, keys_to_ignore=[]):
    keys = [key for key in d.keys() if key not in keys_to_ignore]
    lengths = [len(d[k]) for k in keys]
    # print(set(lengths))
    assert len(set(lengths)) == 1
    length = lengths[0]
    res = [{k: d[k][i] for k in keys} for i in range(length)]
    return res

class Dataset:
    def __init__(self,json_file,pred_file=None,doc_range=None):
        self.js = self._read(json_file,pred_file)
        if doc_range is not None:
            self.js = self.js[doc_range[0]:doc_range[1]]
        self.documents = [Document(js) for js in self.js]  # 将json文件中的每一条样本实例化为Document类

    def update_from_js(self,js):
        self.js = js
        self.documents = [Document(js) for js in self.js]

    def _read(self,json_file,pred_file=None):
        gold_docs = [json.loads(line) for line in open(json_file)]
        if pred_file is None:
            return gold_docs

        pred_docs = [json.loads(line) for line in open(pred_file)]
        merged_docs = []
        for gold,pred in zip(gold_docs,pred_docs):
            assert gold['doc_key'] == pred['doc_key']
            assert gold['sentences'] == pred['sentences']
            merged = copy.deepcopy(gold)
            for k,v in pred.items():
               if 'predicted' in k:
                   merged[k] = v
            merged_docs.append(merged)

        return merged_docs

    def __getitem__(self, ix):
        return self.documents[ix]

    def __len__(self):
        return len(self.documents)

def get_sentence_of_span(span,sentence_starts,doc_tokens):
    """
    Return the index of the sentence that the span is part of
    """
    # 文本结束位置索引
    sentence_ends = [x-1 for x in sentence_starts[1:]] + [doc_tokens - 1]
    in_betweens = [span[0] >= start and span[1] <= end
        for start,end in zip(sentence_starts,sentence_ends)
    ]  # in_betweens，当前实体在那个句子中 [True,False,False,...]
    assert sum(in_betweens) == 1
    the_sentence = in_betweens.index(True)
    return the_sentence

class Document:
    def __init__(self,js):
        self._doc_key = js['doc_key']
        entries = fields_to_batches(js,keys_to_ignore=['doc_key','clusters','predicted_clusters','section_starts'])  # 根据指定关键词，返回与文本相关的数据
        sentence_lengths = [len(entry['sentences']) for entry in entries]
        # 由于要对不同的句子进行实例化，这样实体索引位置就会改变不对，所以要记录不同句子的起始位置，通过实体在文档中的索引位置（多个句子构成） - 句子起始位置 得到以某个句子为起始的实体索引，比如第二个句子的某个实体位置为（44,45），第二句子起始位置为31，那么以第二句为起始位置，则实体位置为（13，14）
        sentence_starts = np.cumsum(sentence_lengths)  # 累加求和
        sentence_starts = np.roll(sentence_starts,1)  # 滚动列表元素  [ 31  54  90 111 119] --> [119  31  54  90 111]
        sentence_starts[0] = 0  # 第一个句子起始为0
        self.sentence_starts = sentence_starts
        # 将每个句子(sentence,ner,relation)与句子起始位置进行zip，并实例化成Sentence对象
        self.sentences = [Sentence(entry,sentence_start,sentence_ix)
            for sentence_ix, (entry, sentence_start)
            in enumerate(zip(entries,sentence_starts))
        ]
        if 'clusters' in js:
            self.clusters = [Cluster(entry,i,self) for i,entry in enumerate(js['clusters'])]
        if 'predicted_clusters' in js:
            self.predicted_clusters = [Cluster(entry,i,self) for i,entry in enumerate(js['predicted_clusters'])]

    def __repr__(self):
        return '\n'.join([str(i) + ': '+' '.join(sent.text) for i,sent in enumerate(self.sentences)])

    def __getitem__(self, ix):
        return self.sentences[ix]

    def __len__(self):
        return len(self.sentences)

    def print_plaintext(self):
        for sent in self:
            print(' '.join(sent.text))

    def find_cluster(self,entity,predicted=True):
        """
        Search through erence clusters and return the one containing the query entity, if it's
        part of a cluster. If we don't find a match, return None
        """
        clusters = self.predicted_clusters if predicted else self.clusters
        for clust in clusters:
            for entity in clust:
                if entity.span == entity.span:
                    return clust
        return None

    @property
    def n_tokens(self):
        return sum([len(sent) for sent in self.sentences])

class Sentence:
    def __init__(self,entry,sentence_start,sentence_ix):
        self.sentence_start = sentence_start
        self.text = entry['sentences']
        self.sentence_ix = sentence_ix
        # Gold
        if 'ner_flavor' in entry:
            self.ner = [NER(this_ner,self.text,sentence_start,flavor=this_flavor) for this_ner,this_flavor in zip(entry['ner'],entry['ner_flavor'])]
        elif 'ner' in entry:
            self.ner = [NER(this_ner,self.text,sentence_start) for this_ner in entry['ner']]
        if 'relations' in entry:
            self.relatoins = [Relation(this_relation,self.text,sentence_start) for this_relation in entry['relations']]
        if 'events' in entry:
            self.events = Events(entry['events'],self.text,sentence_start)

        # Predicted
        if 'predicted_ner' in entry:
            self.predicted_ner = [NER(this_ner,self.text,sentence_start,flavor=None) for this_ner in entry['predicted_ner']]
        if 'predicted_relations' in entry:
            self.predicted_relations = [Relation(this_relation,self.text,sentence_start) for this_relation in entry['predicted_relations']]
        if 'predicted_events' in entry:
            self.predicted_events = Events(entry['predicted_events'], self.text, sentence_start)

        # Top spans
        if 'top_spans' in entry:
            self.top_spans = [NER(this_ner,self.text,sentence_start,flavor=None) for this_ner in entry['top_spans']]

    def __repr__(self):
        the_text = ' '.join(self.text)  # error: join((self.text))
        the_lengths = np.array([len(x) for x in self.text])
        tok_ixs = ''  # 以空格补全间隔的方式，根据这是第i个word的方式，以i记录该word第一个字符的起始位置，比如在第5个word的第一个字符下标记5，表示这是第5个word的起始位置
        for i,offset in enumerate(the_lengths):  # word长度
            true_offset = offset if i < 10 else offset -1  # 当到了第10个word时，索引数字“10”变成了两个字符，索引word长度-1
            tok_ixs += str(i)
            tok_ixs += ' '*true_offset  # 空len(word)长度

        return the_text + '\n' +tok_ixs

    def __len__(self):
        return len(self.text)

    def get_flavor(self,argument):
        the_ner = [x for x in self.ner if x.span == argument.span]
        if len(the_ner) > 1:
            print('Weird')
        if the_ner:
            the_flavor = the_ner[0].flavor
        else:
            the_flavor = None
        return the_flavor

class NER:
    def __init__(self,ner,text,sentence_start,flavor=None):
        self.span = Span(ner[0],ner[1],text,sentence_start)
        self.label = ner[2]
        self.flavor = flavor

    def __repr__(self):
        # 调用NER对象时，返回如下值，比如(4, 4, ['algorithm']): Generic
        return self.span.__repr__()+': '+self.label

    def __eq__(self, other):
        return (self.span == other.span and
                self.label == other.label and
                self.flavor == other.flavor)

class Span:
    def __init__(self,start,end,text,sentence_start):
        self.start_doc = start
        self.end_doc = end
        self.span_doc = (self.start_doc, self.end_doc)
        self.start_sent = start - sentence_start  # start_sent：实体在当前句子中的起始位置，
        self.end_sent = end - sentence_start  # end_sent：实体在当前句子中的结束位置
        self.span_sent = (self.start_sent, self.end_sent)  # error: (self.start_sent - self.end_sent)
        self.text = text[self.start_sent:self.end_sent + 1]

    def __repr__(self):
        # 调用span对象时，返回如下值：
        return str((self.start_sent, self.end_sent, self.text))

    def __eq__(self, other):
        # 判断当前对象对于是否与另一个对象（other，也是Span类）完全相同
        return (self.span_doc == other.span_doc and
                self.span_sent == other.span_sent and
                self.text == other.text)

    def __hash__(self):
        # 若两个对象的值相同，则直接比较两个对象的哈希值
        tup = self.span_doc + self.span_sent + (" ".join(self.text),)
        return hash(tup)

class Relation:
    def __init__(self,relation, text, sentence_start):
        start1, end1 = relation[0], relation[1]
        start2, end2 = relation[2], relation[3]
        label = relation[4]
        span1 = Span(start1,end1,text,sentence_start)
        span2 = Span(start2,end2,text,sentence_start)
        self.pair = (span1,span2)
        self.label = label

    def __repr__(self):
        # 调用Relation对象时，返回一下值，比如：(20, 21, ['image', 'sequence']), (4, 4, ['algorithm']): USED-FOR
        return self.pair[0].__repr__()+', '+self.pair[1].__repr__()+': '+self.label

    def __eq__(self, other):
        return (self.pair == other.pair) and (self.label == other.label)

class Events:
    def __init__(self,events_json, text, sentence_start):
        self.event_list = [Event(this_event,text,sentence_start) for this_event in events_json]
        self.triggers = set([event.trigger for event in self.event_list])
        self.arguments = set([arg for event in self.event_list for arg in event.arguments])

    def __len__(self):
        return len(self.event_list)

    def __getitem__(self, i):
        return self.event_list[i]

    def __repr__(self):
        return '\n\n'.join([event.__repr__() for event in self.event_list])

    def span_matches(self,argument):
        return set([candidate for candidate in self.arguments
                     if candidate.span.span_sent == argument.span.span_sent])

    def event_type_matches(self,argument):
        return set([candidate for candidate in self.span_matches(self.arguments)
                    if candidate.event_type == argument.event_type])

    def matches_except_event_type(self,argument):
        matched = [candidate for candidate in self.span_matches(argument)
                   if candidate.event_type != argument.event_type
                   and candidate.role == argument.role]
        return set(matched)

    def exact_match(self,argument):
        for candidate in self.arguments:
            if candidate == argument:
                return True
        return False

class Event:
    def __init__(self,event,text,sentence_start):
        trig = event[0]
        args = event[1:]
        trigger_token = Token(trig[0],text,sentence_start)
        self.trigger = Trigger(trigger_token,trig[1])

        self.arguments = []
        for arg in args:
            span = Span(arg[0],arg[1],text,sentence_start)
            self.arguments.append(Argument(span,arg[2],self.trigger.label))

    def __repr__(self):
        res = '<'
        res += self.trigger.__repr__() + ':\n'
        for arg in self.arguments:
            res += 6*" " + arg.__repr__() + ';\n'
        res = res[:-2] + '>'
        return res

class Token:
    def __init__(self,ix,text,sentence_start):
        self.ix_doc = ix
        self.ix_sent = ix - sentence_start
        self.text = text[self.ix_sent]

    def __repr__(self):
        return str((self.ix_sent, self.text))

class Trigger:
    def __init__(self,token,label):
        self.token = token
        self.label = label

    def __repr__(self):
        return self.token.__repr__()[:-1] +', '+self.label+')'

class Argument:
    def __init__(self,span,role,event_type):
        self.span = span
        self.role = role
        self.event_type = event_type

    def __repr__(self):
        return self.span.__repr__()[:-1] +', '+self.event_type +', '+self.role+')'

    def __eq__(self, other):
        return (self.span == other.span and
                self.role == other.role and
                self.event_type == other.event_type)

    def __hash__(self):
        return self.span.__hash__() + hash((self.role, self.event_type))

class Cluster:
    def __init__(self,cluster,cluster_id,document):
        members = []  # [<sent_idx>(start,end,['entity_text',...]),...]
        for entry in cluster:  # entry：一个实体的起始、结束索引位置
            sentence_ix = get_sentence_of_span(entry,document.sentence_starts,document.n_tokens)  # 返回当前实体在第几个句子  # 调用document类中的n_tokens方法，返回当前句子的长度
            sentence = document[sentence_ix]  # 拿到该句子
            span = Span(entry[0],entry[1],sentence.text,sentence.sentence_start)  # 拿到实体span
            ners = [x for x in sentence.ner if x.span == span]  # 判断拿到的span是否与句子给出的span相同
            assert len(ners) <= 1
            ner = ners[0] if len(ners) == 1 else None
            to_append = ClusterMember(span,ner,sentence,cluster_id)
            members.append(to_append)

        self.members = members
        self.cluster_id = cluster_id

    def __repr__(self):
        return f'{self.cluster_id}：' + self.members.__repr__()

    def __getitem__(self, ix):
        return self.members[ix]

class ClusterMember:
    def __init__(self,span,ner,sentence,cluster_id):
        self.span = span
        self.ner = ner
        self.sentence = sentence
        self.cluster_id = cluster_id

    def __repr__(self):
        return f'<{self.sentence.sentence_ix}>' + self.span.__repr__()  # span.__repr__() --> str((self.start_sent, self.end_sent, self.text))

































