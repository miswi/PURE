task_ner_labels = {
    'scierc':['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric']
}

task_rel_labels = {
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i,label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i+1] = label
    return label2id,id2label

