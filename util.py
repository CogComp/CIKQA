import argparse
import glob
import json
import logging
import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import pathlib
from transformers import (BertConfig, BertModel, BertForMultipleChoice, BertTokenizer, RobertaConfig, RobertaModel,
                          RobertaForMultipleChoice, RobertaTokenizer)

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule

Connective_dict = {'Precedence': 'before', 'Succession': 'after', 'Synchronous': 'simultaneously', 'Reason': 'because',
                   'Result': 'so', 'Condition': 'if', 'Contrast': 'but', 'Concession': 'although',
                   'Conjunction': 'and', 'Instantiation': 'for example', 'Restatement': 'in other words',
                   'Alternative': 'or', 'ChosenAlternative': 'instead', 'Exception': 'except', 'Co_Occurrence': 'and',
                   'HasFirstSubevent': 'has first subevent', 'ReceivesAction': 'receives action',
                   'NotCapableOf': 'not capable of', 'CapableOf': 'capable of',
                   'EtymologicallyRelatedTo': 'etymologically related to', 'NotHasProperty': 'not has property',
                   'SimilarTo': 'similar to', 'MotivatedByGoal': 'motivated by', 'cause': 'so', 'Causes': 'so',
                   'UsedFor': 'used for', 'AtLocation': 'at', 'DefinedAs': 'defined as', 'RelatedTo': 'related to',
                   'HasSubevent': 'has subevent', 'HasLastSubevent': 'has last subevent',
                   'CausesDesire': 'causes desire', 'HasProperty': 'has property', 'IsA': 'is a', 'Antonym': 'antonym',
                   'dbpedia': 'dbpedia', 'DerivedFrom': 'derived from', 'FormOf': 'in the form of', 'HasA': 'has a',
                   'Synonym': 'synonym', 'HasPrerequisite': 'has prerequisite', 'NotDesires': 'not desires'}


class K2G(torch.nn.Module):
    def __init__(self, config, encoder_model, args):
        super(K2G, self).__init__()

        self.encoder = encoder_model  

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.kagnet_classifier = nn.Linear(config.hidden_size + 200, 1)
        self.GBR_classifier = nn.Linear(config.hidden_size + 200, 1)
        self.MHKA_classifier = nn.Linear(config.hidden_size*2, 1)
        self.embs = nn.Embedding(30000, 200) 
        self.lstm = nn.LSTM(200, 200)
        self.config = config
        self.args = args

    def get_baseline_logit(self, question_id, question_mask, cand_id, cand_mask):

        input_ids = torch.cat([question_id, cand_id], dim=1)
        input_mask = torch.cat([question_mask, cand_mask], dim=1)
        context_outputs = self.encoder(input_ids, attention_mask=input_mask) 
        pooled_context_output = context_outputs[0]  
        pooled_context_output = self.dropout(pooled_context_output)  
        representation = pooled_context_output[:, 0, :]  
        baseline_logit = self.classifier(representation)  
        return baseline_logit

    def get_JointI_logit(self, question_id, question_mask, cand_id, cand_mask, knowledge_ids, knowledge_mask,
                         batch_size, num_knowledge):
        input_ids = torch.cat([question_id, cand_id], dim=1).unsqueeze(1).repeat(1, num_knowledge, 1)
        input_mask = torch.cat([question_mask, cand_mask], dim=1).unsqueeze(1).repeat(1, num_knowledge, 1)

        input_ids = torch.cat([knowledge_ids, input_ids], dim=2).view(num_knowledge * batch_size, -1)
        input_mask = torch.cat([knowledge_mask, input_mask], dim=2).view(num_knowledge * batch_size, -1)

        context_outputs = self.encoder(input_ids,
                                       attention_mask=input_mask)  
        pooled_context_output = context_outputs[0] 
        pooled_context_output = self.dropout(
            pooled_context_output)  
        representation = pooled_context_output[:, 0, :]  
        JointI_logit = self.classifier(representation)  
        JointI_logit = JointI_logit.view(batch_size, -1)
        JointI_logit = torch.mean(JointI_logit, 1).unsqueeze(1)
        return JointI_logit

    def forward(
            self,
            question_ids,  
            question_mask,
            cand1_ids,
            cand1_mask,
            cand2_ids,
            cand2_mask,
            knowledge_ids,
            knowledge_mask,
            cand1_path_ids,
            cand1_path_mask,
            cand2_path_ids,
            cand2_path_mask,
            topological_path_ids,
            topological_path_mask,
            labels
    ):
        batch_size = question_ids.shape[0]
        num_knowledge = knowledge_ids.shape[1]


        if self.args.model == 'baseline':
            cand1_logit = self.get_baseline_logit(question_ids, question_mask, cand1_ids, cand1_mask)
            cand2_logit = self.get_baseline_logit(question_ids, question_mask, cand2_ids, cand2_mask)
            logits = torch.cat([cand1_logit, cand2_logit], dim=1)

        elif self.args.model == 'JointI':
            cand1_logit = self.get_JointI_logit(question_ids, question_mask, cand1_ids, cand1_mask, knowledge_ids,
                                                knowledge_mask, batch_size, num_knowledge)
            cand2_logit = self.get_JointI_logit(question_ids, question_mask, cand2_ids, cand2_mask, knowledge_ids,
                                                knowledge_mask, batch_size, num_knowledge)
            logits = torch.cat([cand1_logit, cand2_logit], dim=1)
        else:
            raise NotImplementedError



        outputs = (logits,)  

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  


class CKBQAExample(object):
    def __init__(self, qid, question, cand1, cand2, IDK, knowledge, label, helpful, q_type):
        self.qid = qid
        self.question = question
        self.cand1 = cand1
        self.cand2 = cand2
        self.IDK = IDK
        self.knowledge = knowledge
        self.label = label
        self.helpful = helpful
        self.q_type = q_type


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, qid, question_ids, question_mask, cand1_ids, cand1_mask, cand2_ids, cand2_mask, knowledge_ids,
                 knowledge_mask, cand1_path_ids, cand1_path_mask, cand2_path_ids, cand2_path_mask, topological_path_ids,
                 topological_path_mask, label_id, useful_label_id):
        self.qid = qid
        self.question_ids = question_ids
        self.question_mask = question_mask
        self.cand1_ids = cand1_ids
        self.cand1_mask = cand1_mask
        self.cand2_ids = cand2_ids
        self.cand2_mask = cand2_mask
        self.knowledge_ids = knowledge_ids
        self.knowledge_mask = knowledge_mask
        self.cand1_path_ids = cand1_path_ids
        self.cand1_path_mask = cand1_path_mask
        self.cand2_path_ids = cand2_path_ids
        self.cand2_path_mask = cand2_path_mask
        self.topological_path_ids = topological_path_ids
        self.topological_path_mask = topological_path_mask
        self.label_id = label_id
        self.useful_label_id = useful_label_id


class CKBQADataLoader:
    def __init__(self, args, data_folder, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.train_data = self.get_examples(data_folder + '/train.json', args.train_type, args.helpful_only,
                                            args.train_number)
        self.dev_data = self.get_examples(data_folder + '/dev.json', args.test_type, args.helpful_only)
        self.test_data = self.get_examples(data_folder + '/test.json', args.test_type, args.helpful_only)
        self.train_features = convert_examples_to_features(args, self.train_data, args.max_seq_length, tokenizer)
        self.dev_features = convert_examples_to_features(args, self.dev_data, args.max_seq_length, tokenizer)
        self.test_features = convert_examples_to_features(args, self.test_data, args.max_seq_length, tokenizer)
        self.train_set = self.get_dataset(self.train_features)
        self.dev_set = self.get_dataset(self.dev_features)
        self.test_set = self.get_dataset(self.test_features)

    def get_examples(self, file_path, type_requirement, helpful_only, example_lim=100000):
        with open(file_path, 'r') as f:
            records = json.load(f)
        random.shuffle(records)
        examples = list()
        for i, record in enumerate(records):
            if record['label'] == 2:
                helpful_state = False
            else:
                helpful_state = True
            tmp_example = CKBQAExample(
                qid=record['idx'],
                question=record['question'],
                cand1=record['answers'][0],
                cand2=record['answers'][1],
                IDK=record['answers'][2],
                knowledge=record['knowledge'],
                label=record['sourcelabel'],
                helpful=helpful_state,
                q_type=record['source']
            )
            if helpful_only:
                if record['label'] == 2:
                    continue
            if type_requirement == 'All':
                examples.append(tmp_example)
            else:
                if record['source'] == type_requirement:
                    examples.append(tmp_example)
        random.shuffle(examples)
        return examples[:example_lim]

    def get_dataset(self, tensorized_dataset):

        all_question_ids = torch.tensor([f.question_ids for f in tensorized_dataset], dtype=torch.long)
        all_question_mask = torch.tensor([f.question_mask for f in tensorized_dataset], dtype=torch.long)
        all_cand1_ids = torch.tensor([f.cand1_ids for f in tensorized_dataset], dtype=torch.long)
        all_cand1_mask = torch.tensor([f.cand1_mask for f in tensorized_dataset], dtype=torch.long)
        all_cand2_ids = torch.tensor([f.cand2_ids for f in tensorized_dataset], dtype=torch.long)
        all_cand2_mask = torch.tensor([f.cand2_mask for f in tensorized_dataset], dtype=torch.long)
        all_knowledge_ids = torch.tensor([f.knowledge_ids for f in tensorized_dataset], dtype=torch.long)
        all_knowledge_mask = torch.tensor([f.knowledge_mask for f in tensorized_dataset], dtype=torch.long)
        all_cand1_path_ids = torch.tensor([f.cand1_path_ids for f in tensorized_dataset], dtype=torch.long)
        all_cand1_path_mask = torch.tensor([f.cand1_path_mask for f in tensorized_dataset], dtype=torch.long)
        all_cand2_path_ids = torch.tensor([f.cand2_path_ids for f in tensorized_dataset], dtype=torch.long)
        all_cand2_path_mask = torch.tensor([f.cand2_path_mask for f in tensorized_dataset], dtype=torch.long)
        all_topological_path_ids = torch.tensor([f.topological_path_ids for f in tensorized_dataset], dtype=torch.long)
        all_topological_path_mask = torch.tensor([f.topological_path_mask for f in tensorized_dataset],
                                                 dtype=torch.long)

        all_label_ids = torch.tensor([f.label_id for f in tensorized_dataset], dtype=torch.long)
        all_useful_label_ids = torch.tensor([f.useful_label_id for f in tensorized_dataset], dtype=torch.long)

        return TensorDataset(all_question_ids, all_question_mask, all_cand1_ids, all_cand1_mask, all_cand2_ids,
                             all_cand2_mask, all_knowledge_ids, all_knowledge_mask, all_cand1_path_ids,
                             all_cand1_path_mask, all_cand2_path_ids, all_cand2_path_mask, all_topological_path_ids,
                             all_topological_path_mask, all_label_ids, all_useful_label_ids)


def tensorize_a_sentence(sentence, max_seq_length, tokenizer, cls_token='[CLS]', sep_token='[SEP]', pad_token=0):
    input_ids = [tokenizer.convert_tokens_to_ids(cls_token)]
    sentence_tokens = tokenizer.tokenize(sentence)
    for t in sentence_tokens:
        input_ids.append(tokenizer.convert_tokens_to_ids(t))
    input_ids += [tokenizer.convert_tokens_to_ids(sep_token)]
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
    input_mask = [1] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0] * padding_length)
    return input_ids, input_mask


def get_path(knowledge, start_node, end_node):
    cand_paths = list()

    for tmp_edge in knowledge:
        if tmp_edge[0].split('$$')[1] == start_node:
            current_path = list()
            current_path.append(tmp_edge)
            last_edge = tmp_edge
            for new_edge in knowledge:
                if new_edge[0] == last_edge[1]:
                    current_path.append(new_edge)
                if new_edge[1].split('$$')[1] == end_node:
                    break
                last_edge = new_edge
            if current_path[-1][1].split('$$')[1] == end_node:
                cand_paths.append(current_path)
    return cand_paths[:1]


def weighted_random_walk(knowledge, num_sample, length_limit):
    head2tail = dict()
    for tmp_edge in knowledge:
        if tmp_edge[0] not in head2tail:
            head2tail[tmp_edge[0]] = list()
        if tmp_edge[1] not in head2tail:
            head2tail[tmp_edge[1]] = list()
        head2tail[tmp_edge[0]].append((tmp_edge[1], tmp_edge[2], tmp_edge[3]))
    paths = list()
    for i in range(num_sample):
        current_node = random.choice(list(head2tail.keys()))
        tmp_path = []
        while True:
            if len(tmp_path) >= length_limit:
                break
            if len(head2tail[current_node]) == 0:
                break
            tmp_weights = list()
            for tmp_next_n in head2tail[current_node]:
                tmp_weights.append(tmp_next_n[2])
            next_n = random.choices(head2tail[current_node], weights=tmp_weights, k=1)[0]
            # print(next_n)
            tmp_path.append([current_node, next_n[0], next_n[1], next_n[2]])
            current_node = next_n[0]
        paths.append(tmp_path)
    return paths


def topological_random_walk(knowledge, num_sample, length_limit):
    head2tail = dict()
    for tmp_edge in knowledge:
        if tmp_edge[0] not in head2tail:
            head2tail[tmp_edge[0]] = list()
        if tmp_edge[1] not in head2tail:
            head2tail[tmp_edge[1]] = list()
        head2tail[tmp_edge[0]].append((tmp_edge[1], tmp_edge[2], tmp_edge[3]))
    paths = list()
    for i in range(num_sample):
        current_node = random.choice(list(head2tail.keys()))
        tmp_path = []
        while True:
            if len(tmp_path) >= length_limit:
                break
            if len(head2tail[current_node]) == 0:
                break
            next_n = random.choice(head2tail[current_node])
            tmp_path.append([current_node, next_n[0], next_n[1], next_n[2]])
            current_node = next_n[0]
        paths.append(tmp_path)
    return paths


def path_to_feature(tmp_path, max_seq_length, tokenizer, cls_token='[CLS]', sep_token='[SEP]',
                    pad_token=0):
    tmp_knowledge_sentence = ''
    for tmp_edge in tmp_path:
        tmp_sentence = tmp_edge[0].split('$$')[0] + ', ' + Connective_dict[tmp_edge[2]] + tmp_edge[1].split('$$')[
            0] + '. ' + str(
            tmp_edge[3]) + sep_token
        tmp_knowledge_sentence += tmp_sentence

    tmp_knowledge_ids, tmp_knowledge_mask = tensorize_a_sentence(tmp_knowledge_sentence, max_seq_length, tokenizer,
                                                                 cls_token,
                                                                 sep_token, pad_token)
    return tmp_knowledge_ids, tmp_knowledge_mask


def convert_examples_to_features(args, examples, max_seq_length, tokenizer, cls_token='[CLS]', sep_token='[SEP]',
                                 pad_token=0):

    features = []
    for (ex_index, example) in enumerate(examples):
        question_ids, question_mask = tensorize_a_sentence(example.question, max_seq_length, tokenizer, cls_token,
                                                           sep_token, pad_token)
        cand1_ids, cand1_mask = tensorize_a_sentence(example.cand1, max_seq_length, tokenizer, cls_token,
                                                     sep_token, pad_token)
        cand2_ids, cand2_mask = tensorize_a_sentence(example.cand2, max_seq_length, tokenizer, cls_token,
                                                     sep_token, pad_token)
        cand1_paths = get_path(example.knowledge, 'question', 'CandidateA')
        cand2_paths = get_path(example.knowledge, 'question', 'CandidateB')
        random_walk_paths = weighted_random_walk(example.knowledge, args.num_walk, args.walk_length)
        topological_walk_paths = topological_random_walk(example.knowledge, args.num_walk, args.walk_length)
        knowledge_ids = list()
        knowledge_mask = list()
        for tmp_walk in random_walk_paths:
            tmp_knowledge_ids, tmp_knowledge_mask = path_to_feature(tmp_walk, max_seq_length, tokenizer, cls_token,
                                                                    sep_token, 0)
            knowledge_ids.append(tmp_knowledge_ids)
            knowledge_mask.append(tmp_knowledge_mask)
        cand1_path_ids = list()
        cand1_path_mask = list()
        if len(cand1_paths) == 0:
            tmp_knowledge_ids, tmp_knowledge_mask = tensorize_a_sentence('NA', max_seq_length, tokenizer,
                                                                 cls_token,
                                                                 sep_token, pad_token)
            cand1_path_ids.append(tmp_knowledge_ids)
            cand1_path_mask.append(tmp_knowledge_mask)
        else:
            for tmp_walk in cand1_paths:
                tmp_knowledge_ids, tmp_knowledge_mask = path_to_feature(tmp_walk, max_seq_length, tokenizer, cls_token,
                                                                        sep_token, 0)
                cand1_path_ids.append(tmp_knowledge_ids)
                cand1_path_mask.append(tmp_knowledge_mask)
        cand2_path_ids = list()
        cand2_path_mask = list()
        if len(cand2_paths) == 0:
            tmp_knowledge_ids, tmp_knowledge_mask = tensorize_a_sentence('NA', max_seq_length, tokenizer,
                                                                 cls_token,
                                                                 sep_token, pad_token)
            cand2_path_ids.append(tmp_knowledge_ids)
            cand2_path_mask.append(tmp_knowledge_mask)
        else:
            for tmp_walk in cand2_paths:
                tmp_knowledge_ids, tmp_knowledge_mask = path_to_feature(tmp_walk, max_seq_length, tokenizer, cls_token,
                                                                        sep_token, 0)
                cand2_path_ids.append(tmp_knowledge_ids)
                cand2_path_mask.append(tmp_knowledge_mask)
        topological_path_ids = list()
        topological_path_mask = list()
        for tmp_walk in topological_walk_paths:
            tmp_knowledge_ids, tmp_knowledge_mask = path_to_feature(tmp_walk, max_seq_length, tokenizer, cls_token,
                                                                    sep_token, 0)
            topological_path_ids.append(tmp_knowledge_ids)
            topological_path_mask.append(tmp_knowledge_mask)

        if example.helpful:
            useful_id = 0
        else:
            useful_id = 1
        features.append(
            InputFeatures(
                qid=example.qid,
                question_ids=question_ids,
                question_mask=question_mask,
                cand1_ids=cand1_ids,
                cand1_mask=cand1_mask,
                cand2_ids=cand2_ids,
                cand2_mask=cand2_mask,
                knowledge_ids=knowledge_ids,
                knowledge_mask=knowledge_mask,
                cand1_path_ids=cand1_path_ids,
                cand1_path_mask=cand1_path_mask,
                cand2_path_ids=cand2_path_ids,
                cand2_path_mask=cand2_path_mask,
                topological_path_ids=topological_path_ids,
                topological_path_mask=topological_path_mask,
                label_id=example.label,
                useful_label_id=useful_id
            )
        )
    return features


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
