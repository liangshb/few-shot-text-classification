import os
import pickle
import copy
import random
from random import sample
from omegaconf import OmegaConf

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from fastNLP import Vocabulary
from dataset import LenDataset
from dataloader import TrainDataLoader
from utils import padding, collate_fn
from datasets import load_from_disk


def _parse_data(domain_path, domain):
    dataset = load_from_disk(os.path.join(domain_path))
    tokens_key = "merged-tokens-sym"
    length_key = "length"
    label_key = "label"
    neg = {
        'filename': domain,
        'data': dataset["neg"][tokens_key],
        'length': dataset["neg"][length_key],
        'target': dataset["neg"][label_key]
    }
    pos = {
        'filename': domain,
        'data': dataset["pos"][tokens_key],
        'length': dataset["pos"][length_key],
        'target': dataset["pos"][label_key]
    }
    # check
    print(domain, 'neg', len(neg['data']), 'pos', len(pos['data']))
    return neg, pos


def _get_data(data_path, domains):
    data = {}
    for domain in domains:
        domain_path = os.path.join(data_path, domain)
        neg, pos = _parse_data(domain_path, domain)
        data[domain] = {'neg': neg, 'pos': pos}
    return data


def get_train_data(data_path, domains):
    train_data = _get_data(data_path, domains)
    print('train data', len(train_data))
    return train_data


def get_test_data(data_path, domains):
    # get dev, test data
    dev_data = _get_data(data_path, domains)
    test_data = _get_data(data_path, domains)

    print('dev data', len(dev_data), 'test data', len(test_data))
    return dev_data, test_data


def get_vocabulary(data, min_freq):
    # train data -> vocabulary
    vocabulary = Vocabulary(min_freq=min_freq, padding='<pad>', unknown='<unk>')
    for filename in data:
        for value in data[filename]:
            for word_list in data[filename][value]['data']:
                vocabulary.add_word_lst(word_list)
    vocabulary.build_vocab()
    print('vocab size', len(vocabulary), 'pad', vocabulary.padding_idx, 'unk', vocabulary.unknown_idx)
    return vocabulary


def _idx_text(text_list, vocabulary):
    for i in range(len(text_list)):
        for j in range(len(text_list[i])):
            text_list[i][j] = vocabulary.to_index(text_list[i][j])
    return text_list


def idx_all_data(data, vocabulary):
    for filename in data:
        for value in data[filename]:
            for key in data[filename][value]:
                if key in ['data', 'support_data']:
                    data[filename][value][key] = _idx_text(data[filename][value][key], vocabulary)
    return data


def get_train_loader(train_data, support, query, pad_idx):
    batch_size = support + query
    train_loaders = {}

    for filename in train_data:
        neg_dl = DataLoader(LenDataset(train_data[filename]['neg']), batch_size=batch_size, shuffle=True,
                            drop_last=False,
                            collate_fn=collate_fn,
                            **OmegaConf.to_container(config['loader']))
        pos_dl = DataLoader(LenDataset(train_data[filename]['pos']), batch_size=batch_size, shuffle=True,
                            drop_last=False,
                            collate_fn=collate_fn,
                            **OmegaConf.to_container(config['loader']))
        if min(len(neg_dl), len(pos_dl)) > 0:
            train_loaders[filename] = {
                'neg': neg_dl,
                'pos': pos_dl
            }
    print('train loaders', len(train_loaders))
    return TrainDataLoader(train_loaders, support=support, query=query, pad_idx=pad_idx)


def random_select(domain, support):
    neg_support_data = []
    pos_support_data = []
    neg_support_length = []
    pos_support_length = []
    neg_support_target = []
    pos_support_target = []
    for _ in range(support):
        neg_indices = range(len(domain['neg']['data']))
        pos_indices = range(len(domain['pos']['data']))
        neg_selected_index = random.choice(neg_indices)
        pos_selected_index = random.choice(pos_indices)
        neg_support_data.append(domain["neg"]["data"].pop(neg_selected_index))
        pos_support_data.append(domain["pos"]["data"].pop(pos_selected_index))
        neg_support_length.append(domain["neg"]["length"].pop(neg_selected_index))
        pos_support_length.append(domain['pos']['length'].pop(pos_selected_index))
        neg_support_target.append(domain['neg']['target'].pop(neg_selected_index))
        pos_support_target.append(domain['pos']['target'].pop(pos_selected_index))
    return neg_support_data + pos_support_data, neg_support_length + pos_support_length, neg_support_target + pos_support_target


def get_test_loader(full_data, support, query, pad_idx):
    loader = []
    for filename in full_data:
        # support
        support_data, support_length, support_target = random_select(full_data[filename], support)
        support_data = pad_sequence([torch.tensor(data) for data in support_data], batch_first=True,
                                    padding_value=pad_idx)
        support_length = torch.tensor(support_length)
        support_target = torch.tensor(support_target)
        # query
        neg_dl = DataLoader(LenDataset(full_data[filename]['neg']), batch_size=query * 2, shuffle=False,
                            drop_last=False,
                            collate_fn=collate_fn,
                            **OmegaConf.to_container(config['loader']))
        pos_dl = DataLoader(LenDataset(full_data[filename]['pos']), batch_size=query * 2, shuffle=False,
                            drop_last=False,
                            collate_fn=collate_fn,
                            **OmegaConf.to_container(config['loader']))
        # combine
        for dl in [neg_dl, pos_dl]:
            for batch_data, batch_length, batch_target in dl:
                support_data_cp, support_length_cp, support_target_cp = copy.deepcopy(support_data), copy.deepcopy(
                    support_length), copy.deepcopy(support_target)
                support_data_cp, batch_data = padding(support_data_cp, batch_data, pad_idx)
                data = torch.cat([support_data_cp, batch_data], dim=0)
                length = torch.cat([support_length_cp, batch_length], dim=0)
                target = torch.cat([support_target_cp, batch_target], dim=0)
                loader.append((data, length, target))
    print('test loader length', len(loader))
    return loader


def main():
    data_path = config['data']['path']
    save_path = os.path.join(data_path, config["data"]["save_path"])
    os.makedirs(save_path, exist_ok=True)
    vocabulary_path = os.path.join(config['data']['path'], config['data']['vocabulary'])
    train_loader_path = os.path.join(config['data']['path'], config['data']['train_loader'])
    dev_loader_path = os.path.join(config['data']['path'], config['data']['dev_loader'])
    test_loader_path = os.path.join(config['data']['path'], config['data']['test_loader'])
    train_domains, test_domains = config['data']['train_domain'], config['data']['test_domain']

    train_data = get_train_data(data_path, train_domains)
    dev_data, test_data = get_test_data(data_path, test_domains)

    if os.path.exists(vocabulary_path):
        print("Loading vocabulary")
        vocabulary = pickle.load(open(vocabulary_path, "rb"))
    else:
        vocabulary = get_vocabulary(train_data, min_freq=config['data']['min_freq'])
        pickle.dump(vocabulary, open(vocabulary_path, 'wb'))
    pad_idx = vocabulary.padding_idx

    support = int(config['model']['support'])
    query = int(config['model']['query'])
    if not os.path.exists(train_loader_path):
        train_data = idx_all_data(train_data, vocabulary)
        train_loader = get_train_loader(train_data, support, query, pad_idx)
        pickle.dump(train_loader, open(train_loader_path, 'wb'))
    else:
        print("Train loader exists")
    if not os.path.exists(dev_loader_path):
        dev_data = idx_all_data(dev_data, vocabulary)
        dev_loader = get_test_loader(dev_data, support, query, pad_idx)
        pickle.dump(dev_loader, open(dev_loader_path, 'wb'))
    else:
        print("Dev loader exists")
    if not os.path.exists(test_loader_path):
        test_data = idx_all_data(test_data, vocabulary)
        test_loader = get_test_loader(test_data, support, query, pad_idx)
        pickle.dump(test_loader, open(test_loader_path, 'wb'))
    else:
        print("Test loader exists")


if __name__ == "__main__":
    # config
    config = OmegaConf.load("sysevr_config_cnn.yaml")

    # seed
    seed = config['data']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
