import os
import pickle
import copy
import random
from random import sample
from omegaconf import OmegaConf

import numpy as np
import torch
from torch.utils.data import DataLoader
from fastNLP import Vocabulary
from dataset import Dataset
from dataloader import TrainDataLoader
from utils import padding, batch_padding
from datasets import load_from_disk


def _parse_data(domain_path, domain):
    dataset = load_from_disk(os.path.join(domain_path))
    tokens_key = "merged-tokens-sym"
    label_key = "label"
    neg = {
        'filename': domain,
        'data': dataset["neg"][tokens_key],
        'target': dataset["neg"][label_key]
    }
    pos = {
        'filename': domain,
        'data': dataset["pos"][tokens_key],
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
        neg_dl = DataLoader(Dataset(train_data[filename]['neg'], pad_idx), batch_size=batch_size, shuffle=True,
                            drop_last=False, **OmegaConf.to_container(config['loader']))
        pos_dl = DataLoader(Dataset(train_data[filename]['pos'], pad_idx), batch_size=batch_size, shuffle=True,
                            drop_last=False, **OmegaConf.to_container(config['loader']))
        if min(len(neg_dl), len(pos_dl)) > 0:
            train_loaders[filename] = {
                'neg': neg_dl,
                'pos': pos_dl
            }
    print('train loaders', len(train_loaders))
    return TrainDataLoader(train_loaders, support=support, query=query, pad_idx=pad_idx)


def random_select(domain, support):
    neg_indices = list(range(len(domain['neg']['data'])))
    pos_indices = list(range(len(domain['pos']['data'])))
    neg_selected_indices = sample(neg_indices, support)
    pos_selected_indices = sample(pos_indices, support)
    neg_support_data = [domain['neg']['data'][index] for index in neg_selected_indices]
    pos_support_data = [domain['pos']['data'][index] for index in pos_selected_indices]
    neg_support_target = [domain['neg']['target'][index] for index in neg_selected_indices]
    pos_support_target = [domain['pos']['target'][index] for index in pos_selected_indices]
    return neg_support_data + pos_support_data, neg_support_target + pos_support_target


def get_test_loader(full_data, support, query, pad_idx):
    loader = []
    for filename in full_data:
        # support
        support_data, support_target = random_select(full_data[filename], support)
        support_data = batch_padding(support_data, pad_idx)
        support_target = torch.tensor(support_target)
        # query
        neg_dl = DataLoader(Dataset(full_data[filename]['neg'], pad_idx), batch_size=query * 2, shuffle=False,
                            drop_last=False, **OmegaConf.to_container(config['loader']))
        pos_dl = DataLoader(Dataset(full_data[filename]['pos'], pad_idx), batch_size=query * 2, shuffle=False,
                            drop_last=False, **OmegaConf.to_container(config['loader']))
        # combine
        for dl in [neg_dl, pos_dl]:
            for batch_data, batch_target in dl:
                support_data_cp, support_target_cp = copy.deepcopy(support_data), copy.deepcopy(support_target)
                support_data_cp, batch_data = padding(support_data_cp, batch_data, pad_idx)
                data = torch.cat([support_data_cp, batch_data], dim=0)
                target = torch.cat([support_target_cp, batch_target], dim=0)
                loader.append((data, target))
    print('test loader length', len(loader))
    return loader


def main():
    data_path = config['data']['path']
    train_domains, test_domains = config['data']['train_domain'], config['data']['test_domain']

    train_data = get_train_data(data_path, train_domains)
    dev_data, test_data = get_test_data(data_path, test_domains)

    vocabulary = get_vocabulary(train_data, min_freq=config['data']['min_freq'])
    pad_idx = vocabulary.padding_idx
    pickle.dump(vocabulary, open(os.path.join(config['data']['path'], config['data']['vocabulary']), 'wb'))

    train_data = idx_all_data(train_data, vocabulary)
    dev_data = idx_all_data(dev_data, vocabulary)
    test_data = idx_all_data(test_data, vocabulary)
    # print(dev_data['books.t2.dev']['neg']['support_data'])
    # print(dev_data['books.t2.dev']['neg']['support_target'])

    support = int(config['model']['support'])
    query = int(config['model']['query'])
    train_loader = get_train_loader(train_data, support, query, pad_idx)
    dev_loader = get_test_loader(dev_data, support, query, pad_idx)
    test_loader = get_test_loader(test_data, support, query, pad_idx)

    pickle.dump(train_loader, open(os.path.join(config['data']['path'], config['data']['train_loader']), 'wb'))
    pickle.dump(dev_loader, open(os.path.join(config['data']['path'], config['data']['dev_loader']), 'wb'))
    pickle.dump(test_loader, open(os.path.join(config['data']['path'], config['data']['test_loader']), 'wb'))


if __name__ == "__main__":
    # config
    config = OmegaConf.load("sysevr_config.yaml")

    # seed
    seed = config['data']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
